import torch
import torch.nn as nn
from torch.nn.functional import tanh, relu
from typing import Tuple


class PNet(nn.Module):
    """
    Policy network

    Parameters
    ----------
    ncomp: int
        Number of components in the system.
    nstcomp: int
        Number of states for each component.
    nacomp: int
        Number of actions for each component.
    naglobal: int
        Number of global actions.
    objectives: int
        Number of objectives. Default: 1
    use_accrued_reward: bool
        Whether to use accrued reward as input. Default: False
    """

    def __init__(self, ncomp: int, nstcomp: int, nacomp: int, naglobal: int, objectives: int = 1,
                 use_accrued_reward: bool = False):
        super(PNet, self).__init__()
        self.input_dim = ncomp * nstcomp + 1
        self.output_dim = nacomp
        self.naglobal = naglobal
        if use_accrued_reward:
            reward_size = objectives
        else:
            reward_size = 0
        self.fc1 = nn.Linear(self.input_dim + reward_size, 100)
        module_list = []
        for i in range(ncomp):
            module_list.append(nn.Sequential(
                nn.Linear(100, 100),
                nn.Tanh(),
                nn.Linear(100, nacomp)
            ))
        self.global_head = nn.Sequential(
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, naglobal)
        )
        self.fc_outs = nn.ModuleList(module_list)
        self.relu = nn.Tanh()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = tanh(self.fc1(x))
        out_comp = torch.stack([fc(x) for fc in self.fc_outs])
        out_global = self.global_head(x)
        return out_comp, out_global


class VNetSERCat(nn.Module):
    """
    Value network

    Parameters
    ----------
    ncomp: int
        Number of components in the system.
    nstcomp: int
        Number of states for each component.
    c: int
        Number of bins for the critic output. Default: 11
    use_accrued_reward: bool
        Whether to use accrued reward as input. Default: False
    """

    def __init__(self, ncomp: int, nstcomp: int, c: int = 11, objectives: int = 1, use_accrued_reward: bool = False):
        super(VNetSERCat, self).__init__()
        self.c = c
        self.objectives = objectives
        if use_accrued_reward:
            reward_size = objectives
        else:
            reward_size = 0
        self.fc1 = nn.Linear(ncomp * nstcomp + 1 + reward_size, 100)
        self.fc2 = nn.Linear(100, 100)

        self.fc_out = nn.Linear(100, self.c * self.objectives)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))

        x = self.fc_out(x)
        x = self.softmax(x.view(-1, self.objectives, self.c))

        return x


def weight_init(layers):
    for layer in layers:
        torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')


class VNetSERQR(nn.Module):
    """
    Value network

    Parameters
    ----------
    ncomp: int
        Number of components in the system.
    nstcomp: int
        Number of states for each component.
    c: int
        Number of bins for the critic output. Default: 11
    use_accrued_reward: bool
        Whether to use accrued reward as input. Default: False
    """

    def __init__(self, ncomp: int, nstcomp: int, c: int = 11, objectives: int = 1, use_accrued_reward: bool = False):
        super(VNetSERQR, self).__init__()
        self.c = c
        self.objectives = objectives
        if use_accrued_reward:
            reward_size = objectives
        else:
            reward_size = 0
        self.fc1_obj1 = nn.Linear(ncomp * nstcomp + 1 + reward_size, 100)
        self.fc2_obj1 = nn.Linear(100, 100)

        self.fc_out_obj1 = nn.Linear(100, self.c)

        self.fc1_obj2 = nn.Linear(ncomp * nstcomp + 1 + reward_size, 100)
        self.fc2_obj2 = nn.Linear(100, 100)

        self.fc_out_obj2 = nn.Linear(100, self.c)
        # self.softmax = nn.Softmax(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_obj1 = relu(self.fc1_obj1(x))
        x_obj1 = relu(self.fc2_obj1(x_obj1))

        x_obj1 = self.fc_out_obj1(x_obj1)

        x_obj2 = relu(self.fc1_obj2(x))
        x_obj2 = relu(self.fc2_obj2(x_obj2))

        x_obj2 = self.fc_out_obj2(x_obj2)
        # print(x_obj1.shape, x_obj2.shape)
        x = torch.stack([x_obj1, x_obj2], dim=1)
        # print(x.shape)
        # exit()
        # x = x.view(-1, self.objectives, self.c)

        return x


class VNetSERQROld(nn.Module):
    """
    Value network

    Parameters
    ----------
    ncomp: int
        Number of components in the system.
    nstcomp: int
        Number of states for each component.
    c: int
        Number of bins for the critic output. Default: 11
    use_accrued_reward: bool
        Whether to use accrued reward as input. Default: False
    """

    def __init__(self, ncomp: int, nstcomp: int, c: int = 11, objectives: int = 1, use_accrued_reward: bool = False):
        super(VNetSERQROld, self).__init__()
        self.c = c
        self.objectives = objectives
        if use_accrued_reward:
            reward_size = objectives
        else:
            reward_size = 0
        self.fc1 = nn.Linear(ncomp * nstcomp + 1 + reward_size, 100)
        self.fc2 = nn.Linear(100, 100)

        self.fc_out = nn.Linear(100, self.c * self.objectives)
        # self.softmax = nn.Softmax(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))

        x = self.fc_out(x)
        x = x.view(-1, self.objectives, self.c)

        return x


class VNet(nn.Module):
    """
    Value network

    Parameters
    ----------
    ncomp: int
        Number of components in the system.
    nstcomp: int
        Number of states for each component.
    c: int
        Number of bins for the critic output. Default: 11
    use_accrued_reward: bool
        Whether to use accrued reward as input. Default: False
    """

    def __init__(self, ncomp: int, nstcomp: int, c: int = 11, objectives: int = 1, use_accrued_reward: bool = False):
        super(VNet, self).__init__()
        self.c = c
        self.objectives = objectives
        if use_accrued_reward:
            reward_size = objectives
        else:
            reward_size = 0
        self.fc1 = nn.Linear(ncomp * nstcomp + 1 + reward_size, 100)
        self.fc2 = nn.Linear(100, 100)

        self.fc_out = nn.Linear(100, self.c ** self.objectives)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = tanh(self.fc1(x))
        x = tanh(self.fc2(x))

        x = self.fc_out(x)
        x = self.softmax(x)
        return x.view(-1, *([self.c] * self.objectives))


class VNetSER(nn.Module):
    """
    Value network

    Parameters
    ----------
    ncomp: int
        Number of components in the system.
    nstcomp: int
        Number of states for each component.
    c: int
        Number of bins for the critic output. Default: 11
    use_accrued_reward: bool
        Whether to use accrued reward as input. Default: False
    """

    def __init__(self, ncomp: int, nstcomp: int, c: int = 11, objectives: int = 1, use_accrued_reward: bool = False):
        super(VNetSER, self).__init__()
        self.c = c
        self.objectives = objectives
        if use_accrued_reward:
            reward_size = objectives
        else:
            reward_size = 0
        self.fc1 = nn.Linear(ncomp * nstcomp + 1 + reward_size, 100)
        self.fc2 = nn.Linear(100, 100)

        self.fc_out = nn.Linear(100, self.objectives)
        self.activation = relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))

        x = self.fc_out(x)

        return x

# class VNetDCMAC(nn.Module):
#     """
#     Value network
#
#     Parameters
#     ----------
#     ncomp: int
#         Number of components in the system.
#     nstcomp: int
#         Number of states for each component.
#     c: int
#         Number of bins for the critic output. Default: 11
#     use_accrued_reward: bool
#         Whether to use accrued reward as input. Default: False
#     """
#
#     def __init__(self, ncomp: int, nstcomp: int, c: int = 11, objectives: int = 1, use_accrued_reward: bool = False):
#         super(VNetDCMAC, self).__init__()
#         self.c = c
#         self.objectives = objectives
#         if use_accrued_reward:
#             reward_size = objectives
#         else:
#             reward_size = 0
#         self.fc1 = nn.Linear(ncomp * nstcomp + 1 + reward_size, 128)
#         # self.fc2 = nn.Linear(100, 100)
#         module_list = []
#         for i in range(self.objectives):
#             module_list.append(nn.Sequential(
#                 nn.Linear(128, 64),
#                 nn.Tanh(),
#                 nn.Linear(64, 1)
#             ))
#         self.fc_outs = nn.ModuleList(module_list)
#         # self.fc_out = nn.Linear(100, self.objectives)
#         self.activation = tanh
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.activation(self.fc1(x))
#         x = torch.stack([fc(x) for fc in self.fc_outs], dim=1).squeeze()
#         # print(x.shape)
#
#         return x
