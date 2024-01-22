import torch
import torch.nn as nn
from torch.nn.functional import tanh, relu
from gymnasium.spaces import MultiDiscrete, Box
from typing import Tuple


class ActionLayer(nn.Module):
    def __init__(self, input_dim: int, out_dim: int, max_dim: int):
        super(ActionLayer, self).__init__()
        self.layer = nn.Linear(input_dim, out_dim)
        self.max_dim = max_dim

    def forward(self, x: torch.Tensor):
        output_layer = self.layer(x)
        output = torch.full((x.shape[0], self.max_dim,), -1e32, dtype=output_layer.dtype,
                            device=output_layer.device)
        output[:output_layer.size(0), :output_layer.size(1)] = output_layer
        return output


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

    def __init__(self, observation_space: Box, action_space: MultiDiscrete, objectives: int = 1,
                 use_accrued_reward: bool = False, global_layers: Tuple[int] = (50,),
                 local_layers: Tuple[int] = (50,), activation: nn.Module = nn.Tanh):
        super(PNet, self).__init__()

        if use_accrued_reward:
            reward_size = objectives
        else:
            reward_size = 0
        self.input_dim = observation_space.shape[0] + reward_size
        last_dim = self.input_dim
        shared_layers_list = []
        for layer_dim in global_layers:
            shared_layers_list.append(nn.Linear(last_dim, layer_dim))
            shared_layers_list.append(activation())
            last_dim = layer_dim
        action_sizes = action_space.nvec
        max_action = max(action_sizes)
        self.shared = nn.Sequential(*shared_layers_list)
        last_shared_layer_dim = last_dim
        module_list = []
        for act in action_sizes:
            last_dim = last_shared_layer_dim
            head_layers_list = []
            for layer_dim in local_layers:
                head_layers_list.append(nn.Linear(last_dim, layer_dim))
                head_layers_list.append(activation())
                last_dim = layer_dim
            head_layers_list.append(ActionLayer(last_dim, act, max_action))
            module_list.append(nn.Sequential(*head_layers_list))
        self.heads = nn.ModuleList(module_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_global = self.shared(x)
        out = torch.stack([head(out_global) for head in self.heads])
        return out


# class PNet(nn.Module):
#     """
#     Policy network
#
#     Parameters
#     ----------
#     ncomp: int
#         Number of components in the system.
#     nstcomp: int
#         Number of states for each component.
#     nacomp: int
#         Number of actions for each component.
#     naglobal: int
#         Number of global actions.
#     objectives: int
#         Number of objectives. Default: 1
#     use_accrued_reward: bool
#         Whether to use accrued reward as input. Default: False
#     """
#
#     def __init__(self, ncomp: int, nstcomp: int, nacomp: int, naglobal: int, objectives: int = 1,
#                  use_accrued_reward: bool = False):
#         super(PNet, self).__init__()
#         self.input_dim = ncomp * nstcomp + 1
#         self.output_dim = nacomp
#         self.naglobal = naglobal
#         self.diff_actions = nstcomp - nacomp
#         if use_accrued_reward:
#             reward_size = objectives
#         else:
#             reward_size = 0
#         self.fc1 = nn.Linear(self.input_dim + reward_size, 50)
#         module_list = []
#         for i in range(ncomp):
#             module_list.append(nn.Sequential(
#                 nn.Linear(50, 50),
#                 nn.Tanh(),
#                 nn.Linear(50, nacomp)
#             ))
#         self.global_head = nn.Sequential(
#             nn.Linear(50, 50),
#             nn.Tanh(),
#             nn.Linear(50, naglobal)
#         )
#         self.fc_outs = nn.ModuleList(module_list)
#         self.relu = nn.Tanh()
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = tanh(self.fc1(x))
#         out_comp = torch.stack([fc(x) for fc in self.fc_outs])
#         out_global = self.global_head(x).view(1, x.shape[0], self.naglobal)
#         inf_tensor = torch.full((1, x.shape[0], 1), -float("inf"))
#         out_global = torch.cat((out_global, inf_tensor), dim=2)
#         out = torch.cat((out_comp, out_global), dim=0)
#         return out


class PNetMobile(nn.Module):
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

    def __init__(self, input_size: int, n_actions: int, n_heads: int, objectives: int = 1,
                 use_accrued_reward: bool = False):
        super(PNetMobile, self).__init__()
        self.input_dim = input_size
        self.output_dim = n_actions
        self.n_heads = n_heads
        if use_accrued_reward:
            reward_size = objectives
        else:
            reward_size = 0
        self.fc1 = nn.Linear(self.input_dim + reward_size, 128)
        module_list = []
        for i in range(self.n_heads):
            module_list.append(nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, self.output_dim)
            ))
        self.fc_outs = nn.ModuleList(module_list)
        self.relu = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = relu(self.fc1(x))
        out = torch.stack([fc(x) for fc in self.fc_outs])
        return out


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

    def __init__(self, observation_space: Box, c: int = 11, objectives: int = 1, use_accrued_reward: bool = False,
                 activation: nn.Module = nn.Tanh, hidden_layers: Tuple[int] = (50, 50, 50)):
        super(VNet, self).__init__()
        self.c = c
        self.objectives = objectives
        self.activation = activation
        if use_accrued_reward:
            reward_size = objectives
        else:
            reward_size = 0
        self.input_dim = observation_space.shape[0] + reward_size
        last_dim = self.input_dim
        layers_list = []
        layers_list.append(nn.Linear(last_dim, hidden_layers[0]))
        layers_list.append(activation())
        last_dim = hidden_layers[0]
        for i in range(1, len(hidden_layers)):
            layer_dim = hidden_layers[i]
            layers_list.append(nn.Linear(last_dim, layer_dim))
            layers_list.append(activation())
            last_dim = layer_dim
        self.out_dim = self.c ** self.objectives
        layers_list.append(nn.Linear(last_dim, self.out_dim))
        layers_list.append(nn.Softmax(dim=1))
        self.layers = nn.Sequential(*layers_list)

        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x.view(-1, *([self.c] * self.objectives))


class VNetMobile(nn.Module):
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

    def __init__(self, input_size, c: int = 11, objectives: int = 1, use_accrued_reward: bool = False,
                 activation: nn.Module = nn.Tanh):
        super(VNetMobile, self).__init__()
        self.c = c
        self.objectives = objectives
        self.input_dim = input_size
        self.activation = activation
        if use_accrued_reward:
            reward_size = objectives
        else:
            reward_size = 0
        self.fc1 = nn.Linear(self.input_dim + reward_size, 128)
        self.activation1 = self.activation()
        self.fc2 = nn.Linear(128, 64)
        self.activation2 = self.activation()

        self.fc_out = nn.Linear(64, self.c ** self.objectives)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation1(self.fc1(x))
        x = self.activation2(self.fc2(x))

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


class ContinuousHead(nn.Module):
    def __init__(self, input_dim: int, beta: int = 1, threshold: int = 20, min_action: float = -1,
                 max_action: float = 1):
        super().__init__()
        assert min_action < max_action, "min_reward should be lower then max_reward"
        self.input_dim = input_dim
        self.beta = beta
        self.threshold = threshold
        self.mu = nn.Sequential(
            nn.Linear(self.input_dim, 1),
            nn.Tanh()
        )
        self.var = nn.Sequential(
            nn.Linear(self.input_dim, 1),
            nn.Softplus(beta=self.beta, threshold=self.threshold)
        )

    def forward(self, x: torch.Tensor):
        mu_out = self.mu(x)
        var_out = self.var(x)
        out = torch.cat((mu_out, var_out), dim=1)
        return out


class PNetContinuous(nn.Module):
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

    def __init__(self, input_size: int, n_heads: int, objectives: int = 1, use_accrued_reward: bool = False,
                 activation: nn.Module = nn.Tanh):
        super(PNetContinuous, self).__init__()
        self.input_dim = input_size
        self.n_heads = n_heads
        self.activation = activation
        if use_accrued_reward:
            reward_size = objectives
        else:
            reward_size = 0
        self.fc1 = nn.Linear(self.input_dim + reward_size, 128)
        self.activation1 = self.activation()
        module_list = []
        for i in range(self.n_heads):
            module_list.append(nn.Sequential(
                nn.Linear(128, 64),
                self.activation(),
                ContinuousHead(64)
            ))
        self.fc_outs = nn.ModuleList(module_list)
        self.relu = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation1(self.fc1(x))
        out = torch.stack([fc(x) for fc in self.fc_outs])
        return out
