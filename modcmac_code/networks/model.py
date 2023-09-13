import torch
import torch.nn as nn
from torch.nn.functional import tanh
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
