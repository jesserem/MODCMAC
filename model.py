import torch
import torch.nn as nn
from torch.nn.functional import tanh, relu


class PNet(nn.Module):
    def __init__(self, ncomp, nstcomp, nacomp, naglobal, objectives=1, use_accrued_reward=False):
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

    def forward(self, x):

        x = tanh(self.fc1(x))

        out_comp = torch.stack([fc(x) for fc in self.fc_outs])
        out_global = self.global_head(x)
        return out_comp, out_global


class VNet(nn.Module):
    def __init__(self, ncomp, nstcomp, c=11, objectives=1, use_accrued_reward=False):
        super(VNet, self).__init__()
        self.c = c
        self.objectives = objectives
        if use_accrued_reward:
            reward_size = objectives
        else:
            reward_size = 0
        self.fc1 = nn.Linear(ncomp * nstcomp + 1 + reward_size, 100)
        self.fc2 = nn.Linear(100, 100)
        # self.fc3 = nn.Linear(100, 100)
        self.fc_out = nn.Linear(100, self.c**self.objectives)
        self.softmax = nn.Softmax(dim=1)
        # self.relu = nn.()

    def forward(self, x):
        # s = s.view(s.size(0), -1)

        # x = torch.cat([s, t], dim=1)
        x = tanh(self.fc1(x))
        x = tanh(self.fc2(x))
        # x = tanh(self.fc3(x))
        x = self.fc_out(x)
        x = self.softmax(x)
        # print(x)
        return x.view(-1, *([self.c]*self.objectives))
