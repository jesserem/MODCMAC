import torch
import numpy as np
import random
from dataclasses import dataclass, astuple

@dataclass
class Transition(object):
    belief: torch.Tensor
    belief_next: torch.Tensor
    action: np.ndarray
    behavior_ac_comp: np.ndarray
    behavior_ac_glob: np.ndarray
    reward: float
    terminal: bool
    gamma: torch.Tensor=torch.tensor([1.])


class BatchTransition(object):

    def __init__(self, belief, belief_next, action, behavior_ac_comp, behavior_ac_glob, cost, terminal, gamma=1.,
                 device='cpu'):

        if len(action.shape) == 1: action = action[:, None]
        if len(cost.shape) == 1: cost = cost[:, None]
        if len(terminal.shape) == 1: terminal = terminal[:, None]
        self.belief = belief.to(device)
        self.action = action.to(device)
        self.reward = cost.to(device)
        self.behavior_ac_comp = behavior_ac_comp.to(device)
        self.behavior_ac_glob = behavior_ac_glob.to(device)
        self.belief_next = belief_next.to(device)
        self.terminal = terminal.to(device)
        self.gamma = gamma.to(device)


class Memory(object):
    def __init__(self, size, device='cpu'):
        self.buffer_size = size
        self.memory = []
        self.current = 0
        self.device = device

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        batch = BatchTransition(*[torch.cat(i) for i in zip(*batch)], device=self.device)
        return batch

    def add(self, transition):
        if len(self.memory) < self.buffer_size:
            self.memory.append(None)
        self.memory[self.current] = astuple(transition)
        self.current = (self.current + 1) % self.buffer_size

    def last(self, batch_size):
        assert len(self.memory) >= batch_size, 'not enough samples in memory'
        s_i, e_i = self.current - batch_size, self.current

        if s_i < 0:
            batch = self.memory[s_i:] + self.memory[:e_i]
        else:
            batch = self.memory[s_i:e_i]

        batch = BatchTransition(*[torch.cat(i) for i in zip(*batch)], device=self.device)
        return batch


class ReplayBuffer:
    def __init__(self, buffer_size, ncomp, nstcomp, nacomp, replace=False):
        self.buffer_size = buffer_size
        self.buffer_belief = torch.zeros((self.buffer_size, ncomp, nstcomp, 1))  # damage state memory
        self.buffer_belief_next = torch.zeros((self.buffer_size, ncomp, nstcomp, 1))
        self.buffer_time = torch.zeros((self.buffer_size))  # time step memory
        self.buffer_action = torch.zeros((self.buffer_size, ncomp), dtype=torch.int)  # action memory
        self.buffer_cost = torch.zeros(self.buffer_size)  # cost memory
        self.buffer_behavior_ac = torch.zeros((self.buffer_size, ncomp, nacomp))  # policy probability memory
        self.buffer_terminal_flag = torch.zeros(self.buffer_size)  # terminal state memory
        self.i = 0
        self.num_in_buffer = 0
        self.replace = replace

    def add(self, belief, belief_next, time, action, cost, behavior_ac, terminal_flag):
        self.buffer_belief[self.i] = belief
        self.buffer_belief_next[self.i] = belief_next
        self.buffer_time[self.i] = time
        self.buffer_action[self.i] = action
        self.buffer_cost[self.i] = cost
        self.buffer_behavior_ac[self.i] = behavior_ac
        self.buffer_terminal_flag[self.i] = terminal_flag
        self.i = (self.i + 1) % self.buffer_size
        self.num_in_buffer = min(self.num_in_buffer + 1, self.buffer_size)

    def sample(self, batch_size):
        idx = np.random.choice(self.num_in_buffer, batch_size, replace=self.replace)
        return self.buffer_belief[idx], self.buffer_belief_next[idx], self.buffer_time[idx], self.buffer_action[idx], \
            self.buffer_cost[idx], self.buffer_behavior_ac[idx], self.buffer_terminal_flag[idx]


    def check_enough(self, batch_size):
        return self.num_in_buffer >= batch_size

