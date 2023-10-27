import torch
import numpy as np
import random
from dataclasses import dataclass, astuple
from typing import Union, Callable
from itertools import product


@dataclass
class Transition(object):
    observation: torch.Tensor
    next_observation: torch.Tensor
    action: np.ndarray
    reward: float
    terminal: bool
    gamma: torch.Tensor = torch.tensor([1.])


def batch_generator(buffer_size, batch_size):
    indexes = list(range(buffer_size))
    random.shuffle(indexes)
    for i in range(0, buffer_size, batch_size):
        yield indexes[i:i + batch_size]


class CPPOReplayBuffer(object):
    def __init__(self, buffer_size: int, c: int, utility: Callable[[torch.Tensor], torch.Tensor], n_objectives: int,
                 z: torch.Tensor, global_gamma: float, v_min: torch.Tensor, v_max: torch.Tensor, d_z: torch.Tensor,
                 r_z: torch.Tensor, observation_size: int, n_heads: int, device: Union[str, torch.device] = 'cpu',
                 action_size: int = 1):
        self.buffer_size = buffer_size
        self.c = c
        self.z = z
        self.d_z = d_z
        self.r_z = r_z
        self.global_gamma = global_gamma
        self.v_min = v_min
        self.v_max = v_max
        self.n_objectives = n_objectives
        self.utility = utility
        self.observation_size = observation_size
        self.n_heads = n_heads
        self.device = device
        self.observations = torch.zeros((self.buffer_size, self.observation_size))
        self.actions = torch.zeros((self.buffer_size, self.n_heads))
        self.rewards = torch.zeros((self.buffer_size, self.n_objectives))
        self.returns = torch.zeros((self.buffer_size, self.c, self.n_objectives))
        self.accrued = torch.zeros((self.buffer_size, self.n_objectives))
        self.values = torch.zeros((self.buffer_size, self.c, self.c))
        self.terminal = torch.zeros(self.buffer_size)
        self.log_probs = torch.zeros((self.buffer_size, self.n_heads))
        self.advantages = torch.zeros(self.buffer_size)
        self.value_target = torch.zeros((self.buffer_size, self.c, self.c))
        self.next_observations = torch.zeros((self.buffer_size, self.observation_size))
        self.gamma = torch.zeros(self.buffer_size)
        self.pos = 0

    def reset(self):
        self.observations = torch.zeros((self.buffer_size, self.observation_size))
        self.actions = torch.zeros((self.buffer_size, self.n_heads))
        self.rewards = torch.zeros((self.buffer_size, self.n_objectives))
        self.returns = torch.zeros((self.buffer_size, self.c, self.n_objectives))
        self.accrued = torch.zeros((self.buffer_size, self.n_objectives))
        self.values = torch.zeros((self.buffer_size, self.c, self.c))
        self.terminal = torch.zeros(self.buffer_size)
        self.log_probs = torch.zeros((self.buffer_size, self.n_heads))
        self.advantages = torch.zeros(self.buffer_size)
        self.value_target = torch.zeros((self.buffer_size, self.c, self.c))
        self.next_observations = torch.zeros((self.buffer_size, self.observation_size))
        self.gamma = torch.zeros(self.buffer_size)
        self.pos = 0

    def add(self, observation: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, terminal: torch.Tensor,
            log_prob: torch.Tensor, value: torch.Tensor, accrued: torch.Tensor, gamma: float) -> None:
        self.observations[self.pos].copy_(observation)
        self.actions[self.pos].copy_(action)
        self.rewards[self.pos].copy_(reward)
        self.terminal[self.pos].copy_(terminal)
        self.log_probs[self.pos].copy_(log_prob)
        self.values[self.pos].copy_(value)
        self.accrued[self.pos].copy_(accrued)
        self.gamma[self.pos] = gamma
        self.pos = (self.pos + 1) % self.buffer_size

    def calculate_target(self, p_ns: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        # print(returns.sh)
        tz = torch.stack(
            [returns[..., o].clamp(min=self.v_min[o], max=self.v_max[o]) for o in range(len(self.v_min))], dim=-1)
        b = (tz - self.v_min) / self.d_z
        l = torch.floor(b).long()
        # change b to not be exactly on border of categories
        b = torch.where(b != l, b, b - self.d_z / 100)
        b = b.clamp(min=0, max=self.c - 1)
        u = torch.ceil(b).long()
        m = torch.zeros_like(p_ns)
        i_s = torch.arange(len(returns))
        # for each objective, for each category, get lower and upper neighbour
        for c_i in product(range(self.c), repeat=self.n_objectives):
            b_i = [b[i_s, c_i[j], j] for j in range(self.n_objectives)]  # b_i..k
            l_i = [l[i_s, c_i[j], j] for j in range(self.n_objectives)]
            u_i = [u[i_s, c_i[j], j] for j in range(self.n_objectives)]
            # (b - floor(b))
            nl_i = [(b_j - l_j) for b_j, l_j in zip(b_i, l_i)]
            # (ceil(b) - b)
            nu_i = [(u_j - b_j) for b_j, u_j in zip(b_i, u_i)]
            lower_or_upper_i = [l_i, u_i]
            lower_or_upper_p = [nu_i, nl_i]
            current_i = (i_s,) + c_i
            # for each combination of lower, upper neighbour, update probabilities
            for n_i in product(range(2), repeat=self.n_objectives):
                # tuple (Batch, neighbour[0], ..., neighbour[n])
                neighbour_i = (i_s,) + tuple(lower_or_upper_i[j][i] for i, j in enumerate(n_i))
                neighbour_p = [lower_or_upper_p[j][i] for i, j in enumerate(n_i)]
                m[neighbour_i] += p_ns[current_i] * torch.stack(neighbour_p).prod(dim=0)
        return m

    def calculate_advantage(self, p_ns: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        objective_dims = tuple(range(1, len(self.values.shape)))
        accrued = self.accrued.view(len(returns), *(1,) * self.n_objectives, self.n_objectives).to(self.device)
        gamma = self.gamma.view(len(returns), *(1,) * (self.n_objectives + 1))
        # shift back discounted return: accrued + gamma^t*R_t
        accrued_v = accrued + gamma * self.r_z
        u_v_s = self.utility(accrued_v.view(-1, self.n_objectives)).view_as(self.values)
        # expected utility for current state [Batch C51 .. C51]*[Batch C51 .. C51] -> [Batch]
        u_v_s = torch.sum(u_v_s * self.values, dim=objective_dims)
        # get all combinations of n0,n1,... (so C51 goes to c51**nO)
        o_n = torch.meshgrid(*[torch.arange(self.c) for _ in range(self.n_objectives)], indexing="xy")
        # [Batch C51 .. C51 nO]
        r_z = torch.stack(tuple(returns[:, o_i, i] for i, o_i in enumerate(o_n)), dim=-1)
        accrued_r = accrued + gamma * r_z
        # compute the utility for all these returns [Batch C51 .. C51]
        u_r_s = self.utility(accrued_r.view(-1, self.n_objectives)).view_as(self.values)
        # expected utility using n-step returns: [Batch]
        u_r_s = torch.sum(u_r_s * p_ns[-1].unsqueeze(0), dim=objective_dims)
        advantage = u_r_s - u_v_s
        return advantage

    def compute_returns_and_advantage(self, next_ps, last_observation):
        non_terminal = torch.logical_not(self.terminal).unsqueeze(1)

        # [Batch C51 nO]
        self.returns = self.rewards.unsqueeze(1).expand(self.buffer_size, self.c, self.n_objectives).clone()

        # [C51 nO] + gamma*[C51 nO]*[1 1] -> [C51 nO]

        self.returns[-1] += self.global_gamma * self.z * non_terminal[-1]

        for i in range(len(self.returns) - 1, 0, -1):
            self.returns[i - 1] += self.global_gamma * self.returns[i] * non_terminal[i - 1]
        value_next = torch.cat((self.values[1:], next_ps), 0)
        self.next_observations = torch.cat((self.observations[1:], last_observation), 0)

        self.value_target = self.calculate_target(value_next, self.returns)
        # print(self.value_target[1])
        # print(returns[0])
        # exit()
        self.advantages = self.calculate_advantage(value_next, self.returns)

    def get_sample(self, batch_size: int):
        for b in batch_generator(self.buffer_size, batch_size):
            yield self.observations[b], self.actions[b], self.log_probs[b], self.advantages[b], self.value_target[b], \
                self.gamma[b], self.accrued[b], self.next_observations[b], self.returns[b]


class BatchTransition(object):
    """
    Batched Transition class. This is used for the replay buffer.

    Attributes:
    ----------
    belief: torch.Tensor
        The belief state of the environment.
    belief_next: torch.Tensor
        The next belief state of the environment.
    action: torch.Tensor
        The action taken by the agent.
    behavior_ac_comp: torch.Tensor
        The action probabilities of the agent for each component.
    behavior_ac_glob: torch.Tensor
        The action probabilities of the agent for the global actions.
    reward: torch.Tensor
        The reward received by the agent.
    terminal: torch.Tensor
        Whether the episode is terminated.
    gamma: torch.Tensor
        The discount factor. Default is 1.
    device: str or torch.device
        The device on which the tensors are stored. Default is 'cpu'.
    """

    def __init__(self, observation: torch.Tensor, next_observation: torch.Tensor, action: torch.Tensor,
                 cost: torch.Tensor, terminal: torch.Tensor, gamma: torch.Tensor = torch.tensor([1.]),
                 device: Union[str, torch.device] = 'cpu'):
        # print(gamma.shape)
        # exit()

        if len(action.shape) == 1:
            action = action[:, None]
        if len(cost.shape) == 1:
            cost = cost[:, None]
        if len(terminal.shape) == 1:
            terminal = terminal[:, None]
        self.observation = observation.to(device)
        self.action = action.to(device)
        self.reward = cost.to(device)
        self.next_observation = next_observation.to(device)
        self.terminal = terminal.to(device)
        self.gamma = gamma.to(device)


class Memory(object):
    """
    Replay buffer class. This is used to store the transitions of the agent and retrieve them.

    Attributes:
    ----------
    buffer_size: int
        The size of the replay buffer.
    memory: list
        The list containing the transitions.
    current: int
        The current index in the replay buffer.
    device: str or torch.device
        The device on which the tensors are stored. Default is 'cpu'.
    """

    def __init__(self, size: int, device: Union[str, torch.device] = 'cpu'):
        self.buffer_size = size
        self.memory = []
        self.current = 0
        self.device = device

    def sample(self, batch_size: int) -> BatchTransition:
        """
        Randomly samples transition from the buffer.

        Parameters
        ----------
        batch_size: int
            The number of transitions to sample.

        Returns
        -------
        batch: BatchTransition
            The batch of transitions.
        """
        batch = random.sample(self.memory, batch_size)
        batch = BatchTransition(*[torch.cat(i) for i in zip(*batch)], device=self.device)
        return batch

    def add(self, transition: Transition) -> None:
        """
        Adds a transition to the buffer.

        Parameters
        ----------
        transition: Transition
            The transition to add to the buffer.

        """
        if len(self.memory) < self.buffer_size:
            self.memory.append(None)
        self.memory[self.current] = astuple(transition)
        self.current = (self.current + 1) % self.buffer_size

    def last(self, batch_size: int) -> BatchTransition:
        """
        Returns the last batch_size transitions in the buffer.

        Parameters
        ----------
        batch_size: int
            The number of transitions to return.

        Returns
        -------
        batch: BatchTransition
            The batch of transitions.
        """
        assert len(self.memory) >= batch_size, 'not enough samples in memory'
        s_i, e_i = self.current - batch_size, self.current

        if s_i < 0:
            batch = self.memory[s_i:] + self.memory[:e_i]
        else:
            batch = self.memory[s_i:e_i]
        # print(self.memory[0][5].shape)
        batch = BatchTransition(*[torch.cat(i) for i in zip(*batch)], device=self.device)
        # print(batch.gamma.shape)
        return batch


class replay_buffer_sec(object):
    def __init__(self, buffer_size, ncomp, nstcomp, nacomp, nacomp_global, n_objectives, replace=False, device="cpu"):
        self.buffer_size = buffer_size
        self.buffer_belief = torch.zeros((self.buffer_size, ncomp, nstcomp, 1))  # damage state memory
        self.buffer_belief_next = torch.zeros((self.buffer_size, ncomp, nstcomp, 1))
        self.buffer_time = torch.zeros((self.buffer_size))  # time step memory
        self.buffer_action = torch.zeros((self.buffer_size, ncomp), dtype=torch.int)  # action memory
        self.buffer_action_global = torch.zeros(self.buffer_size, dtype=torch.int)
        self.buffer_accrued_reward = torch.zeros((self.buffer_size, n_objectives))  # cost memory
        self.buffer_accrued_reward_next = torch.zeros((self.buffer_size, n_objectives))  # cost memory
        self.buffer_behavior_ac = torch.zeros((self.buffer_size, ncomp, nacomp))  # policy probability memory
        self.buffer_behavior_ac_global = torch.zeros((self.buffer_size, ncomp, nacomp_global))
        self.buffer_terminal_flag = torch.zeros(self.buffer_size)  # terminal state memory
        self.buffer_gamma = torch.zeros(self.buffer_size)
        self.i = 0
        self.device = device
        self.num_in_buffer = 0
        self.replace = replace

    """
    Add batch to buffer.
    """

    def add(self, belief, belief_next, time, action, action_global, accrued_reward, accrued_reward_next, behavior_ac,
            behavior_ac_global, terminal_flag, gamma):
        self.buffer_belief[self.i] = belief
        self.buffer_belief_next[self.i] = belief_next
        self.buffer_time[self.i] = time
        self.buffer_action[self.i] = action
        self.buffer_action[self.i] = action_global
        self.buffer_accrued_reward[self.i] = accrued_reward
        self.buffer_accrued_reward_next[self.i] = accrued_reward_next
        self.buffer_behavior_ac[self.i] = behavior_ac
        self.buffer_behavior_ac_global[self.i] = behavior_ac_global
        self.buffer_terminal_flag[self.i] = terminal_flag
        self.buffer_gamma[self.i] = gamma
        self.i = (self.i + 1) % self.buffer_size
        self.num_in_buffer = min(self.num_in_buffer + 1, self.buffer_size)

    """
    Sample a batch of data from the buffer
    """

    def sample(self, batch_size):
        idx = np.random.choice(self.num_in_buffer, batch_size, replace=self.replace)
        batch_buffer_belief = self.buffer_belief[idx].to(self.device)
        batch_buffer_belief_next = self.buffer_belief_next[idx].to(self.device)
        batch_buffer_time = self.buffer_time[idx].to(self.device)
        batch_buffer_action = self.buffer_action[idx].to(self.device)
        batch_buffer_action_global = self.buffer_action_global[idx]
        batch_buffer_accrued_reward = self.buffer_accrued_reward[idx].to(self.device)
        batch_buffer_accrued_reward_next = self.buffer_accrued_reward_next[idx].to(self.device)
        batch_buffer_behaviour_ac = self.buffer_behavior_ac[idx].to(self.device)
        batch_buffer_behaviour_ac_global = self.buffer_behavior_ac_global[idx].to(self.device)
        batch_buffer_terminal_flag = self.buffer_terminal_flag[idx].to(self.device)
        batch_buffer_gamma = self.buffer_gamma[idx].to(self.device)
        return (batch_buffer_belief, batch_buffer_belief_next, batch_buffer_time, batch_buffer_action,
                batch_buffer_action_global, batch_buffer_accrued_reward, batch_buffer_accrued_reward_next,
                batch_buffer_behaviour_ac, batch_buffer_behaviour_ac_global, batch_buffer_terminal_flag,
                batch_buffer_gamma)

    """
    Check if there are enough samples for the batch size
    """

    def check_enough(self, batch_size):
        return self.num_in_buffer >= batch_size
