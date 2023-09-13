import torch
import numpy as np
import random
from dataclasses import dataclass, astuple
from typing import Union


@dataclass
class Transition(object):
    belief: torch.Tensor
    belief_next: torch.Tensor
    action: np.ndarray
    behavior_ac_comp: np.ndarray
    behavior_ac_glob: np.ndarray
    reward: float
    terminal: bool
    gamma: torch.Tensor = torch.tensor([1.])


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

    def __init__(self, belief: torch.Tensor, belief_next: torch.Tensor, action: torch.Tensor,
                 behavior_ac_comp: torch.Tensor, behavior_ac_glob: torch.Tensor, cost: torch.Tensor,
                 terminal: torch.Tensor, gamma: torch.Tensor = torch.tensor([1.]),
                 device: Union[str, torch.device] = 'cpu'):

        if len(action.shape) == 1:
            action = action[:, None]
        if len(cost.shape) == 1:
            cost = cost[:, None]
        if len(terminal.shape) == 1:
            terminal = terminal[:, None]
        self.belief = belief.to(device)
        self.action = action.to(device)
        self.reward = cost.to(device)
        self.behavior_ac_comp = behavior_ac_comp.to(device)
        self.behavior_ac_glob = behavior_ac_glob.to(device)
        self.belief_next = belief_next.to(device)
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

        batch = BatchTransition(*[torch.cat(i) for i in zip(*batch)], device=self.device)
        return batch
