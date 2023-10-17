import numpy as np
from typing import Optional


class Scenario(object):
    def __init__(self, transitions: np.ndarray, initial_state: np.ndarray, det_rate: np.ndarray, ncomp: int,
                 timesteps: int, initial_belief: Optional[np.ndarray] = None, name: Optional[str] = None,
                 global_action: bool = False):
        self.transitions = transitions
        self.initial_state = initial_state
        self.initial_belief = initial_belief
        self.det_rate = det_rate
        self.ncomp = ncomp
        self.timesteps = timesteps
        self.name = name
        self.global_action = global_action
        if self.global_action:
            self.global_action_taken = np.zeros(self.timesteps, dtype=int)
        self.actions_taken = np.zeros((self.ncomp, self.timesteps), dtype=int)
