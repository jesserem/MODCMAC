import gymnasium as gym
import numpy as np
from typing import Optional, Dict, Any, Tuple


class MOMobileNetWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.reward_space = gym.spaces.Box(low=-3, high=0)
