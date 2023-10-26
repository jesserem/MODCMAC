import gymnasium as gym
import numpy as np


class PistonBallWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.num_agents = self.env.n_agents
        self.action_space = gym.spaces.MultiDiscrete([3] * self.env.action_space)

    def step(self, action):
        return self.env.step(action)
