import gymnasium as gym
import numpy as np


class BayesianObservation(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1, self.ncomp, self.nstcomp, 1))
        self.belief = None

    """
    Reset the belief to the initial state, which is a uniform distribution over the initial damage states of each 
    component, except the failed state.
    """

    def reset(self, options=None, seed=None):
        super().reset(seed=seed, options=options)
        self.belief = np.zeros((1, self.ncomp, self.nstcomp, 1))
        for i in range(self.ncomp):
            self.belief[0, i, :, 0] = np.array([0.25, 0.25, 0.25, 0.2, 0.05])
        return self.belief

    """
    The step function is overwritten to update the belief based on the observation and the action taken.
    """

    def step(self, action):
        obs_state, reward, terminated, truncated, info = self.env.step(action)
        action_rep, action_in = self.get_action(action)
        self.belief = self.belief_update(self.belief, action_in, action_rep, obs_state)
        return self.belief, reward, terminated, truncated, info

    def get_observation_matrix(self, a_in, a_rep):
        o = self.O[a_in]
        if a_rep == 2:
            o = self.O[1]
        return o

    """
    Calculates the belief update based on the observation and the action taken. The belief is updated based on the
    observation matrix, which is a matrix that contains the probability of observing a certain damage state given the
    current damage state and the action taken.

    The belief is updated based on the following setup:
    - If action 1 is taken, the component is sent to the previous damage state.
    - If action 2 is taken, the component is sent to the initial damage state.
    - If failed state is reached, the component is sent to the failed state.

    """
    def belief_update(self, b, a_in, a_rep,
                      obs_state):  # Bayesian belief update based on previous belief, current observation, and action taken
        o = obs_state[0]
        det_rate = obs_state[1]
        b_prime = np.zeros((1, self.ncomp, self.nstcomp, 1))
        b_prime[:] = b

        for i in range(self.ncomp):
            curr_det_rate = det_rate[i] - 1
            comp_type = self.comp_setup[i]

            if a_rep[i] == 1:  # if action 1 is taken, component is sent to previous damage state
                b = np.append(b_prime[0, i, :, 0], np.zeros(1))
                b_prime[0, i, :, 0] = b[1:self.nstcomp + 1]
                if np.sum(b_prime[0, i, :, 0]) < 1:
                    b_prime[0, i, 0, 0] += 1 - np.sum(b_prime[0, i, :, 0])

            elif a_rep[i] == 2:  # if action 2 is taken, component is sent to initial damage state (0)
                b_prime[0, i, :, 0] = 0 * b_prime[0, i, :, 0]
                b_prime[0, i, 0, 0] = 1
                o[i] = 0

            ob_matrix = self.get_observation_matrix(a_in, a_rep[i])
            p1 = self.P[curr_det_rate, comp_type].T.dot(b_prime[0, i, :, 0])  # environment transition
            b_prime[0, i, :, 0] = p1 * ob_matrix[:, int(o[i])] / (p1.dot(ob_matrix[:, int(o[i])]))  # belief update
        return b_prime

