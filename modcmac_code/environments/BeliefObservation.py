import gymnasium as gym
import numpy as np
from typing import Optional, Dict, Any, Tuple


class BayesianObservation(gym.Wrapper):
    """
    This class is a wrapper for the environment that adds a Bayesian belief state to the observation space. The belief
    state is updated based on the observation and the action taken.

    The input is the environment that is wrapped.

    Attributes:
    ----------
    env: gymnasium environment
        The environment that is wrapped.
    belief: numpy array
        The belief state of the environment.

    """

    def __init__(self, env: gym.Env, episode_length: int = 50):
        super().__init__(env)
        belief_obs = self.ncomp * self.nstcomp + 1
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(belief_obs,))
        self.belief = None
        self.timestep = 0
        self.episode_length = episode_length

    def reset(self, options: Optional[Dict[str, Any]] = None, seed: Optional[int] = None, **kwargs) \
            -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        The reset function is overwritten to reset the belief state.

        Parameters
        ----------
        options: dict
            Dictionary containing the options for the environment.
        seed: int
            Seed for the random number generator. Default is None.

        Returns
        -------
        observation: numpy array
            The initial observation of the environment with the timestep.
        info: dict
            Dictionary containing additional information. For debugging.
        """
        super().reset(seed=seed, options=options)
        self.belief = np.zeros((1, self.ncomp, self.nstcomp, 1))
        self.timestep = 0
        for i in range(self.ncomp):
            self.belief[0, i, :, 0] = np.array([0.25, 0.25, 0.25, 0.2, 0.05])
        return self.create_observation(), {}

    def create_observation(self) -> np.ndarray:
        """
        Creates the observation of the environment by flattening the belief state and adding the timestep.

        Returns:
        -------
        observation: numpy array
            The observation of the environment.
        """
        observation = self.belief.flatten()
        observation = np.append(observation, self.timestep / self.episode_length)
        return observation

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, bool, Dict[str, Any]]:
        """
        The step function is overwritten to update the belief state based on the observation and the action taken.

        Parameters
        ----------
        action: numpy array
            The action taken by the agent.

        Returns
        -------
        self.belief: numpy array
            The updated belief state of the environment.
        reward: numpy array
            The reward of the agent.
        terminated: bool
            Boolean indicating if the episode is terminated.
        truncated: bool
            Boolean indicating if the episode is truncated.
        info: dict
            Dictionary containing additional information. For debugging.
        """
        obs_state, reward, terminated, truncated, info = self.env.step(action)
        action_rep, action_in = self.get_action(action)
        self.timestep += 1
        self.belief = self.belief_update(self.belief, action_in, action_rep, obs_state)
        return self.create_observation(), reward, terminated, truncated, info

    def get_observation_matrix(self, a_in, a_rep):
        o = self.O[a_in]
        if a_rep == 2:
            o = self.O[1]
        return o

    def belief_update(self, b: np.ndarray, a_in: int, a_rep: np.ndarray, obs_state: np.ndarray) -> np.ndarray:
        """
        Calculates the belief update based on the observation and the action taken. The belief is updated based on the
        observation matrix, which is a matrix that contains the probability of observing a certain damage state given
        the current damage state and the action taken.

        The belief is updated based on the following setup:
        - If action 1 is taken, the component is sent to the previous damage state.
        - If action 2 is taken, the component is sent to the initial damage state.
        - If failed state is reached, the component is sent to the failed state.

        Parameters
        ----------
        b: numpy array
            The belief state of the environment.
        a_in: int
            The inspection action taken.
        a_rep: numpy array
            The repair actions taken.
        obs_state: numpy array
            The observation state of the environment.
        """
        o = obs_state[0]
        det_rate = obs_state[1]
        b_prime = np.zeros((1, self.ncomp, self.nstcomp, 1))
        b_prime[:] = b

        for i in range(self.ncomp):
            curr_det_rate = det_rate[i] - 1
            comp_type = self.comp_setup[i]
            # print("b_prime before", b_prime[0, i, :, 0])

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
            # print("ob[i]", o[i])
            # print("a_in", a_in)
            # print("a_rep[i]", a_rep[i])
            p1 = self.P[curr_det_rate, comp_type].T.dot(b_prime[0, i, :, 0])  # environment transition
            # print("p1", p1)
            # print("ob_matrix", ob_matrix[:, int(o[i])])
            # print("p1.dot(ob_matrix[:, int(o[i])])", p1.dot(ob_matrix[:, int(o[i])]))
            b_prime[0, i, :, 0] = p1 * ob_matrix[:, int(o[i])] / (p1.dot(ob_matrix[:, int(o[i])]))  # belief update
            # print("b_prime after", b_prime[0, i, :, 0])
            # print("\n")
        return b_prime
