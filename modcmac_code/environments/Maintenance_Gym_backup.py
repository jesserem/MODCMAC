import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
from .scenario import Scenario
from typing import Tuple, Optional, Dict, Any


class MaintenanceEnv(gym.Env):
    """
    The MaintenanceEnv class is the environment for the maintenance problem. The environment uses a discrete action
    and observation space. The action space is a MultiDiscrete space, where the first ncomp actions are the component
    actions and the last action is the global action. The observation space is a MultiDiscrete space for the components.

    Attributes:
        ncomp : int
            Number of components
        ndeterioration : int
            Number of deterioration steps
        ntypes : int
            Number of component types
        nstcomp : int
            Number of states per component
        naglobal : int
            Number of global actions
        nacomp : int
            Number of component actions
        nobs : int
            Number of observations
        nfail : int
            Number of failure modes
        P : np.ndarray
            Transition matrix for the deterioration of the components
        O : np.ndarray
            Observation matrix for the observation of the components
        C_glo : np.ndarray
            Cost matrix for the global actions
        C_rep : np.ndarray
            Cost matrix for the component actions
        comp_setup : np.ndarray
            Component setup, the index of the component corresponds to the component type
        f_modes : np.ndarray
            Failure modes, the first dimension corresponds to the failure mode, the second dimension corresponds to the
            substructure and the third dimension corresponds to the components in the substructure
        start_S : np.ndarray
            Initial state of the components
        total_cost : float
            Total cost of the maintenance problem
        action_space : MultiDiscrete
            Action space of the environment
        observation_space : MultiDiscrete
            Observation space of the environment
        reward_space : Box
            Reward space of the environment. It is a box due it being a multi-objective problem
        state : np.ndarray
            Current state of the environment
        det_rate : np.ndarray
            Current deterioration rate of the components
    """

    def __init__(self, ncomp: int, ndeterioration: int, ntypes: int, nstcomp: int, naglobal: int, nacomp: int,
                 nobs: int, nfail: int, P: np.ndarray, O: np.ndarray, C_glo: np.ndarray, C_rep: np.ndarray,
                 comp_setup: np.ndarray, f_modes: np.ndarray, start_S: np.ndarray, total_cost: float):
        self.ncomp = ncomp
        self.start_S = start_S
        self.ndeterioration = ndeterioration
        self.ntypes = ntypes
        self.nstcomp = nstcomp
        self.total_cost = total_cost

        self.nacomp = nacomp
        self.naglobal = naglobal
        self.nobs = nobs
        self.nfail = nfail
        self.P = P
        self.O = O

        self.C_glo = C_glo
        self.C_rep = C_rep

        self.comp_setup = comp_setup
        self.f_modes = f_modes

        self.action_space = MultiDiscrete([self.nacomp] * self.ncomp + [self.naglobal])
        self.observation_space = MultiDiscrete(
            np.array([[self.nstcomp] * self.ncomp, [self.ndeterioration] * self.ncomp]))
        self.reward_space = Box(np.array([-225000, -5]), np.array([0, 0]))

        self.state = np.zeros((self.ncomp, self.nstcomp))
        self.det_rate = np.zeros((self.ncomp, 1), dtype=int)
        self.scenario = None
        self.curr_step = 0

    def get_action(self, action: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Splits the action into the component actions and the global action.

        Parameters:
        ----------
        action : np.ndarray
            Action taken by the agent

        Returns:
        -------
        action_comp : np.ndarray
            Component actions
        action_glob : int
            Global action
        """
        action_comp, action_glob = action[:-1], action[-1]
        return action_comp, action_glob

    def reset(self, scenario: Optional[Scenario] = None, options: Optional[Dict[str, Any]] = None,
              seed: Optional[int] = None, **kwargs) -> None:
        """
        Resets the environment. The state is set to the initial state and the deterioration rate is set to 0.

        Parameters:
        ----------
        options : Dict[str, Any]
            Dictionary containing the options for the environment
        seed : int
            Seed for the environment
        **kwargs
            Additional arguments

        """
        super().reset(seed=seed)
        self.scenario = scenario
        if self.scenario is not None:
            if self.scenario.transitions.shape[0] != self.ncomp:
                raise ValueError("Scenario file does not have the correct number of components")
            if self.scenario.initial_state.max() >= self.nstcomp:
                raise ValueError("Scenario file has an initial state that is not possible")
            start_state = np.zeros((self.ncomp, self.nstcomp), dtype=int)
            for i in range(self.ncomp):
                start_state[i, self.scenario.initial_state[i]] = 1
            self.state = np.copy(start_state)
            self.det_rate = np.copy(self.scenario.det_rate)
        else:
            self.state = np.copy(self.start_S)
            self.det_rate = np.zeros((self.ncomp, 1), dtype=int)
        self.curr_step = 0
        return None

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, bool, Dict[str, Any]]:
        """
        The step function is overwritten to update the state based on the action taken. The action is split into the
        component actions and the global action. The component actions are used to update the state and the global
        action is used to calculate the immediate cost. The failure mode is calculated based on the state and the
        failure modes. The observation is calculated based on the state, the global action and the component actions.

        Parameters:
        ----------
        action : np.ndarray
            Action taken by the agent

        Returns:
        -------
        return_observation : np.ndarray
            Observation of the environment
        reward : np.ndarray
            Reward of the environment (immediate cost and failure cost)
        terminated : bool
            Whether the environment is terminated
        done : bool
            Whether the environment is done
        info : Dict[str, Any]
            Additional information (mainly for debugging)
        """
        terminated = False
        action_comp, action_glob = self.get_action(action)

        cost = self.immediate_cost(action_glob, action_comp)

        self.state = self.state_prime(self.state, action_comp)
        failure_cost = self.failure_mode(self.state)

        return_observation = np.zeros((2, self.ncomp), dtype=int)

        observation = self.observation(self.state, action_glob, action_comp)

        return_observation[0] = observation
        return_observation[1] = self.det_rate.T

        reward = np.array([cost, failure_cost])
        self.curr_step += 1
        return return_observation, reward, terminated, False, {"state": self.state}

    def failure_mode(self, s: np.ndarray) -> float:
        """
        Calculates the failure probability of each of the components. The failure probability is calculated based on the
        failure modes of the components. The failure modes are defined in the f_modes variable. The failure modes are
        defined as follows (CURRENTLY EVENTS ARE INDEPENDENT, MIGHT BE DEPENDENT IN THE FUTURE):
        F1: 1 component failed = 0.05
            2 components failed = 0.1
            3 components failed = 0.4
        F2: 1 component failed = 0.02
            2 components failed = 0.33
        F3: 1 component failed = 0.05

        Parameters:
        ----------
        s : np.ndarray
            Current state of the environment

        Returns:
        -------
        collapse_fail : float
            Failure probability of the components of the current state
        """
        fail_state = (np.argmax(s, axis=1) == self.nstcomp - 1).astype(int)
        fail_prob = 1
        # FAIL MODE 1
        for i in range(self.f_modes[0].shape[0]):
            n_fail = np.sum(fail_state[self.f_modes[0][i, :]])
            if n_fail == 1:
                fail_prob *= (1 - 0.01)
            elif n_fail == 2:
                fail_prob *= (1 - 0.10)
            elif n_fail == 3:
                fail_prob *= (1 - 0.40)
        # FAIL MODE 2
        for i in range(self.f_modes[1].shape[0]):
            n_fail = np.sum(fail_state[self.f_modes[1][i, :]])
            if n_fail == 1:
                fail_prob *= (1 - 0.03)
            elif n_fail == 2:
                fail_prob *= (1 - 0.33)
        # FAIL MODE 3
        for i in range(self.f_modes[2].shape[0]):
            n_fail = np.sum(fail_state[self.f_modes[2][i, :]])
            if n_fail == 1:
                fail_prob *= (1 - 0.05)
        collapse_fail = np.log(fail_prob)
        return collapse_fail

    def state_prime(self, s: np.ndarray, a: np.ndarray) -> np.ndarray:
        """
        Calculates the next state based on the action and the current state. If the component action is 1 (repair) and
        the component is not in the fail state, the component is sent to the previous damage state. If the component
        action is 2 (replace) the component is sent to the initial damage state (0) and the deterioration rate for that
        component is set to 0.

        Afterward the next state is probabilistically calculated based on the deterioration rate of the component and
        the transition matrix and the deterioration rate increased by 1. Lastly each component has a probability of
        failing based on the current state of the component.

        Parameters:
        ----------
        s : np.ndarray
            Current state of the environment
        a : np.ndarray
            The current component actions

        Returns:
        -------
        s_prime : np.ndarray
            Next state of the environment

        """
        s_prime = np.zeros((self.ncomp, self.nstcomp))
        s_prime[:] = s

        for i in range(self.ncomp):
            if a[i] == 1:  # if action 1 is taken, component is sent to previous damage state
                s = np.append(s_prime[i, :], np.zeros(1))
                s_prime[i, :] = s[1:self.nstcomp + 1]
                if np.sum(s_prime[i, :]) < 1:
                    s_prime[i, 0] += 1 - np.sum(s_prime[i, :])
            elif a[i] == 2:  # if action 2 is taken, component is sent to initial damage state (0)
                s_prime[i, :] = 0 * s_prime[i, :]
                s_prime[i, 0] = 1
                self.det_rate[i, 0] = 0
            if self.scenario:
                s_curr_idx = np.argmax(s_prime[i, :])
                s_pr_idx = np.minimum(s_curr_idx + self.scenario.transitions[i, self.curr_step], self.nstcomp - 1)
            else:
                p_dist = self.P[self.det_rate[i, 0], self.comp_setup[i]].T.dot(s_prime[i, :])  # environment transition
                s_pr_idx = np.random.choice(range(0, self.nstcomp), size=None, replace=True, p=p_dist)
            s_prime[i, :] = 0 * s_prime[i, :]
            s_prime[i, s_pr_idx] = 1

        self.det_rate += 1
        return s_prime

    def immediate_cost(self, a_in: int, a_rep: np.ndarray) -> float:
        """
        Calculates immediate cost, which is the sum of the global cost and the repair cost per component.

        Parameters:
        ----------
        a_in : int
            Global action
        a_rep : np.ndarray
            Component actions

        Returns:
        -------
        cost : float
            Immediate cost
        """
        cost_global = -np.sum(self.C_glo[0, a_in])
        cost_repair = -np.sum(self.C_rep[self.comp_setup, a_rep])
        cost = cost_global + cost_repair
        return cost

    def observation(self, s: np.ndarray, a_in: int, a_rep: np.ndarray) -> np.ndarray:
        """
        Calculates observation, which is the probability of observing a damage state given the current damage state and
        action. When a replacement action is taken (action 2, component) or the inspection action is taken (action 1,
        global), the observation is a full reveal for the component or global damage state, respectively.

        Parameters:
        ----------
        s : np.ndarray
            Current state of the environment
        a_in : int
            Global action
        a_rep : np.ndarray
            Component actions

        Returns:
        -------
        ob : np.ndarray
            Observation of the environment

        """
        ob = np.zeros(self.ncomp)
        for i in range(self.ncomp):
            ob_matrix = self.O[a_in]
            if a_rep[i] == 2:
                ob_matrix = self.O[1]

            ob_dist = ob_matrix[np.argmax(s[i, :]), :]
            ob[i] = np.random.choice(range(0, self.nobs), size=None, replace=True, p=ob_dist)
        return ob
