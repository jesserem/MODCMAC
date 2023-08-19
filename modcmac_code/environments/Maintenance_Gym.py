import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np

class MaintenanceEnv(gym.Env):
    def __init__(self, ncomp, ndeterioration, ntypes, nstcomp, naglobal, nacomp, nobs, nfail, P, O, C_glo, C_rep,
                 comp_setup, f_modes, start_S, total_cost):
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
        self.observation_space = MultiDiscrete(np.array([[self.nstcomp] * self.ncomp, [self.ndeterioration] * self.ncomp]))
        self.reward_space = Box(np.array([-225000, -5]), np.array([0, 0]))

        self.state = np.zeros((self.ncomp, self.nstcomp))
        self.det_rate = np.zeros((self.ncomp, 1), dtype=int)

    """
    Separates the action into the inspection action (action_both[:, 0]) and the repair action (action_both[:, 1])
    """
    def get_action(self, action):
        action_comp, action_glob = action[:-1], action[-1]
        return action_comp, action_glob

    """
    Resets the environment to the initial state (start_S) and sets the deterioration rate to zero
    """
    def reset(self, options=None, seed=None):
        super().reset(seed=seed)
        self.state = np.copy(self.start_S)
        self.det_rate = np.zeros((self.ncomp, 1), dtype=int)
        return None

    def step(self, action):
        terminated = False
        action_comp, action_glob = self.get_action(action)

        cost = self.immediate_cost(action_glob, action_comp)
        #TODO: DRAAI OM

        self.state = self.state_prime(self.state, action_comp)
        failure_cost = self.failure_mode(self.state)

        return_observation = np.zeros((2, self.ncomp), dtype=int)

        observation = self.observation(self.state, action_glob, action_comp)

        return_observation[0] = observation
        return_observation[1] = self.det_rate.T

        reward = np.array([cost, failure_cost])
        return return_observation, reward, terminated, False, {"state": self.state}

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
    """
    def failure_mode(self, s):
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

    def failure_mode(self, s):
        fail_state = (np.argmax(s, axis=1) == self.nstcomp - 1).astype(int)
        fail_prob = 1
        # FAIL MODE 1
        for i in range(self.f_modes[0].shape[0]):
            n_fail = np.sum(fail_state[self.f_modes[0][i, :]])
            if n_fail == 1:
                fail_prob *= (1 - 0.12)
            elif n_fail == 2:
                fail_prob *= (1 - 0.20)
            elif n_fail == 3:
                fail_prob *= (1 - 0.60)
        # FAIL MODE 2
        for i in range(self.f_modes[1].shape[0]):
            n_fail = np.sum(fail_state[self.f_modes[1][i, :]])
            if n_fail == 1:
                fail_prob *= (1 - 0.23)
            elif n_fail == 2:
                fail_prob *= (1 - 0.53)
        # FAIL MODE 3
        for i in range(self.f_modes[2].shape[0]):
            n_fail = np.sum(fail_state[self.f_modes[2][i, :]])
            if n_fail == 1:
                fail_prob *= (1 - 0.15)
        collapse_fail = np.log(fail_prob)
        return collapse_fail

    """
    Calculates the next state based on the action and the current state. If the component action is 1 (repair) and the
    component is not in the fail state, the component is sent to the previous damage state. If the component action is 2
    (replace) the component is sent to the initial damage state (0) and the deterioration rate for that component is 
    set to 0.
    
    Afterwards the next state is probabilistically calculated based on the deterioration rate of the component and
    the transition matrix and the deterioration rate is increased by 1. Lastly each component has a probability of
    failing based on the current state of the component.
    """
    def state_prime(self, s, a):  # damage state transition - find next state, based on current damage state and action
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

            p_dist = self.P[self.det_rate[i, 0], self.comp_setup[i]].T.dot(s_prime[i, :])  # environment transition
            s_pr_idx = np.random.choice(range(0, self.nstcomp), size=None, replace=True, p=p_dist)
            s_prime[i, :] = 0 * s_prime[i, :]
            s_prime[i, s_pr_idx] = 1

        self.det_rate += 1
        return s_prime

    """
    Calculates immediate cost, which is the sum of the global cost and the repair cost per component
    """
    def immediate_cost(self, a_in, a_rep):  # immediate reward (-cost), based on current damage state and action
        cost_global = -np.sum(self.C_glo[0, a_in])
        cost_repair = -np.sum(self.C_rep[self.comp_setup, a_rep])
        cost = cost_global + cost_repair
        return cost

    """
    Calculates observation, which is the probability of observing a damage state given the current damage state and 
    action. When a replacement action is taken (action 2, component) or the inspection action is taken (action 1,
    global), the observation is a full reveal for the component or global damage state, respectively.
    """
    def observation(self, s, a_in, a_rep):  # observation based on current damage state and action
        ob = np.zeros(self.ncomp)
        for i in range(self.ncomp):
            ob_matrix = self.O[a_in]
            if a_rep[i] == 2:
                ob_matrix = self.O[1]

            ob_dist = ob_matrix[np.argmax(s[i, :]), :]
            ob[i] = np.random.choice(range(0, self.nobs), size=None, replace=True, p=ob_dist)
        return ob



