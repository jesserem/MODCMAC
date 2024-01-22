import numpy as np
import torch


class n_year_repair_agent:
    def __init__(self, env, utility, n_year=5, sample=False):
        self.env = env
        self.n_year = n_year
        self.utility = utility
        self.sample = sample
        self.n_comp = env.ncomp

    def act(self, obs):
        actions = np.zeros(self.env.ncomp + 1, dtype=int)
        for i in range(self.env.ncomp):
            if obs[i] == 0:
                actions[i] = 0
            elif obs[i] == 1 or obs[i] == 2:
                actions[i] = 1
            elif obs[i] == 3 or obs[i] == 4:
                actions[i] = 2
        return actions

    def do_run(self, n_steps=50):
        obs, _ = self.env.reset()
        total_reward = np.zeros(2)
        done = False
        no_repair_action = np.zeros(self.n_comp + 1, dtype=int)
        repair_action = np.ones(self.n_comp + 1, dtype=int)
        repair_action[-1] = 0
        i = 0
        while not done:

            if i % self.n_year == 0:
                action = repair_action
            else:
                action = no_repair_action
            # action = self.act(obs)
            obs, reward, terminal, trunc, _ = self.env.step(action)

            total_reward += reward
            if terminal or trunc:
                done = True
            i += 1
        return total_reward, i, self.env.get_episode()

    def do_test(self, n_times=5, n_steps=50):
        rewards = np.zeros((n_times, 2))
        utilities = np.zeros((n_times, 1))
        length = np.zeros(n_times)
        cost_list = []
        risk_list = []
        utility_list = []
        ep_list = []
        for i in range(n_times):
            reward, n, ep = self.do_run(n_steps)
            length[i] = n
            rewards[i] = reward
            ep_list.append(ep)
            cost = np.abs(reward[0])
            risk = 1 - np.exp(reward[1])
            util = self.utility(torch.from_numpy(reward.reshape(1, -1))).numpy()
            cost_list.append(cost)
            risk_list.append(risk)
            utility_list.append(np.abs(util[0, 0]))
            utilities[i] = util

        return np.array(cost_list), np.array(risk_list), np.array(utility_list), ep_list
