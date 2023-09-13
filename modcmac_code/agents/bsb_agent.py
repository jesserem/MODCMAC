import numpy as np
import torch


class BSB:
    def __init__(self, env, utility, sample=False):
        self.env = env
        self.utility = utility
        self.sample = sample

    def act(self, obs):
        obs = obs.squeeze()
        action_probs = np.zeros((self.env.ncomp, self.env.nacomp))
        state1 = np.zeros(self.env.ncomp)
        for i in range(self.env.ncomp):
            state1[i] = obs[i, 0]
            action_probs[i, 0] = obs[i, 0] + obs[i, 1] * 0.5
            action_probs[i, 1] = obs[i, 1] * 0.5 + obs[i, 2]
            action_probs[i, 2] = obs[i, 3] + obs[i, 4]
        action_comp = np.ones(self.env.ncomp + 1, dtype=int) * -1
        if self.sample:
            for i in range(self.env.ncomp):
                action_comp[i] = np.random.choice(self.env.nacomp, p=action_probs[i])
        else:
            action_comp[:self.env.ncomp] = np.argmax(action_probs, axis=1)
        if np.mean(state1) > 0.5:
            action_comp[self.env.ncomp] = 1
        else:
            action_comp[self.env.ncomp] = 0
        return action_comp

    def do_run(self, n_steps=50):
        obs = self.env.reset()
        total_reward = np.zeros(2)
        has_failed = False
        i = 0
        while True:
            i += 1
            action = self.act(obs)
            obs, reward, terminal, trunc, _ = self.env.step(action)
            total_reward += reward
            if terminal:
                has_failed = True
                break
            if i >= n_steps:
                break
        return total_reward, i, has_failed

    def do_test(self, n_times=5, n_steps=50):
        rewards = np.zeros((n_times, 2))
        utilities = np.zeros((n_times, 1))
        length = np.zeros(n_times)
        cost_list = []
        risk_list = []
        utility_list = []
        for i in range(n_times):
            reward, n, _ = self.do_run(n_steps)
            length[i] = n
            rewards[i] = reward
            cost = np.abs(reward[0])
            risk = 1 - np.exp(reward[1])
            util = self.utility(torch.from_numpy(reward.reshape(1, -1))).numpy()
            cost_list.append(cost)
            risk_list.append(risk)
            utility_list.append(np.abs(util[0, 0]))
            utilities[i] = util

        return np.array(cost_list), np.array(risk_list), np.array(utility_list)
