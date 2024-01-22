import sys

import pandas as pd

from ..distributions.distributions import MultiCategorical, MultiNormal
import torch
import torch.optim as optim
import os
import numpy as np
from torch.distributions import Categorical, Normal
from ..replaybuffer.ReplayBuffer import Memory as ReplayBuffer, Transition, BatchTransition
from datetime import datetime
from time import time, sleep
from gymnasium import Env
from typing import Optional, Union, List, Callable, Tuple
from torch import nn
import wandb
from dataclasses import dataclass


class Log(object):
    def __init__(self, n_objectives: int):
        self.n_objectives = n_objectives
        self.n_update = 0
        self.n_episodes = 0
        self.wandb_policy_loss = 0
        self.wandb_value_loss = 0
        self.wandb_entropy = 0
        self.wandb_episode_utility = 0
        self.wandb_episode_discounted_utility = 0
        self.wandb_episode_reward = np.zeros(self.n_objectives)
        self.wandb_episode_discounted_reward = np.zeros(self.n_objectives)

    def add_losses(self, policy_loss: float, value_loss: float, entropy: float):
        self.wandb_policy_loss += policy_loss
        self.wandb_value_loss += value_loss
        self.wandb_entropy += entropy
        self.n_update += 1

    def add_episode(self, utility: float, discounted_utility: float, reward: np.ndarray, discounted_reward: np.ndarray):
        self.wandb_episode_utility += utility
        self.wandb_episode_discounted_utility += discounted_utility
        self.wandb_episode_reward += reward
        self.wandb_episode_discounted_reward += discounted_reward
        self.n_episodes += 1

    def get_loss(self) -> Tuple[float, float, float]:
        policy_loss = self.wandb_policy_loss / self.n_update
        value_loss = self.wandb_value_loss / self.n_update
        entropy = self.wandb_entropy / self.n_update
        self.wandb_policy_loss = 0
        self.wandb_value_loss = 0
        self.wandb_entropy = 0
        self.n_update = 0
        return policy_loss, value_loss, entropy

    def get_episode(self) -> Tuple[float, float, np.ndarray, np.ndarray]:
        utility = self.wandb_episode_utility / self.n_episodes
        discounted_utility = self.wandb_episode_discounted_utility / self.n_episodes
        reward = self.wandb_episode_reward / self.n_episodes
        discounted_reward = self.wandb_episode_discounted_reward / self.n_episodes
        self.wandb_episode_utility = 0
        self.wandb_episode_discounted_utility = 0
        self.wandb_episode_reward = np.zeros(self.n_objectives)
        self.wandb_episode_discounted_reward = np.zeros(self.n_objectives)
        self.n_episodes = 0
        return utility, discounted_utility, reward, discounted_reward

    @property
    def can_print(self):
        return self.n_update > 0 and self.n_episodes > 0


class MODCMACBase:
    """
    MO_DCMAC (Multi-Objective Deep Continuous Actor-Critic) class for reinforcement learning.

    This class provides the implementation for the MO_DCMAC agent, which includes actor and critic networks,
    and provides functionalities for training and evaluation based on given environments.

    Attributes:
    -----------
    env : Env
        The environment for the agent to interact with.
    ncomp : int
        Number of components.
    nstcomp : int
        Number of state components.
    nacomp : int
        Number of action components.
    naglobal : int
        Number of global actions.
    utility : Callable
        A utility function that maps a torch tensor to another torch tensor.
    lr_critic : float, optional
        Learning rate for the critic network. Default is 0.001.
    lr_policy : float, optional
        Learning rate for the policy (actor) network. Default is 0.0001.
    device : str, optional
        Device for computation ('cpu' or 'cuda'). It's automatically set based on availability.
    buffer_size : int, optional
        Size of the replay buffer. Default is 1000.
    gamma : float, optional
        Discount factor for future rewards. Default is 0.975.
    name : str, optional
        Base name for saving and logging. Default is "MO_DCMAC".
    save_folder : str, optional
        Path to the folder where models will be saved. Default is "./models".
    use_lr_scheduler : bool, optional
        Whether to use a learning rate scheduler. Default is True.
    num_episodes : int, optional
        Total number of training episodes. Default is 500,000.
    eval_only : bool, optional
        If True, the model is in evaluation mode only. Default is False.
    ep_length : int, optional
        Length of each episode. Default is 50.
    v_min : Union[List, float], optional
        Minimum value for the value distribution. Can be a list or a single float. Default is -10.
    v_max : Union[List, float], optional
        Maximum value for the value distribution. Can be a list or a single float. Default is 0.
    c : int, optional
        Number of atoms in the value distribution. Default is 11.
    n_step_update : int, optional
        Number of steps to take before updating the networks. Default is 1.
    v_coef : float, optional
        Coefficient for the value loss in the total loss. Default is 0.5.
    e_coef : float, optional
        Coefficient for the entropy term in the total loss. Default is 0.01.
    clip_grad_norm : int or None, optional
        Maximum allowed gradient norm. If None, no gradient clipping is applied. Default is None.
    do_eval_every : int, optional
        Number of episodes to wait before conducting an evaluation. Default is 1000.
    use_accrued_reward : bool, optional
        Whether to use accrued rewards in training. Default is True.
    n_eval : int, optional
        Number of evaluations to perform. Default is 100.

    Note:
    -----
    Some attributes are derived from the provided arguments and are not directly provided as parameters.
    """

    def __init__(self, pnet: torch.nn.Module, vnet: torch.nn.Module, env: Env,
                 utility: Callable[[torch.Tensor], torch.Tensor], lr_critic: float = 0.001,
                 lr_policy: float = 0.0001, device: Optional[str] = None, buffer_size: int = 1000, gamma: float = 0.975,
                 name: str = "MODCMAC_base", save_folder: str = "./models", use_lr_scheduler: bool = True,
                 num_steps: int = 2_500_000, eval_only: bool = False, ep_length: int = 50, n_step_update: int = 1,
                 v_coef: float = 0.5, e_coef: float = 0.01, clip_grad_norm: Optional[int] = None,
                 do_eval_every: int = 1000, use_accrued_reward: bool = True, update_wandb_step: int = 1000,
                 n_eval: int = 100, log_run: bool = False, obj_names: Optional[List[str]] = None,
                 project_name: str = "modcmac_base", continuous: bool = False, normalize_advantage: bool = True,
                 seed: Optional[int] = None, print_values: bool = True):
        self.seed = seed
        self.print_values = print_values
        self.np_random = np.random.default_rng(seed)
        self.use_accrued_reward = use_accrued_reward
        self.normalize_advantage = normalize_advantage
        self.update_wandb_step = update_wandb_step
        self.continuous = continuous
        self.save_folder = save_folder
        self.num_steps = num_steps
        self.use_lr_scheduler = use_lr_scheduler
        self.n_eval = n_eval
        self.eval_only = eval_only
        self.log_run = log_run
        if self.seed is not None:
            self.name = "{name}_seed_{seed}_date_{date}".format(name=name,
                                                                seed=self.seed,
                                                                date=datetime.now().strftime("%Y%m%d-%H%M%S"))
        else:
            self.name = "{name}_{date}".format(name=name, date=datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.total_steps = 0
        self.do_eval_every = do_eval_every
        self.project_name = project_name
        self.env = env
        self.utility = utility
        self.n_step_update = n_step_update
        self.v_coef = v_coef
        self.e_coef = e_coef
        self.clip_grad_norm = clip_grad_norm
        self.n_objectives = np.prod(self.env.reward_space.shape)
        if continuous:
            self.distribution = MultiNormal
        else:
            self.distribution = MultiCategorical
        if obj_names is None:
            self.obj_names = [f"obj_{i + 1}" for i in range(self.n_objectives)]
        else:
            self.obj_names = obj_names

        if device is None and torch.cuda.is_available():
            self.device = "cuda"
        elif device is None and not torch.cuda.is_available():
            self.device = "cpu"
        self.device = device
        self.lr_critic = lr_critic
        self.lr_policy = lr_policy

        self.Pnet = pnet.to(self.device)
        # exit()
        self.Vnet = vnet.to(self.device)
        self.optim = optim.Adam([
            {"params": self.Pnet.parameters(), "lr": self.lr_policy},
            {"params": self.Vnet.parameters(), "lr": self.lr_critic}
        ])
        self.lr_policy_end = self.lr_policy * 0.1
        self.lr_critic_end = self.lr_critic * 0.1
        n_update_steps = self.num_steps // self.n_step_update
        lr_lambda_policy = lambda step: 1 - (min(step, n_update_steps) / n_update_steps) * (
                1 - (self.lr_policy_end / self.lr_policy))
        lr_lambda_critic = lambda step: 1 - (min(step, n_update_steps) / n_update_steps) * (
                1 - (self.lr_critic_end / self.lr_critic))
        self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=[lr_lambda_policy, lr_lambda_critic])

        self.buffer_size = buffer_size
        self.buffer = ReplayBuffer(self.buffer_size, device=self.device)

        self.gamma = gamma
        self.ep_length = ep_length

        self.model_path = os.path.join(self.save_folder, self.name)
        if not self.eval_only:
            os.mkdir(self.model_path)

        self.accrued = torch.tensor([]).view(0, self.n_objectives)
        self.hparams = None
        self.log = Log(self.n_objectives)

    def init_wandb(self) -> None:
        wandb.init(
            project=self.project_name,
            name=self.name,
            config=self.hparams,
            save_code=True,
        )
        pass
        # wandb.define_metric("*", step_metric="global_step")

    def close_wandb(self) -> None:
        if self.log_run:
            wandb.finish()

    def save_model(self, name: str = "") -> None:
        """
        Saves the state dictionaries of the Pnet (policy network) and Vnet (value network) to the specified path.

        Parameters:
        -----------
        name : str, optional
            A unique identifier or suffix for the saved model filenames. Default is an empty string.

        Note:
        -----
        The saved models will be stored in the class's defined `model_path` directory. The filenames will be of the
        format:
        - `Pnet_f{name}.pt` for the policy network.
        - `Vnet_{name}.pt` for the value network.
        """
        torch.save(self.Pnet.state_dict(), os.path.join(self.model_path, f"Pnet_{name}.pt"))
        torch.save(self.Vnet.state_dict(), os.path.join(self.model_path, f"Vnet_{name}.pt"))

    def select_action(self, observation: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Selects an action based on the given observation using the policy network (Pnet).

        Parameters:
        -----------
        observation : torch.Tensor
            The current state observation from the environment.
        training : bool, optional
            If True, the action is sampled from the policy's distribution. If False, the action with the highest
            probability is chosen (greedy action). Default is True.

        Returns:
        --------
        action : torch.Tensor
            The selected action.

        """
        observation = observation.to(self.device)
        with torch.no_grad():
            action_output = self.Pnet(observation)
        action_dist = self.distribution(logits=action_output)
        if training:
            action = action_dist.sample().detach()
        else:
            action = action_dist.get_best_action().detach()
        return action

    def create_input(self, observation: torch.Tensor, accrued: torch.Tensor) -> torch.Tensor:
        """
        Creates the input for the policy network (Pnet) based on the given belief state.

        Parameters:
        -----------
        belief : torch.Tensor
            The current belief state.
        i : int
            The current time step.
        accrued : torch.Tensor
            The accrued rewards.

        Returns:
        --------
        observation : torch.Tensor
            The input for the policy network.

        Notes:
        ------
        The `observation` is a flattened tensor containing the belief state, the current time step, and the accrued
        rewards.
        """
        if self.use_accrued_reward:
            observation = torch.cat([observation.unsqueeze(0), accrued[-1:].float()], dim=1)
        return observation.float()

    def evaluate_old(self, scoring_table: Optional[np.ndarray] = None, run: int = 0) -> Tuple[
        np.ndarray, int, bool, np.ndarray]:
        """
        Evaluates the policy on the environment for a single episode.

        Returns:
        --------
        reward : np.ndarray
            The total reward for the episode.
        i : int
            The number of steps taken in the episode.
        has_failed : bool
            Whether the episode has failed or not.
        """
        done = False
        has_failed = False
        i = 0
        belief = self.env.reset()
        belief = torch.tensor(belief).float()
        i_torch = torch.tensor(i / self.ep_length).float().view(1, 1)
        if self.use_accrued_reward:

            observation = torch.cat([belief.view(belief.size(0), -1), i_torch, torch.zeros((1, self.n_objectives))],
                                    dim=1)
        else:
            observation = torch.cat([belief.view(belief.size(0), -1), i_torch], dim=1)
        accrued = torch.tensor([]).view(0, self.n_objectives)
        total_reward = np.zeros(self.n_objectives)
        while not done:
            action, behavior_ac_comp = self.select_action(observation, training=False)  # select action

            next_belief, reward, done, trunc, info = self.env.step(action)
            total_reward += reward

            gamma = torch.tensor([self.gamma ** i])
            reward_tensor = torch.tensor(reward).float().view(1, self.n_objectives)
            if i == 0:
                accrued = torch.cat((accrued[:-1], torch.zeros_like(reward_tensor), reward_tensor), dim=0)
            else:
                accrued = torch.cat((accrued, accrued[-1] + gamma * reward_tensor), dim=0)
            next_observation = self.create_input_old(torch.tensor(next_belief).float(), i + 1, accrued)
            observation = next_observation

            if done:
                has_failed = True
            if i == self.ep_length - 1:
                done = True
            i += 1

        return total_reward, i, has_failed, scoring_table

    def evaluate(self, scoring_table: Optional[np.ndarray] = None, run: int = 0, gamma_val=1.0) -> Tuple[
        np.ndarray, np.ndarray, int, bool, np.ndarray]:
        """
        Evaluates the policy on the environment for a single episode.

        Returns:
        --------
        reward : np.ndarray
            The total reward for the episode.
        i : int
            The number of steps taken in the episode.
        has_failed : bool
            Whether the episode has failed or not.
        """
        done = False
        has_failed = False
        i = 0
        observation, _ = self.env.reset()
        observation = torch.tensor(observation).float()
        observation = self.create_input(observation, torch.zeros((1, self.n_objectives)))
        accrued = torch.tensor([]).view(0, self.n_objectives)
        total_reward = np.zeros(self.n_objectives)
        total_discounted_reward = np.zeros(self.n_objectives)

        while not done:
            action = self.select_action(observation, training=False)  # select action

            next_observation, reward, done, trunc, info = self.env.step(action.detach().cpu().numpy())
            done = done or trunc
            # print(type((gamma_val ** i) * reward))
            # print(type(reward))
            total_reward += reward
            total_discounted_reward += (gamma_val ** i) * reward
            gamma = torch.tensor([self.gamma ** i])
            reward_tensor = torch.tensor(reward).float().view(1, self.n_objectives)
            if i == 0:
                accrued = torch.cat((accrued[:-1], torch.zeros_like(reward_tensor), reward_tensor), dim=0)
            else:
                accrued = torch.cat((accrued, accrued[-1] + gamma * reward_tensor), dim=0)
            observation = self.create_input(torch.tensor(next_observation), accrued)
            if done:
                has_failed = True
            i += 1

        return total_reward, total_discounted_reward, i, has_failed, self.env.get_episode()

    def do_eval(self, episodes: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluates the policy for a number of episodes, and returns the cost, risk and utility for each episode.

        Parameters:
        -----------
        episodes : int, optional
            Number of episodes to evaluate. Default is 1000.

        Returns:
        --------
        reward_cost_array : np.ndarray
            The total cost for each episode.
        reward_risk_array : np.ndarray
            The total risk for each episode.
        reward_uti_array : np.ndarray
            The total utility for each episode.
        """
        reward_cost_array = np.zeros(episodes)
        reward_risk_array = np.zeros(episodes)
        reward_uti_array = np.zeros(episodes)
        for i in range(episodes):
            reward, _, _, _, scoring_table = self.evaluate()

            total_cost_eval_curr = reward[0]
            total_prob_eval_curr = reward[1]
            eval_cost_curr = self.utility(torch.tensor(reward).view(1, self.n_objectives)).item()
            reward_cost_array[i] = total_cost_eval_curr
            reward_risk_array[i] = total_prob_eval_curr
            reward_uti_array[i] = eval_cost_curr

        return reward_cost_array, reward_risk_array, reward_uti_array

    def do_eval_new_exp(self, episodes: int = 1000, gamma=1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, List[pd.DataFrame]]:
        """
        Evaluates the policy for a number of episodes, and returns the cost, risk and utility for each episode.

        Parameters:
        -----------
        episodes : int, optional
            Number of episodes to evaluate. Default is 1000.

        Returns:
        --------
        reward_cost_array : np.ndarray
            The total cost for each episode.
        reward_risk_array : np.ndarray
            The total risk for each episode.
        reward_uti_array : np.ndarray
            The total utility for each episode.
        """
        reward_cost_array = np.zeros(episodes)
        reward_risk_array = np.zeros(episodes)
        reward_uti_array = np.zeros(episodes)
        discounted_reward_cost_array = np.zeros(episodes)
        discounted_reward_risk_array = np.zeros(episodes)
        discounted_reward_uti_array = np.zeros(episodes)
        pd_frames = []
        for i in range(episodes):
            reward, discount_reward, _, _, scoring_table = self.evaluate(gamma_val=gamma)
            pd_frames.append(scoring_table)
            total_cost_eval_curr = reward[0]
            total_prob_eval_curr = reward[1]
            total_discounted_cost_eval_curr = discount_reward[0]
            total_discounted_prob_eval_curr = discount_reward[1]

            eval_cost_curr = self.utility(torch.tensor(reward).view(1, self.n_objectives)).item()
            eval_discounted_cost_curr = self.utility(torch.tensor(discount_reward).view(1, self.n_objectives)).item()
            reward_cost_array[i] = total_cost_eval_curr
            reward_risk_array[i] = total_prob_eval_curr
            reward_uti_array[i] = eval_cost_curr
            discounted_reward_cost_array[i] = total_discounted_cost_eval_curr
            discounted_reward_risk_array[i] = total_discounted_prob_eval_curr
            discounted_reward_uti_array[i] = eval_discounted_cost_curr

        return (reward_cost_array, reward_risk_array, reward_uti_array, discounted_reward_cost_array,
                discounted_reward_risk_array, discounted_reward_uti_array, pd_frames)

    def train(self, training_steps: int = 25_000_000, seed: Optional[int] = None) -> float:
        """
        Trains the MODCMAC agent for `episodes` amount of episodes.

        Parameters:
        -----------
        episodes : int, optional
            Number of episodes to train for. Default is 500,000.
        seed : int, optional
            Seed for the environment. Default is None.

        Notes:
        ------
        The training process will save the weights at the end of the run and at a set amount of episodes. The saved
        weights will be stored in the class's defined `model_path` directory. The filenames will be of the format:
        - `Pnet_f{name}.pt` for the policy network.
        - `Vnet_{name}.pt` for the value network.
        """

        if self.log_run:
            self.init_wandb()

        curr_ep = 0
        start_time = time()
        while self.total_steps < training_steps:
            i = 0

            if seed is not None:
                seed += 1
                observation, _ = self.env.reset(seed=seed)
            else:
                observation, _ = self.env.reset()
            curr_ep += 1
            observation = torch.tensor(observation).float()

            observation = self.create_input(observation, torch.zeros((1, self.n_objectives)))

            done = False
            total_reward = np.zeros(self.n_objectives)
            total_reward_discounted = np.zeros(self.n_objectives)

            while not done:
                action = self.select_action(observation, training=True)  # select action
                next_observation, reward, done, trunc, info = self.env.step(action.detach().cpu().numpy())
                done = done or trunc
                gamma = torch.tensor([self.gamma ** i])

                total_reward_discounted += (self.gamma ** i) * reward
                total_reward += reward
                reward = torch.tensor(reward, dtype=torch.float32)
                reward = reward.view(1, self.n_objectives)
                # print(reward.shape)

                if i == 0:
                    self.accrued = torch.cat((self.accrued[:-1], torch.zeros_like(reward), reward), dim=0)
                else:
                    self.accrued = torch.cat((self.accrued, self.accrued[-1] + gamma * reward), dim=0)

                next_observation = self.create_input(torch.tensor(next_observation).float(), self.accrued)

                # if i == self.ep_length - 1:
                #     done = True
                transition = Transition(observation=observation,
                                        next_observation=next_observation,
                                        action=action.unsqueeze(0),
                                        reward=reward,
                                        terminal=torch.tensor(done).view(1, 1),
                                        gamma=gamma)

                self.buffer.add(transition)

                observation = next_observation  # set initial observations for next step
                if (self.total_steps + 1) % self.n_step_update == 0:
                    loss_p, loss_v, entropy, update_loss = self.learn()
                    self.log.add_losses(loss_p, loss_v, entropy)
                i += 1
                self.total_steps += 1
                # print(curr_step)
                if self.print_values and self.total_steps % 5000 == 0:
                    curr_run_time = time() - start_time
                    run_time_per_ep = curr_run_time / self.total_steps
                    eta = run_time_per_ep * (training_steps - self.total_steps)
                    eta_min = eta / 60
                    eta_hour = eta_min / 60
                    print(f'Step {self.total_steps}/{training_steps} | '
                          f'Run time: {curr_run_time:.2f}s | '
                          f'ETA seconds: {eta:.2f}s | '
                          f'ETA minutes: {eta_min:.2f}min | '
                          f'ETA hours: {eta_hour:.2f}h | '
                          f'Current Episode {curr_ep}', file=sys.stderr)
                if self.log_run and self.total_steps % self.update_wandb_step == 0 and self.log.can_print:
                    curr_wandb_policy_loss, curr_wandb_value_loss, curr_wandb_entropy = self.log.get_loss()
                    (curr_wandb_episode_utility, curr_wandb_episode_discounted_utility, curr_wandb_episode_reward,
                     curr_wandb_episode_discounted_reward) = self.log.get_episode()
                    wandb_dict = {
                        f"losses/policy": curr_wandb_policy_loss,
                        f"losses/value": curr_wandb_value_loss,
                        f"losses/entropy": curr_wandb_entropy,
                        f"training/Utility": curr_wandb_episode_utility,
                        f"training_discounted/Utility": curr_wandb_episode_discounted_utility,
                        f"learning_rate/actor": self.optim.param_groups[0]["lr"],
                        f"learning_rate/critic": self.optim.param_groups[1]["lr"],
                    }
                    for s in range(self.n_objectives):
                        wandb_dict[f"training/{self.obj_names[s]}"] = curr_wandb_episode_reward[s]
                        wandb_dict[f"training_discounted/{self.obj_names[s]}"] = (
                            curr_wandb_episode_discounted_reward[s])
                    wandb.log(wandb_dict, step=self.total_steps)

            if (curr_ep + 1) % self.do_eval_every == 0:
                reward_array = np.zeros((self.n_eval, self.n_objectives))
                reward_uti_array = np.zeros(self.n_eval)
                """
                Evaluates the policy on the environment for n_eval episodes (n_eval is a parameter of the class)
                """
                for ne in range(self.n_eval):
                    reward, _, n, has_failed, _ = self.evaluate()
                    eval_cost_curr = self.utility(torch.tensor(reward).view(1, self.n_objectives)).item()
                    reward_array[ne] = reward
                    reward_uti_array[ne] = eval_cost_curr

                utility_mean = np.mean(reward_uti_array)
                utility_std = np.std(reward_uti_array)
                if self.log_run:
                    wandb_dict = {
                        f"evaluation/Utility": utility_mean,
                        f"Utility": utility_mean,
                        f"evaluation/Utility_std": utility_std,

                    }
                    for n_o in range(self.n_objectives):
                        wandb_dict[f"evaluation/{self.obj_names[n_o]}"] = np.mean(reward_array[:, n_o])
                        wandb_dict[f"evaluation/{self.obj_names[n_o]}_std"] = np.std(reward_array[:, n_o])
                    wandb.log(wandb_dict, step=self.total_steps)
                if self.print_values:
                    print("\n----------------------------------------------------------")
                    print_mes = (f"The evaluation at step {self.total_steps} returned:\n" +
                                 f"Utility: {np.round(utility_mean, 3)} SD={np.round(utility_std, 3)}\n")
                    for n_o in range(self.n_objectives):
                        print_mes += (f"{self.obj_names[n_o]}: {np.round(np.mean(reward_array[:, n_o]), 3)} "
                                      f"SD={np.round(np.std(reward_array[:, n_o]), 3)}\n")
                    print(print_mes)
                    print("----------------------------------------------------------\n")
                self.save_model()
            if (self.total_steps + 1) % 1_000_000 == 0:
                self.save_model(name="curr_step_" + str(self.total_steps + 1))

            curr_total_uti = self.utility(torch.tensor(total_reward).view(1, self.n_objectives))
            curr_total_uti_discounted = self.utility(torch.tensor(total_reward_discounted).view(1, self.n_objectives))
            self.log.add_episode(curr_total_uti.item(), curr_total_uti_discounted.item(), total_reward,
                                 total_reward_discounted)

        reward_array = np.zeros((1000, self.n_objectives))
        reward_uti_array = np.zeros(1000)

        for ne in range(1000):
            reward, _, _, has_failed, _ = self.evaluate()
            eval_uti_curr = self.utility(torch.tensor(reward).view(1, self.n_objectives)).item()
            reward_uti_array[ne] = eval_uti_curr
            reward_array[ne] = reward

        utility_mean = np.mean(reward_uti_array)
        utility_std = np.std(reward_uti_array)

        if self.log_run:
            wand_dict = {
                f"evaluation/Utility": utility_mean,
                f"Utility": utility_mean,
                f"evaluation/Utility_std": utility_std,
            }
            for n_o in range(self.n_objectives):
                wand_dict[f"evaluation/{self.obj_names[n_o]}"] = np.mean(reward_array[:, n_o])
                wand_dict[f"evaluation/{self.obj_names[n_o]}_std"] = np.std(reward_array[:, n_o])
            wandb.log(wand_dict, step=self.total_steps)

        self.save_model("final")
        self.close_wandb()
        # sleep(60)
        return utility_mean

    def load_model(self, path_pnet: str, path_vnet: str) -> None:
        """
        Loads the model from the given paths

        Parameters:
        -----------
        path_pnet : str
            Path to the policy network weights.
        path_vnet : str
            Path to the value network weights.

        """
        print(path_vnet)
        print(path_pnet)
        vnet_weights = torch.load(path_vnet)
        self.Vnet.load_state_dict(vnet_weights)
        self.Pnet.load_state_dict(torch.load(path_pnet))

    def learn_critic(self, batch: BatchTransition) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def learn_policy_discrete_old(self, advantage: torch.Tensor, batch: BatchTransition) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        pi_ac = self.Pnet(batch.belief)
        pi_ac = pi_ac.softmax(dim=2)
        n_heads = pi_ac.shape[0]
        ind = range(self.n_step_update)
        pi_aa = torch.zeros((n_heads, self.n_step_update), device=self.device)
        mu_aa = torch.zeros((n_heads, self.n_step_update), device=self.device)
        for k in range(n_heads):
            pi_aa[k] = pi_ac[k][ind, batch.action[:,
                                     k]].detach()  # target (current) probability of action retrieved from replay buffer for each component
            mu_aa[k] = batch.behavior_ac[ind, k, list(batch.action[:,
                                                      k])].detach()  # sampled (original) probability of action retrieved from replay buffer for each component

        rho = torch.prod(pi_aa / mu_aa, dim=0)  # joint importance weight
        rho = torch.minimum(rho, 2 * torch.ones(self.n_step_update, device=self.device))  # clip importance weight

        advantage = torch.mul(advantage, rho)  # weighted advantage
        log_prob = torch.zeros((n_heads, self.n_step_update), device=self.device)

        entropy = torch.zeros(self.n_step_update, dtype=torch.float)
        for j in range(n_heads):
            dist = Categorical(probs=pi_ac[j])
            log_prob[j] = -dist.log_prob(batch.action[:, j])
            entropy += dist.entropy()

        actor_loss = torch.sum(log_prob, dim=0) * advantage.detach()
        return actor_loss, entropy

    def learn_policy_discrete(self, advantage: torch.Tensor, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        pi_ac = self.Pnet(batch.observation)
        pi_ac_probs = pi_ac.softmax(dim=2)

        # Shape: (n_heads, n_step_update)
        dist = Categorical(probs=pi_ac_probs)

        log_prob = -dist.log_prob(batch.action.t()).t()

        # Shape: (n_step_update,)
        entropy = torch.sum(dist.entropy(), dim=0)

        # Shape: (n_step_update,)
        actor_loss = torch.sum(log_prob, dim=1) * advantage.detach()

        return actor_loss, entropy

    def learn_policy(self, advantage: torch.Tensor, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        pi_ac = self.Pnet(batch.observation)

        # Shape: (n_heads, n_step_update)
        dist = self.distribution(logits=pi_ac)

        log_prob = -dist.log_prob(batch.action)

        # Shape: (n_step_update,)
        entropy = dist.entropy()

        # Shape: (n_step_update,)
        actor_loss = log_prob * advantage.detach()

        return actor_loss, entropy

    def learn_policy_continuous(self, advantage: torch.Tensor, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        pi_ac = self.Pnet(batch.observation)
        mu, var = pi_ac[:, :, 0], pi_ac[:, :, 1]
        # log_std = torch.log(var)
        # Shape: (n_heads, n_step_update)
        dist = Normal(mu, var)
        log_prob = -dist.log_prob(batch.action.t()).t()

        # Shape: (n_step_update,)
        entropy = torch.sum(dist.entropy(), dim=0)
        # print(entropy)
        # exit()
        # entropy = (-(torch.log(2 * torch.pi * var) + 1) / 2).mean(dim=0)
        # # print(entropy.shape)
        # exit()

        # Shape: (n_step_update,)
        actor_loss = torch.sum(log_prob, dim=1) * advantage.detach()

        return actor_loss, entropy

    def learn(self) -> Tuple[float, float, float, int]:
        """
        Performs a learning step for the policy network (Pnet) and value network (Vnet).

        Returns:
        --------
        loss_p : float
            The loss of the policy network.
        loss_v : float
            The loss of the value network.
        entropy : float
            The entropy of the policy network.
        update_loss : int
            The number of updates performed NOTE: This is legacy code and should be removed.
        """
        batch = self.buffer.last(self.n_step_update)
        critic_loss, advantage = self.learn_critic(batch)
        if self.normalize_advantage:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        actor_loss, entropy = self.learn_policy(advantage, batch)

        loss = actor_loss + self.v_coef * critic_loss - self.e_coef * entropy
        self.optim.zero_grad()
        loss = loss.mean()
        loss.backward()

        if self.clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.Pnet.parameters(), self.clip_grad_norm)
            nn.utils.clip_grad_norm_(self.Vnet.parameters(), self.clip_grad_norm)

        self.optim.step()
        if self.use_lr_scheduler:
            self.lr_scheduler.step()
        self.accrued = self.accrued[-1:]

        return actor_loss.mean().item(), critic_loss.mean().item(), entropy.mean().item(), 1
