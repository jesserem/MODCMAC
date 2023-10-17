import sys

import torch
import torch.optim as optim
import os
import torch.nn as nn
from ..networks.model import PNet as PnetOrig, VNet as VnetOrig
import numpy as np
from collections.abc import Iterable
from itertools import product
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
from ..replaybuffer.ReplayBuffer import Memory as ReplayBuffer, Transition
from datetime import datetime
from time import time
from gymnasium import Env
from typing import Optional, Union, List, Callable, Tuple
import wandb


class MODCMAC:
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

    def __init__(self, env: Env, ncomp: int, nstcomp: int, nacomp: int, naglobal: int,
                 utility: Callable[[torch.Tensor], torch.Tensor], lr_critic: float = 0.001, lr_policy: float = 0.0001,
                 device: Optional[str] = None, buffer_size: int = 1000, gamma: float = 0.975, name: str = "MO_DCMAC",
                 save_folder: str = "./models", use_lr_scheduler: bool = True, num_episodes: int = 500_000,
                 eval_only: bool = False, ep_length: int = 50, v_min: Union[List, float] = -10,
                 v_max: Union[List, float] = 0, c: int = 11, n_step_update: int = 1, v_coef: float = 0.5,
                 e_coef: float = 0.01, clip_grad_norm: Optional[int] = None, do_eval_every: int = 1000,
                 use_accrued_reward: bool = True, n_eval: int = 100, log_run: bool = True):

        self.use_accrued_reward = use_accrued_reward
        self.save_folder = save_folder
        self.num_episodes = num_episodes
        self.use_lr_scheduler = use_lr_scheduler
        self.n_eval = n_eval
        self.ncomp = ncomp
        self.nstcomp = nstcomp
        self.nacomp = nacomp
        self.naglobal = naglobal
        self.eval_only = eval_only
        self.log_run = log_run
        self.name = "{name}_{date}".format(name=name, date=datetime.now().strftime("%Y%m%d-%H%M%S"))
        # if not self.eval_only:
        #     self.writer = SummaryWriter(os.path.join(save_folder, "runs/{name}".format(name=self.name)))

        self.do_eval_every = do_eval_every

        self.env = env
        self.c = c
        self.utility = utility
        self.n_step_update = n_step_update
        self.v_coef = v_coef
        self.e_coef = e_coef
        self.clip_grad_norm = clip_grad_norm
        self.n_objectives = np.prod(self.env.reward_space.shape)

        if device is None and torch.cuda.is_available():
            self.device = "cuda"
        elif device is None and not torch.cuda.is_available():
            self.device = "cpu"
        self.device = device
        self.lr_critic = lr_critic
        self.lr_policy = lr_policy

        self.Pnet = PnetOrig(ncomp, nstcomp, nacomp, naglobal, objectives=self.n_objectives,
                             use_accrued_reward=self.use_accrued_reward).to(self.device)
        # exit()
        self.Vnet = VnetOrig(ncomp, nstcomp, c=c, objectives=self.n_objectives,
                             use_accrued_reward=self.use_accrued_reward).to(self.device)
        self.optim = optim.Adam([
            {"params": self.Pnet.parameters(), "lr": self.lr_policy},
            {"params": self.Vnet.parameters(), "lr": self.lr_critic}
        ])
        self.lr_policy_end = self.lr_policy * 0.1
        self.lr_critic_end = self.lr_critic * 0.1
        lr_lambda_policy = lambda episode: 1 - (episode / self.num_episodes) * (
                1 - (self.lr_policy_end / self.lr_policy))
        lr_lambda_critic = lambda episode: 1 - (episode / self.num_episodes) * (
                1 - (self.lr_critic_end / self.lr_critic))
        self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=[lr_lambda_policy, lr_lambda_critic])

        self.buffer_size = buffer_size
        self.buffer = ReplayBuffer(self.buffer_size, device=self.device)

        self.gamma = gamma
        self.ep_length = ep_length

        self.model_path = os.path.join(self.save_folder, self.name)
        if not self.eval_only:
            os.mkdir(self.model_path)

        if not isinstance(v_min, Iterable):
            v_min = [v_min] * self.n_objectives
        if not isinstance(v_max, Iterable):
            v_max = [v_max] * self.n_objectives
        self.v_min = torch.tensor(v_min).to(self.device)
        self.v_max = torch.tensor(v_max).to(self.device)

        self.d_z = (self.v_max - self.v_min) / (self.c - 1.)
        # [C51 nO]
        self.z = torch.arange(c)[:, None].to(self.device) * self.d_z + self.v_min
        # get the utility of every possible V-value (meshgrid over objectives: *[nO Batch])
        r_ = torch.stack(torch.meshgrid(*self.z.T, indexing="xy"), dim=-1)

        self.r_z = r_.unsqueeze(0)  # [1 C51 .. c51 nO]

        self.u_z = self.utility(r_.view(-1, self.n_objectives)).view(1, *([self.c] * self.n_objectives))  # constant

        self.accrued = torch.tensor([]).view(0, self.n_objectives)
        self.hparams = {
            "clip_grad_norm": self.clip_grad_norm,
            "n_step_update": self.n_step_update,
            "v_coef": self.v_coef,
            "e_coef": self.e_coef,
            "gamma": self.gamma,
            "lr_critic": self.lr_critic,
            "lr_policy": self.lr_policy,
            "use_accrued_reward": self.use_accrued_reward,
            "v_min_cost": v_min[0],
            "v_max_cost": v_max[0],
            "v_min_risk": v_min[1],
            "v_max_risk": v_max[1],
        }

    def init_wandb(self) -> None:
        wandb.init(
            project="modcmac",
            name=self.name,
            config=self.hparams,
            save_code=True,
        )
        wandb.define_metric("*", step_metric="global_step")

    def close_wandb(self) -> None:
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
        torch.save(self.Pnet.state_dict(), os.path.join(self.model_path, f"Pnet_f{name}.pt"))
        torch.save(self.Vnet.state_dict(), os.path.join(self.model_path, f"Vnet_{name}.pt"))

    def select_action(self, observation: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor,
    torch.Tensor]:
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
        action_probs_comp : torch.Tensor
            Probabilities of each action component.
        action_probs_global : torch.Tensor
            Probabilities of each global action.

        Notes:
        ------
        - The `action` is a flattened tensor containing both component and global actions.
        - `action_probs_comp` provides the probabilities of each action component for the given observation.
        - `action_probs_global` provides the probabilities of each global action for the given observation.
        """
        observation = observation.to(self.device)
        if training:
            with torch.no_grad():
                action_output_comp, action_output_global = self.Pnet(observation)

            action_dist_comp = Categorical(logits=action_output_comp)
            action_probs_comp = action_dist_comp.probs
            action_comp = action_dist_comp.sample()

            action_dist_global = Categorical(logits=action_output_global)
            action_probs_global = action_dist_global.probs
            action_global = action_dist_global.sample()

            action = torch.cat([action_comp.flatten(), action_global.flatten()], dim=0).cpu()
            return action.flatten(), action_probs_comp.view(1, self.ncomp, self.nacomp), \
                action_probs_global.view(1, 1, self.naglobal)
        else:
            with torch.no_grad():
                action_output_comp, action_output_global = self.Pnet(observation)

            action_dist_comp = Categorical(logits=action_output_comp)
            action_probs_comp = action_dist_comp.probs
            action_comp = torch.argmax(action_probs_comp, dim=2)

            action_dist_global = Categorical(logits=action_output_global)
            action_probs_global = action_dist_global.probs
            action_global = torch.argmax(action_probs_global, dim=1)

            action = torch.cat([action_comp.flatten(), action_global.flatten()], dim=0).cpu()
            return action.flatten(), action_probs_comp.view(1, self.ncomp, self.nacomp), \
                action_probs_global.view(1, 1, self.naglobal)

    def create_input(self, belief: torch.Tensor, i: int, accrued: torch.Tensor) -> torch.Tensor:
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
        i_torch = torch.tensor(i / self.ep_length).float().view(1, 1)
        if self.use_accrued_reward:
            observation = torch.cat([belief.view(belief.size(0), -1), i_torch, accrued[-1:].float()], dim=1)
        else:
            observation = torch.cat([belief.view(belief.size(0), -1), i_torch], dim=1)
        return observation

    def evaluate(self) -> Tuple[np.ndarray, int, bool]:
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
        reward = np.zeros(self.n_objectives)
        while not done:
            action, behavior_ac_comp, behavior_ac_glob = self.select_action(observation,
                                                                            training=False)  # select action
            next_belief, cost, done, trunc, info = self.env.step(action)
            reward += cost
            gamma = torch.tensor([self.gamma ** i])
            cost_tensor = torch.tensor(cost).float().view(1, self.n_objectives)
            if i == 0:
                accrued = torch.cat((accrued[:-1], torch.zeros_like(cost_tensor), cost_tensor), dim=0)
            else:
                accrued = torch.cat((accrued, accrued[-1] + gamma * cost_tensor), dim=0)
            next_observation = self.create_input(torch.tensor(next_belief).float(), i + 1, accrued)
            observation = next_observation

            if done:
                has_failed = True
            if i == self.ep_length - 1:
                done = True
            i += 1

        return reward, i, has_failed

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
            reward, _, _ = self.evaluate()
            total_cost_eval_curr = np.abs(reward[0])
            total_prob_eval_curr = 1 - np.exp(reward[1])
            eval_cost_curr = np.abs(self.utility(torch.tensor(reward).view(1, self.n_objectives)).item())
            reward_cost_array[i] = total_cost_eval_curr
            reward_risk_array[i] = total_prob_eval_curr
            reward_uti_array[i] = eval_cost_curr

        return reward_cost_array, reward_risk_array, reward_uti_array

    def train(self, episodes: int = 500_000, seed: Optional[int] = None) -> None:
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
        total_steps = 0
        total_ep_cost = np.zeros(episodes + 1)  # episode cost (life-cycle cost)
        total_manpower = np.zeros(episodes + 1)
        start_time = time()
        self.init_wandb()

        for ep in range(0, episodes + 1):
            i = 0
            if seed is not None:
                seed += 1
                belief = self.env.reset(seed=seed)
            else:
                belief = self.env.reset()
            belief = torch.tensor(belief).float()
            i_torch = torch.tensor(i / self.ep_length).float().view(1, 1)

            if self.use_accrued_reward:
                observation = torch.cat([belief.view(belief.size(0), -1),
                                         i_torch,
                                         torch.zeros((1, self.n_objectives))],
                                        dim=1)
            else:
                observation = torch.cat([belief.view(belief.size(0), -1), i_torch], dim=1)
            total_ep_cost[ep] = 0  # initialize episode cost (life-cycle cost)
            done = False
            total_cost = np.zeros(self.n_objectives)
            total_cost_discounted = np.zeros(self.n_objectives)
            total_cost_money_discounted = 0
            total_cost_manpower_discounted = 0
            while not done:
                action, behavior_ac_comp, behavior_ac_glob = self.select_action(observation,
                                                                                training=True)  # select action
                next_belief, cost, done, trunc, info = self.env.step(action)
                gamma = torch.tensor([self.gamma ** i])
                cost_inc = cost[0]
                manpower = cost[1]
                total_cost_discounted += (self.gamma ** i) * cost
                total_cost_money_discounted += (self.gamma ** i) * cost[0]
                total_cost_manpower_discounted += (self.gamma ** i) * cost[1]
                total_cost += cost
                cost = torch.tensor(cost, dtype=torch.float32).unsqueeze(0)
                if i == 0:
                    self.accrued = torch.cat((self.accrued[:-1], torch.zeros_like(cost), cost), dim=0)
                else:
                    self.accrued = torch.cat((self.accrued, self.accrued[-1] + gamma * cost), dim=0)

                next_observation = self.create_input(torch.tensor(next_belief).float(), i + 1, self.accrued)
                total_ep_cost[ep] += cost_inc  # cumulative cost up to time i
                total_manpower[ep] += manpower
                if i == self.ep_length - 1:
                    done = True
                transition = Transition(belief=observation,
                                        belief_next=next_observation,
                                        # time=torch.tensor(i / self.ep_length).float().view(1, 1),
                                        action=action.unsqueeze(0),
                                        behavior_ac_comp=behavior_ac_comp,
                                        behavior_ac_glob=behavior_ac_glob,
                                        reward=cost,
                                        terminal=torch.tensor(done).view(1, 1),
                                        gamma=gamma)

                self.buffer.add(transition)

                observation = next_observation  # set initial observations for next step
                if (total_steps + 1) % self.n_step_update == 0:
                    loss_p, loss_v, entropy, update_loss = self.learn()
                    if self.log_run:
                        wandb.log({
                            f"losses_{self.name}/policy": loss_p,
                            f"losses_{self.name}/value": loss_v,
                            f"losses_{self.name}/entropy": entropy,
                            f"global_step": total_steps,
                        })
                    # self.writer.add_scalar('Loss/Policy', loss_p, total_steps)
                    # self.writer.add_scalar('Loss/Value', loss_v, total_steps)
                    # self.writer.add_scalar('Loss/Entropy', entropy, total_steps)

                i += 1
                total_steps += 1

            if (ep + 1) % 100 == 0:
                curr_run_time = time() - start_time
                run_time_per_ep = curr_run_time / (ep + 1)
                eta = run_time_per_ep * (episodes - ep)
                eta_min = eta / 60
                eta_hour = eta_min / 60
                print(f'Episode {ep + 1}/{episodes} | '
                      f'Run time: {curr_run_time:.2f}s | '
                      f'ETA seconds: {eta:.2f}s | '
                      f'ETA minutes: {eta_min:.2f}min | '
                      f'ETA hours: {eta_hour:.2f}h', file=sys.stderr)

            if (ep + 1) % self.do_eval_every == 0:
                reward_cost_array = np.zeros(self.n_eval)
                reward_risk_array = np.zeros(self.n_eval)
                reward_uti_array = np.zeros(self.n_eval)
                reward_len_array = np.zeros(self.n_eval)
                """
                Evaluates the policy on the environment for n_eval episodes (n_eval is a parameter of the class)
                """
                for i in range(self.n_eval):
                    reward, n, has_failed = self.evaluate()
                    total_cost_eval_curr = np.abs(reward[0])
                    total_prob_eval_curr = 1 - np.exp(reward[1])
                    eval_cost_curr = np.abs(self.utility(torch.tensor(reward).view(1, self.n_objectives)).item())
                    reward_cost_array[i] = total_cost_eval_curr
                    reward_risk_array[i] = total_prob_eval_curr
                    reward_uti_array[i] = eval_cost_curr
                    reward_len_array[i] = n
                cost_mean = np.mean(reward_cost_array)
                risk_mean = np.mean(reward_risk_array)
                utility_mean = np.mean(reward_uti_array)
                length_mean = np.mean(reward_len_array)
                cost_std = np.std(reward_cost_array)
                risk_std = np.std(reward_risk_array)
                utility_std = np.std(reward_uti_array)
                length_std = np.std(reward_len_array)
                if self.log_run:
                    wandb.log({
                        f"evaluation_{self.name}/Utility": utility_mean,
                        f"evaluation_{self.name}/Utility_std": utility_std,
                        f"evaluation_{self.name}/Cost": cost_mean,
                        f"evaluation_{self.name}/Cost_std": cost_std,
                        f"evaluation_{self.name}/Risk": risk_mean,
                        f"evaluation_{self.name}/Risk_std": risk_std,
                        f"global_step": total_steps,
                    })

                # self.writer.add_scalar('Evaluation/Utility', utility_mean, total_steps)
                # self.writer.add_scalar('Evaluation/Cost', cost_mean, total_steps)
                # self.writer.add_scalar('Evaluation/Collapse_prob', risk_mean, total_steps)
                # self.writer.add_scalar('Evaluation/Length', length_mean, total_steps)

                print("\n----------------------------------------------------------")
                print(f"The evaluation at episode {ep + 1} returned:\n"
                      f"Utility: {np.round(utility_mean, 3)} SD={np.round(utility_std, 3)}\n"
                      f"Cost: {np.round(cost_mean, 3)} SD={np.round(cost_std, 3)}\n"
                      f"Risk: {np.round(risk_mean, 3)} SD={np.round(risk_std, 3)}\n"
                      f"Length: {np.round(length_mean, 3)} SD={np.round(length_std, 3)}")
                print("----------------------------------------------------------\n")
                # self.save_model()
            if (ep + 1) % 100_000 == 0:
                self.save_model(name="ep_" + str(ep + 1))

            curr_total_uti = self.utility(torch.tensor(total_cost).view(1, self.n_objectives))
            curr_total_prob = 1 - np.exp(total_manpower[ep])
            # self.writer.add_scalar('Training/Cost', np.abs(total_ep_cost[ep]), total_steps)
            # self.writer.add_scalar('Training/Collapse_prob', curr_total_prob, total_steps)
            # self.writer.add_scalar('Training/Utility', np.abs(curr_total_uti), total_steps)
            # self.writer.add_scalar('Training/Length', i, total_steps)
            curr_total_uti_discounted = self.utility(torch.tensor(total_cost_discounted).view(1, self.n_objectives))
            curr_total_cost_discounted = np.abs(total_cost_money_discounted)
            curr_total_risk_discounted = 1 - np.exp(total_cost_manpower_discounted)
            if self.log_run:
                wandb.log({
                    f"training_{self.name}/Cost": np.abs(total_ep_cost[ep]),
                    f"training_{self.name}/Collapse_prob": curr_total_prob,
                    f"training_{self.name}/Utility": np.abs(curr_total_uti),
                    f"training_discounted_{self.name}": np.abs(curr_total_uti_discounted),
                    f"training_discounted_{self.name}/Cost": curr_total_cost_discounted,
                    f"training_discounted_{self.name}/Collapse_prob": curr_total_risk_discounted,
                    f"learning_rate_{self.name}/actor": self.optim.param_groups[0]["lr"],
                    f"learning_rate_{self.name}/critic": self.optim.param_groups[1]["lr"],
                    f"global_step": total_steps,
                })
            # self.writer.add_scalar('Hyperparameter/Utility_discounted', curr_total_uti_discounted, total_steps)
            # self.writer.add_scalar('Hyperparameter/Cost_discounted', curr_total_cost_discounted, total_steps)
            # self.writer.add_scalar('Hyperparameter/Risk_discounted', curr_total_risk_discounted, total_steps)
            # self.writer.add_scalar('Learning_rate/actor', self.optim.param_groups[0]["lr"], total_steps)
            # self.writer.add_scalar('Learning_rate/critic', self.optim.param_groups[1]["lr"], total_steps)
            if self.use_lr_scheduler:
                self.lr_scheduler.step()

        reward_cost_array = np.zeros(self.n_eval)
        reward_risk_array = np.zeros(self.n_eval)
        reward_uti_array = np.zeros(self.n_eval)
        reward_len_array = np.zeros(self.n_eval)
        for i in range(self.n_eval):
            reward, n, has_failed = self.evaluate()
            total_cost_eval_curr = np.abs(reward[0])
            total_prob_eval_curr = 1 - np.exp(reward[1])
            eval_cost_curr = np.abs(self.utility(torch.tensor(reward).view(1, self.n_objectives)).item())
            reward_cost_array[i] = total_cost_eval_curr
            reward_risk_array[i] = total_prob_eval_curr
            reward_uti_array[i] = eval_cost_curr
            reward_len_array[i] = n
        cost_mean = np.mean(reward_cost_array)
        risk_mean = np.mean(reward_risk_array)
        utility_mean = np.mean(reward_uti_array)
        cost_std = np.std(reward_cost_array)
        risk_std = np.std(reward_risk_array)
        utility_std = np.std(reward_uti_array)

        if self.log_run:
            wandb.log({
                f"evaluation_{self.name}/Utility": utility_mean,
                f"evaluation_{self.name}/Utility_std": utility_std,
                f"evaluation_{self.name}/Cost": cost_mean,
                f"evaluation_{self.name}/Cost_std": cost_std,
                f"evaluation_{self.name}/Risk": risk_mean,
                f"evaluation_{self.name}/Risk_std": risk_std,
                f"global_step": total_steps,
            })

        # res_hparams = {'hparam/cost_mean': cost_mean,
        #                'hparam/cost_std': cost_std,
        #                'hparam/risk_mean': risk_mean,
        #                'hparam/risk_std': risk_std,
        #                'hparam/utility_mean': utility_mean,
        #                'hparam/utility_std': utility_std,
        #                'hparam/length_mean': length_mean,
        #                'hparam/length_std': length_std,
        #                }
        # self.writer.add_hparams(self.hparams, res_hparams)
        self.save_model("final")
        self.close_wandb()

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
        self.Pnet.load_state_dict(torch.load(path_pnet))
        self.Vnet.load_state_dict(torch.load(path_vnet))

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
        with torch.no_grad():
            p_ns = self.Vnet(batch.belief_next.squeeze(1))

            non_terminal = torch.logical_not(batch.terminal).unsqueeze(1)
            s_ = batch.reward.shape

            # [Batch C51 nO]
            returns = batch.reward.unsqueeze(1).expand(s_[0], self.c, s_[1]).clone()

            # [C51 nO] + gamma*[C51 nO]*[1 1] -> [C51 nO]
            returns[-1] += self.gamma * self.z * non_terminal[-1]

            for i in range(len(returns) - 1, 0, -1):
                # if episode ended in last n-steps, do not add to return
                returns[i - 1] += self.gamma * returns[i] * non_terminal[i - 1]
            tz = torch.stack(
                [returns[..., o].clamp(min=self.v_min[o], max=self.v_max[o]) for o in range(len(self.v_min))], dim=-1)
            b = (tz - self.v_min) / self.d_z
            l = torch.floor(b).long()
            # change b to not be exactly on border of categories
            b = torch.where(b != l, b, b - self.d_z / 100)
            b = b.clamp(min=0, max=self.c - 1)
            u = torch.ceil(b).long()
            m = torch.zeros_like(p_ns)
            i_s = torch.arange(len(returns))
            # for each objective, for each category, get lower and upper neighbour
            for c_i in product(range(self.c), repeat=self.n_objectives):
                b_i = [b[i_s, c_i[j], j] for j in range(self.n_objectives)]  # b_i..k
                l_i = [l[i_s, c_i[j], j] for j in range(self.n_objectives)]
                u_i = [u[i_s, c_i[j], j] for j in range(self.n_objectives)]
                # (b - floor(b))
                nl_i = [(b_j - l_j) for b_j, l_j in zip(b_i, l_i)]
                # (ceil(b) - b)
                nu_i = [(u_j - b_j) for b_j, u_j in zip(b_i, u_i)]
                lower_or_upper_i = [l_i, u_i]
                lower_or_upper_p = [nu_i, nl_i]
                current_i = (i_s,) + c_i
                # for each combination of lower, upper neighbour, update probabilities
                for n_i in product(range(2), repeat=self.n_objectives):
                    # tuple (Batch, neighbour[0], ..., neighbour[n])
                    neighbour_i = (i_s,) + tuple(lower_or_upper_i[j][i] for i, j in enumerate(n_i))
                    neighbour_p = [lower_or_upper_p[j][i] for i, j in enumerate(n_i)]
                    m[neighbour_i] += p_ns[current_i] * torch.stack(neighbour_p).prod(dim=0)
        p_s = self.Vnet(batch.belief)

        objective_dims = tuple(range(1, len(p_s.shape)))
        critic_loss = -torch.sum(m * torch.log(p_s), dim=objective_dims).unsqueeze(-1)
        with torch.no_grad():
            # expand accrued from [Batch nO] to [Batch 1 .. 1 nO]
            accrued = self.accrued[:-1].view(len(returns), *(1,) * self.n_objectives, self.n_objectives).to(self.device)
            gamma = batch.gamma.view(len(returns), *(1,) * (self.n_objectives + 1))
            # shift back discounted return: accrued + gamma^t*R_t
            accrued_v = accrued + gamma * self.r_z
            u_v_s = self.utility(accrued_v.view(-1, self.n_objectives)).view_as(p_s)
            # expected utility for current state [Batch C51 .. C51]*[Batch C51 .. C51] -> [Batch]
            u_v_s = torch.sum(u_v_s * p_s, dim=objective_dims)
            # get all combinations of n0,n1,... (so C51 goes to c51**nO)
            o_n = torch.meshgrid(*[torch.arange(self.c) for _ in range(self.n_objectives)], indexing="xy")
            # [Batch C51 .. C51 nO]
            r_z = torch.stack(tuple(returns[:, o_i, i] for i, o_i in enumerate(o_n)), dim=-1)
            accrued_r = accrued + gamma * r_z
            # compute the utility for all these returns [Batch C51 .. C51]
            u_r_s = self.utility(accrued_r.view(-1, self.n_objectives)).view_as(p_s)
            # expected utility using n-step returns: [Batch]
            u_r_s = torch.sum(u_r_s * p_ns[-1].unsqueeze(0), dim=objective_dims)
            advantage = u_r_s - u_v_s

        pi_ac_comp, pi_ac_glob = self.Pnet(batch.belief)
        pi_ac_comp = pi_ac_comp.softmax(dim=2)
        pi_ac_glob = pi_ac_glob.softmax(dim=1)

        ind = range(self.n_step_update)
        pi_aa = torch.zeros((self.ncomp + 1, self.n_step_update), device=self.device)
        mu_aa = torch.zeros((self.ncomp + 1, self.n_step_update), device=self.device)
        for k in range(self.ncomp):
            pi_aa[k] = pi_ac_comp[k][ind, batch.action[:,
                                          k]].detach()  # target (current) probability of action retrieved from replay buffer for each component
            mu_aa[k] = batch.behavior_ac_comp[ind, k, list(batch.action[:,
                                                           k])].detach()  # sampled (original) probability of action retrieved from replay buffer for each component

        pi_aa[self.ncomp] = pi_ac_glob[ind, batch.action[:,
                                            self.ncomp]].detach()  # target (current) probability of action retrieved from replay buffer for global component
        mu_aa[self.ncomp] = batch.behavior_ac_glob[ind, 0, list(batch.action[:,
                                                                self.ncomp])].detach()  # sampled (original) probability of action retrieved from replay buffer for global component
        rho = torch.prod(pi_aa / mu_aa, dim=0)  # joint importance weight
        rho = torch.minimum(rho, 2 * torch.ones(self.n_step_update, device=self.device))  # clip importance weight
        advantage = torch.mul(advantage, rho)  # weighted advantage
        log_prob = torch.zeros((self.ncomp + 1, self.n_step_update), device=self.device)

        entropy = 0
        for j in range(self.ncomp):
            dist = Categorical(probs=pi_ac_comp[j])
            log_prob[j] = -dist.log_prob(batch.action[:, j])
            entropy += dist.entropy()
        dist = Categorical(probs=pi_ac_glob)
        log_prob[self.ncomp] = -dist.log_prob(batch.action[:, self.ncomp])
        entropy += dist.entropy()

        actor_loss = torch.sum(log_prob, dim=0) * advantage.detach()

        loss = actor_loss + self.v_coef * critic_loss - self.e_coef * entropy
        self.optim.zero_grad()
        loss = loss.mean()
        loss.backward()

        if self.clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.Pnet.parameters(), self.clip_grad_norm)
            nn.utils.clip_grad_norm_(self.Vnet.parameters(), self.clip_grad_norm)

        self.optim.step()
        self.accrued = self.accrued[-1:]

        return actor_loss.mean().item(), critic_loss.mean().item(), entropy.mean().item(), 1
