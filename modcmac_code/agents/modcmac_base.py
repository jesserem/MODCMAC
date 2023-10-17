import sys

import pandas as pd
import torch
import torch.optim as optim
import os
import numpy as np
from torch.distributions import Categorical
from ..replaybuffer.ReplayBuffer import Memory as ReplayBuffer, Transition
from datetime import datetime
from time import time
from gymnasium import Env
from typing import Optional, Union, List, Callable, Tuple
import wandb


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

    def __init__(self, pnet: torch.nn.Module, vnet: torch.nn.Module, env: Env, ncomp: int, nstcomp: int, nacomp: int,
                 naglobal: int, utility: Callable[[torch.Tensor], torch.Tensor], lr_critic: float = 0.001,
                 lr_policy: float = 0.0001, device: Optional[str] = None, buffer_size: int = 1000, gamma: float = 0.975,
                 name: str = "MODCMAC_base", save_folder: str = "./models", use_lr_scheduler: bool = True,
                 num_episodes: int = 500_000, eval_only: bool = False, ep_length: int = 50, n_step_update: int = 1,
                 v_coef: float = 0.5, e_coef: float = 0.01, clip_grad_norm: Optional[int] = None,
                 do_eval_every: int = 1000, use_accrued_reward: bool = True, n_eval: int = 100, log_run: bool = True):

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
        self.total_steps = 0
        self.do_eval_every = do_eval_every

        self.env = env
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

        self.Pnet = pnet.to(self.device)
        # exit()
        self.Vnet = vnet.to(self.device)
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

        self.accrued = torch.tensor([]).view(0, self.n_objectives)
        self.hparams = None

    def init_wandb(self) -> None:
        wandb.init(
            project="modcmac",
            name=self.name,
            config=self.hparams,
            save_code=True,
        )
        wandb.define_metric("*", step_metric="global_step")

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

    def evaluate(self, scoring_table: Optional[np.ndarray] = None, run: int = 0) -> Tuple[
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
        reward = np.zeros(self.n_objectives)
        while not done:
            action, behavior_ac_comp, behavior_ac_glob = self.select_action(observation,
                                                                            training=False)  # select action
            # columns = ["global_step", "timestep", "accrued_cost", "accrued_risk"]

            next_belief, cost, done, trunc, info = self.env.step(action)
            reward += cost
            # for i in range(self.ncomp):
            #     columns.append(f"comp_{i + 1}_state")
            # for i in range(self.ncomp):
            #     columns.append(f"comp_{i + 1}_belief")
            # for i in range(self.ncomp):
            #     columns.append(f"comp_{i + 1}_action")
            # columns.append("global_action")
            # state_input = (np.argmax(info["state"], axis=1).flatten() + 1).tolist()
            # print(state_input)
            # belief_input = belief.flatten().tolist()
            # print(belief_input)
            # action_input = action.flatten().tolist()
            # print(action_input)
            # print(len(state_input))
            # print(len(belief_input))
            # print(len(action_input))
            # exit()
            if scoring_table is not None:
                state_input = (np.argmax(info["state"], axis=1).flatten() + 1).tolist()
                belief_input = belief.flatten().tolist()
                action_input = action.flatten().tolist()
                scoring_table[i, 0] = run
                scoring_table[i, 1] = i
                scoring_table[i, 2] = reward[0]
                scoring_table[i, 3] = reward[1]
                # scoring_table[i, 4:4 + self.ncomp] = state_input
                for s in range(len(state_input)):
                    scoring_table[i, 4 + s] = state_input[s]
                for b in range(len(belief_input)):
                    scoring_table[i, 4 + len(state_input) + b] = belief_input[b]
                for a in range(len(action_input)):
                    scoring_table[i, 4 + len(state_input) + len(belief_input) + a] = action_input[a]
            belief = next_belief

            # scoring_table.add_data(run, i, reward[0], reward[1], *state_input, *belief_input,
            #                        *action_input)
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

        return reward, i, has_failed, scoring_table

    def do_eval(self, episodes: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
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
        list_of_scoring_tables = []
        columns = ["episode", "timestep", "accrued_cost", "accrued_risk"]
        for i in range(self.ncomp):
            columns.append(f"comp_{i + 1}_state")
        for i in range(self.ncomp):
            for j in range(self.nstcomp):
                columns.append(f"comp_{i + 1}_state_{j + 1}_belief")
        for i in range(self.ncomp):
            columns.append(f"comp_{i + 1}_action")
        columns.append("global_action")
        for i in range(episodes):
            curr_scoring_table = np.zeros((self.ep_length, len(columns)))
            reward, _, _, scoring_table = self.evaluate(scoring_table=curr_scoring_table, run=i)
            # print(curr_scoring_table)
            # exit()
            list_of_scoring_tables.append(scoring_table)
            total_cost_eval_curr = np.abs(reward[0])
            total_prob_eval_curr = 1 - np.exp(reward[1])
            eval_cost_curr = np.abs(self.utility(torch.tensor(reward).view(1, self.n_objectives)).item())
            reward_cost_array[i] = total_cost_eval_curr
            reward_risk_array[i] = total_prob_eval_curr
            reward_uti_array[i] = eval_cost_curr
        scoring_table = np.vstack(list_of_scoring_tables)
        scoring_table = pd.DataFrame(scoring_table, columns=columns)

        return reward_cost_array, reward_risk_array, reward_uti_array, scoring_table

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

        total_ep_cost = np.zeros(episodes + 1)  # episode cost (life-cycle cost)
        total_manpower = np.zeros(episodes + 1)
        start_time = time()
        if self.log_run:
            self.init_wandb()
        columns = ["episode", "timestep", "accrued_cost", "accrued_risk"]
        for i in range(self.ncomp):
            columns.append(f"comp_{i + 1}_state")
        for i in range(self.ncomp):
            for j in range(self.nstcomp):
                columns.append(f"comp_{i + 1}_state_{j + 1}_belief")
        for i in range(self.ncomp):
            columns.append(f"comp_{i + 1}_action")
        columns.append("global_action")

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
                if (self.total_steps + 1) % self.n_step_update == 0:
                    loss_p, loss_v, entropy, update_loss = self.learn()
                    if self.log_run:
                        wandb.log({
                            f"losses_{self.name}/policy": loss_p,
                            f"losses_{self.name}/value": loss_v,
                            f"losses_{self.name}/entropy": entropy,
                            f"global_step": self.total_steps,
                        })
                i += 1
                self.total_steps += 1

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
                    reward, n, has_failed, _ = self.evaluate()
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
                        f"global_step": self.total_steps,
                    })

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
            curr_total_uti_discounted = self.utility(torch.tensor(total_cost_discounted).view(1, self.n_objectives))
            curr_total_cost_discounted = np.abs(total_cost_money_discounted)
            curr_total_risk_discounted = 1 - np.exp(total_cost_manpower_discounted)
            if self.log_run:
                wandb.log({
                    f"training_{self.name}/Cost": np.abs(total_ep_cost[ep]),
                    f"training_{self.name}/Collapse_prob": curr_total_prob,
                    f"training_{self.name}/Utility": np.abs(curr_total_uti),
                    f"training_discounted_{self.name}/Utility": np.abs(curr_total_uti_discounted),
                    f"training_discounted_{self.name}/Cost": curr_total_cost_discounted,
                    f"training_discounted_{self.name}/Collapse_prob": curr_total_risk_discounted,
                    f"learning_rate_{self.name}/actor": self.optim.param_groups[0]["lr"],
                    f"learning_rate_{self.name}/critic": self.optim.param_groups[1]["lr"],
                    f"global_step": self.total_steps,
                })

            if self.use_lr_scheduler:
                self.lr_scheduler.step()

        reward_cost_array = np.zeros(self.n_eval)
        reward_risk_array = np.zeros(self.n_eval)
        reward_uti_array = np.zeros(self.n_eval)
        reward_len_array = np.zeros(self.n_eval)

        for i in range(self.n_eval):
            reward, n, has_failed, _ = self.evaluate()
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
                f"global_step": self.total_steps,
            })

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
        raise NotImplementedError
