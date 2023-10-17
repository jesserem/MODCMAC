import sys

import torch
import torch.optim as optim
import os
from .modcmac_base import MODCMACBase
import torch.nn as nn
from ..networks.model import PNet as PnetOrig, VNetSER as VnetOrig
import wandb
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


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    :param y_pred: the prediction
    :param y_true: the expected value
    :return: explained variance of ypred and y
    """
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


class MODCMAC_SER_CAT(MODCMACBase):
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
                 naglobal: int,
                 utility: Callable[[torch.Tensor], torch.Tensor], lr_critic: float = 0.001, lr_policy: float = 0.0001,
                 device: Optional[str] = None, buffer_size: int = 1000, gamma: float = 0.975,
                 name: str = "MODCMAC_SER_CAT",
                 save_folder: str = "./models", use_lr_scheduler: bool = True, num_episodes: int = 500_000,
                 eval_only: bool = False, ep_length: int = 50, v_min: Union[List, float] = -10,
                 v_max: Union[List, float] = 0, c: int = 11, n_step_update: int = 1, v_coef: float = 0.5,
                 e_coef: float = 0.01, clip_grad_norm: Optional[int] = None, do_eval_every: int = 1000,
                 use_accrued_reward: bool = True, n_eval: int = 100, log_run: bool = True):

        super().__init__(pnet, vnet, env, ncomp, nstcomp, nacomp, naglobal, utility, lr_critic, lr_policy, device,
                         buffer_size, gamma, name, save_folder, use_lr_scheduler, num_episodes, eval_only, ep_length,
                         n_step_update, v_coef, e_coef, clip_grad_norm, do_eval_every, use_accrued_reward, n_eval,
                         log_run)

        self.c = c
        self.n_objectives = np.prod(self.env.reward_space.shape)

        if not isinstance(v_min, Iterable):
            v_min = [v_min] * self.n_objectives
        if not isinstance(v_max, Iterable):
            v_max = [v_max] * self.n_objectives
        self.v_min = torch.tensor(v_min).view(1, self.n_objectives, 1).to(self.device)
        self.v_max = torch.tensor(v_max).view(1, self.n_objectives, 1).to(self.device)

        self.d_z = (self.v_max - self.v_min) / (self.c - 1.)
        # [C51 nO]
        self.z = torch.stack([torch.linspace(v_min[i], v_max[i], self.c) for i in range(self.n_objectives)]).to(
            self.device)

        self.hparams = {
            "clip_grad_norm": self.clip_grad_norm,
            "n_step_update": self.n_step_update,
            "v_coef": self.v_coef,
            "e_coef": self.e_coef,
            "gamma": self.gamma,
            "bins": self.c,
            "lr_critic": self.lr_critic,
            "lr_policy": self.lr_policy,
            "use_accrued_reward": self.use_accrued_reward,
            "v_min_cost": v_min[0],
            "v_max_cost": v_max[0],
            "v_min_risk": v_min[1],
            "v_max_risk": v_max[1],
        }

    def learn(self, total_steps=0) -> Tuple[float, float, float, int]:
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
            returns = batch.reward.unsqueeze(2).expand(s_[0], s_[1], self.c).clone()

            returns[-1] += self.gamma * self.z * non_terminal[-1]
            # print(returns.shape)

            for i in range(len(returns) - 1, 0, -1):
                # if episode ended in last n-steps, do not add to return
                returns[i - 1] += self.gamma * returns[i] * non_terminal[i - 1]

            tz = torch.stack(
                [returns[..., o, :].clamp(min=self.v_min[0, o], max=self.v_max[0, o]) for o in
                 range(
                     self.n_objectives
                 )],
                dim=1)

            b = (tz - self.v_min) / self.d_z

            l = torch.floor(b).long()

            b = torch.where(b != l, b, b - self.d_z / 100)
            b = b.clamp(min=0, max=self.c - 1)

            u = torch.ceil(b).long()

            m = torch.zeros_like(p_ns)

            lower = p_ns * (u - b)
            upper = p_ns * (b - l)
            m.scatter_add_(2, l, lower)
            m.scatter_add_(2, u, upper)

        p_s = self.Vnet(batch.belief)

        objective_dims = tuple(range(1, len(p_s.shape)))
        critic_loss = -torch.sum(m * torch.log(p_s), dim=objective_dims).unsqueeze(-1)
        # print(critic_loss)
        with torch.no_grad():

            value_next = torch.sum(self.z * p_ns, dim=2)
            value_curr = torch.sum(self.z * p_s, dim=2)
            gamma = batch.gamma.unsqueeze(1)

        uti_next = self.utility(
            (self.accrued[:-1] + gamma * batch.reward) + ((self.gamma * gamma) * value_next))
        uti_curr = self.utility(self.accrued[:-1] + (gamma * value_curr.detach()))

        advantage = uti_next - uti_curr

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
        # print(critic_loss.shape)
        loss = actor_loss + self.v_coef * critic_loss - self.e_coef * entropy
        self.optim.zero_grad()
        loss = loss.mean()
        loss.backward()

        if self.clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.Pnet.parameters(), self.clip_grad_norm)
            nn.utils.clip_grad_norm_(self.Vnet.parameters(), self.clip_grad_norm)

        self.optim.step()
        self.accrued = self.accrued[-1:]
        exp_var_arr = np.zeros(self.n_objectives)

        for i in range(self.n_objectives):
            exp_var_arr[i] = explained_variance(value_curr[:, i].detach().numpy(), returns[:, i, 0].numpy())

        if self.log_run:
            exp_var_arr = np.zeros(self.n_objectives)

            for i in range(self.n_objectives):
                exp_var_arr[i] = explained_variance(value_curr[:, i].detach().numpy(), returns[:, i, 0].numpy())
            res_dict = {
                f"debugging_{self.name}/explained_variance_cost": exp_var_arr[0],
                f"global_step": self.total_steps
            }
            if not np.isnan(exp_var_arr).any():
                res_dict[f"debugging_{self.name}/explained_variance_risk"] = exp_var_arr[1]
                res_dict[f"debugging_{self.name}/explained_variance"] = np.mean(exp_var_arr)
            wandb.log(res_dict)

        return actor_loss.mean().item(), critic_loss.mean().item(), entropy.mean().item(), 1
