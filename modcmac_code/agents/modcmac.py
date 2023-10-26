import torch
from .modcmac_base import MODCMACBase
from collections.abc import Iterable
from itertools import product
from gymnasium import Env
from ..replaybuffer.ReplayBuffer import CPPOReplayBuffer
from typing import Optional, Union, List, Callable


class MODCMAC(MODCMACBase):
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
                 utility: Callable[[torch.Tensor], torch.Tensor], lr_critic: float = 0.001, lr_policy: float = 0.0001,
                 device: Optional[str] = None, buffer_size: int = 1000, gamma: float = 0.975, name: str = "MO_DCMAC",
                 save_folder: str = "./models", use_lr_scheduler: bool = True, num_steps: int = 500_000,
                 eval_only: bool = False, ep_length: int = 50, v_min: Union[List, float] = -10,
                 v_max: Union[List, float] = 0, c: int = 11, n_step_update: int = 1, v_coef: float = 0.5,
                 e_coef: float = 0.01, clip_grad_norm: Optional[int] = None, do_eval_every: int = 1000,
                 use_accrued_reward: bool = True, n_eval: int = 100, log_run: bool = True,
                 obj_names: Optional[List[str]] = None, project_name: str = "modcmac", continuous: bool = False,
                 normalize_advantage: bool = True, seed: Optional[int] = None):
        super().__init__(pnet=pnet,
                         vnet=vnet,
                         env=env,
                         normalize_advantage=normalize_advantage,
                         utility=utility,
                         lr_critic=lr_critic,
                         lr_policy=lr_policy,
                         device=device,
                         buffer_size=buffer_size,
                         gamma=gamma,
                         name=name,
                         save_folder=save_folder,
                         use_lr_scheduler=use_lr_scheduler,
                         num_steps=num_steps,
                         eval_only=eval_only,
                         ep_length=ep_length,
                         n_step_update=n_step_update,
                         v_coef=v_coef,
                         e_coef=e_coef,
                         seed=seed,
                         clip_grad_norm=clip_grad_norm,
                         do_eval_every=do_eval_every,
                         use_accrued_reward=use_accrued_reward,
                         n_eval=n_eval,
                         log_run=log_run,
                         obj_names=obj_names,
                         project_name=project_name,
                         continuous=continuous)

        self.c = c

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
            "c": self.c,
            "lr_critic": self.lr_critic,
            "lr_policy": self.lr_policy,
            "use_accrued_reward": self.use_accrued_reward,
            "v_min": v_min,
            "v_max": v_max,
            "normalize_advantage": self.normalize_advantage,
            "use_lr_scheduler": self.use_lr_scheduler,
            "seed": self.seed,
        }

    def calculate_target(self, p_ns: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
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
        return m

    def calculate_advantage(self, p_s: torch.Tensor, p_ns: torch.Tensor, returns: torch.Tensor, batch) -> torch.Tensor:
        objective_dims = tuple(range(1, len(p_s.shape)))
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
        return advantage

    def learn_critic(self, batch):
        with torch.no_grad():
            p_ns = self.Vnet(batch.next_observation.squeeze(1))

            non_terminal = torch.logical_not(batch.terminal).unsqueeze(1)
            s_ = batch.reward.shape

            # [Batch C51 nO]
            returns = batch.reward.unsqueeze(1).expand(s_[0], self.c, s_[1]).clone()

            # [C51 nO] + gamma*[C51 nO]*[1 1] -> [C51 nO]
            returns[-1] += self.gamma * self.z * non_terminal[-1]

            for i in range(len(returns) - 1, 0, -1):
                # if episode ended in last n-steps, do not add to return
                returns[i - 1] += self.gamma * returns[i] * non_terminal[i - 1]
            # print("returns original", returns[0])

            m = self.calculate_target(p_ns, returns)

        p_s = self.Vnet(batch.observation)

        objective_dims = tuple(range(1, len(p_s.shape)))

        critic_loss = -torch.sum(m * torch.log(p_s), dim=objective_dims).unsqueeze(-1)

        with torch.no_grad():
            advantage = self.calculate_advantage(p_s, p_ns, returns, batch)
        # print("advantage", advantage)

        return critic_loss, advantage
