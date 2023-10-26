import numpy as np
import torch
import os
from modcmac_code.agents.ppo.cppo import CPPO
from modcmac_code.environments.Maintenance_Gym import MaintenanceEnv
from modcmac_code.environments.BeliefObservation import BayesianObservation
from modcmac_code.networks.model import VNet, PNet
import pandas as pd
import argparse
import signal
import sys


def parse_arguments():
    parser = argparse.ArgumentParser(description="MO_DCMAC parameters")
    parser.add_argument("--v_min_cost", type=float, default=-3.0, help="Minimum value of cost.")
    parser.add_argument("--v_max_cost", type=float, default=0.0, help="Maximum value of cost.")
    parser.add_argument("--v_min_risk", type=float, default=-0.5, help="Minimum value of risk.")
    parser.add_argument("--v_max_risk", type=float, default=0.0, help="Maximum value of risk.")
    parser.add_argument("--clip_grad_norm", type=float, default=10, help="Gradient norm clipping value.")
    parser.add_argument("--c", type=int, default=11, help="Number of bins for critic.")
    parser.add_argument("--n_step_update", type=int, default=50, help="Number of steps for update.")
    parser.add_argument("--episode_length", type=int, default=50, help="The length of the episode")
    parser.add_argument("--v_coef", type=float, default=0.5, help="Coefficient for value function.")
    parser.add_argument("--e_coef", type=float, default=0.01, help="Coefficient for entropy.")
    parser.add_argument("--lr_critic", type=float, default=0.0005, help="Learning rate for critic.")
    parser.add_argument("--lr_policy", type=float, default=0.0005, help="Learning rate for policy.")
    parser.add_argument("--no_accrued_reward", action='store_false', help="Flag to use accrued reward.")
    parser.add_argument("--gamma", type=float, default=0.975, help="Discount factor.")
    parser.add_argument("--num_steps", type=int, default=25_000_000, help="Number of training episodes.")
    parser.add_argument("--name", type=str, default="MO_DCMAC", help="Name of the experiment.")
    parser.add_argument("--device", type=str, default="cpu", help="Device.", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--save_folder", type=str, default="./models", help="Folder to save models.")
    parser.add_argument("--do_eval", action='store_true', help="Flag to do evaluation.")
    parser.add_argument("--path_pnet", type=str, default=None, help="Path to policy network.")
    parser.add_argument("--path_vnet", type=str, default=None, help="Path to value network.")
    parser.add_argument("--no_log", action="store_false", help="Flag to not log the run")
    parser.add_argument("--save_eval_folder", type=str, default=None, help="Folder to save evaluation results.")

    args = parser.parse_args()
    if (args.do_eval and
            (args.path_pnet is None or args.path_vnet is None or args.save_eval_folder is None)):
        raise ValueError("You must specify the path to the policy and value networks, and save folder eval.")
    if (args.do_eval and not os.path.exists(args.save_eval_folder)):
        raise ValueError(f"The folder ({args.save_eval_folder}) to save evaluation results does not exist.")
    if args.do_eval and not os.path.exists(args.path_pnet):
        raise ValueError(f"The path ({args.path_pnet}) to the policy network does not exist.")
    if args.do_eval and not os.path.exists(args.path_vnet):
        raise ValueError(f"The path ({args.path_vnet}) to the value network does not exist.")
    return args


ncomp = 13  # number of components
ndeterioration = 50  # number of deterioration steps
ntypes = 3  # number of component types
nstcomp = 5  # number of states per component
naglobal = 2  # number of actions global (inspect X purpose)
nacomp = 3  # number of actions per component
nobs = 5  # number of observations
nfail = 3  # number of failure types

"""
P: transition probability matrix, with dimensions (ndeterioration, ntypes, nstcomp, nstcomp)
P_start: initial transition probability matrix, with dimensions (ntypes, nstcomp, nstcomp)
P_end: final transition probability matrix, with dimensions (ntypes, nstcomp, nstcomp)

The first dimension of P is the deterioration mode, which linear deteriorates from P_start to P_end
"""

P_start = np.zeros((ntypes, nstcomp, nstcomp))
P_start[0] = np.array([
    [0.983, 0.0089, 0.0055, 0.0025, 0.0001],
    [0, 0.9836, 0.0084, 0.0054, 0.0026],
    [0, 0, 0.9862, 0.0084, 0.0054],
    [0, 0, 0, 0.9917, 0.0083],
    [0, 0, 0, 0, 1]
])
P_start[1] = np.array([[0.9748, 0.013, 0.0081, 0.004, 0.0001],
                       [0., 0.9754, 0.0124, 0.0081, 0.0041],
                       [0., 0., 0.9793, 0.0125, 0.0082],
                       [0., 0., 0., 0.9876, 0.0124],
                       [0., 0., 0., 0., 1.]])

P_start[2] = np.array([[0.9848, 0.008, 0.0049, 0.0022, 0.0001],
                       [0., 0.9854, 0.0074, 0.0048, 0.0024],
                       [0., 0., 0.9876, 0.0075, 0.0049],
                       [0., 0., 0., 0.9926, 0.0074],
                       [0., 0., 0., 0., 1.]])

P_end = np.zeros((ntypes, nstcomp, nstcomp))
P_end[0] = np.array([
    [0.9713, 0.0148, 0.0093, 0.0045, 0.0001],
    [0., 0.9719, 0.0142, 0.0093, 0.0046],
    [0, 0, 0.9753, 0.0153, 0.0094],
    [0., 0., 0., 0.9858, 0.0142],
    [0., 0., 0., 0., 1.]
])

P_end[1] = np.array([[0.9534, 0.0237, 0.0153, 0.0075, 0.0001],
                     [0., 0.954, 0.0231, 0.0152, 0.0077],
                     [0., 0., 0.9613, 0.0233, 0.0154],
                     [0., 0., 0., 0.9767, 0.0233],
                     [0., 0., 0., 0., 1.]])

P_end[2] = np.array([[0.9748, 0.013, 0.0081, 0.004, 0.0001],
                     [0., 0.9754, 0.0124, 0.0081, 0.0041],
                     [0., 0., 0.9793, 0.0125, 0.0082],
                     [0., 0., 0., 0.9876, 0.0124],
                     [0., 0., 0., 0., 1.]])

"""
Check if each row in P_start and P_end sums to 1
"""
for i in range(ntypes):
    for j in range(nstcomp):
        if np.sum(P_start[i, j, :]) != 1:
            print('P_start type {} row {} does not sum to 1 with val {}'.format(i, j, np.sum(P_start[i, j, :])))
        P_start[i, j, :] = P_start[i, j, :] / np.sum(P_start[i, j, :])
        if np.sum(P_end[i, j, :]) != 1:
            print('P_end type {} row {} does not sum to 1 with val {}'.format(i, j, np.sum(P_end[i, j, :])))
        P_end[i, j, :] = P_end[i, j, :] / np.sum(P_end[i, j, :])

P = np.zeros((ndeterioration, P_start.shape[0], P_start.shape[1], P_start.shape[2]))
for i in range(ndeterioration):
    P[i, :, :] = P_start + (P_end - P_start) * i / (ndeterioration - 1)

# """
# F: failure probability matrix, with dimensions (ntypes, nstcomp)
#
# F is the probability of failure for each component type given the current state, if failed the component stays failed
# until replaced
# """
# F = np.zeros((ntypes, nstcomp))
# F[0] = np.array([0.0008, 0.0046, 0.0123, 0.0259, 1])
# F[1] = np.array([0.0012, 0.0073, 0.0154, 0.0324, 1])
# F[2] = np.array([0.0019, 0.0067, 0.0115, 0.0177, 1])

"""
Observation matrix
O_no: observation matrix for the no-inspection action
O_in: observation matrix for the inspection action
O is the observation matrix for the inspect, no-inspect and replace action
"""

# O_no = np.array([[1, 0, 0, 0, 0],
#                  [1, 0, 0, 0, 0],
#                  [0, 0.25, 0.5, 0.25, 0],
#                  [0, 0, 0.5, 0.5, 0],
#                  [0, 0, 0, 0, 1]])
# O_no = np.array([[1, 0, 0, 0, 0],
#                  [1, 0, 0, 0, 0],
#                  [1, 0, 0, 0, 0],
#                  [1, 0, 0, 0, 0],
#                  [0, 0, 0, 0, 1]])
O_in = np.eye(nstcomp)
O_no = np.array([[1, 0, 0, 0, 0],
                 [1, 0, 0, 0, 0],
                 [0, 0, 0.34, 0.33, 0.33],
                 [0, 0, 0.34, 0.33, 0.33],

                 [0, 0, 0.34, 0.33, 0.33]])

O = np.zeros((2, nstcomp, nstcomp))
O[0] = O_no
O[1] = O_in

repair_per = 0.25
inspect_per = 0.05

"""
Set the start state of the components
0: No deterioration
1: Small deterioration
2: Large deterioration
3: Near failure
"""
start_state = np.zeros(ncomp, dtype=int)
# Wooden Poles (index 0-8)
start_state[:9] = np.array([3, 3, 2, 3, 2, 2, 3, 2, 3])
# Wooden Kesp (index 9-11)
start_state[9:12] = np.array([2, 3, 2])
# Wooden Floor (index 12)
start_state[12] = np.array([2])
start_S = np.zeros((ncomp, nstcomp))
start_S[np.arange(ncomp), start_state] = 1

"""
TYPE 1: Wooden Pole, N=9, 40% of total cost
TYPE 2: Wooden kesp, N=3, 3.75% of total cost
TYPE 3: Wooden floor, N=1, 11.25% of total cost
"""

total_cost = 1
inspect_cost = 0.005

n_type1 = 9
total_cost_type1 = 0.4 * total_cost
repla_cost_type1 = total_cost_type1 / n_type1
n_type2 = 3
total_cost_type2 = 0.0375 * total_cost
repla_cost_type2 = total_cost_type2 / n_type2
n_type3 = 1
total_cost_type3 = 0.1125 * total_cost
repla_cost_type3 = total_cost_type3 / n_type3

C_glo = np.zeros((1, naglobal))
C_glo[0] = np.array([0, inspect_cost * total_cost])

C_rep = np.zeros((ntypes, nacomp))
C_rep[0] = np.array([0, repair_per * repla_cost_type1, repla_cost_type1])
C_rep[1] = np.array([0, repair_per * repla_cost_type2, repla_cost_type2])
C_rep[2] = np.array([0, repair_per * repla_cost_type3, repla_cost_type3])

"""
Components that will be used for the simulation
Comp: 0, 1 and 2, Wooden Pole connected to Wooden Kesp (9)
Comp: 3, 4 and 5, Wooden Pole connected to Wooden Kesp (10)
Comp: 6, 7 and 8, Wooden Pole connected to Wooden Kesp (11)
Comp: 9 Wooden Kesp connected to Wooden Floor (12)
Comp: 10 Wooden Kesp connected to Wooden Floor (12)
Comp: 11 Wooden Kesp connected to Wooden Floor (12)
Comp: 12 Wooden Floor
"""
comp_setup = np.array(([0] * 9) + ([1] * 3) + [2])

"""
Failure Mode 1: Wooden Pole Failure. 3 substructures (0, 1, 2), (3, 4, 5), (6, 7, 8)
"""
f_mode_1 = np.zeros((3, 3), dtype=int)
f_mode_1[0] = np.array([0, 1, 2])
f_mode_1[1] = np.array([3, 4, 5])
f_mode_1[2] = np.array([6, 7, 8])

"""
Failure Mode 2: Wooden Kesp Failure. 2 substructures (9, 10), (10, 11)
"""
f_mode_2 = np.zeros((2, 2), dtype=int)
f_mode_2[0] = np.array([9, 10])
f_mode_2[1] = np.array([10, 11])

"""
Failure Mode 3: Wooden Floor Failure. 1 substructures (12)
"""
f_mode_3 = np.zeros((1, 1), dtype=int)
f_mode_3[0] = np.array([12])

f_modes = (f_mode_1, f_mode_2, f_mode_3)


def fmeca_log(reward):
    penalty = torch.tensor(3)
    cost = torch.abs(reward[:, 0])
    p_fail = (1 - torch.exp(reward[:, 1]))
    max_cost = torch.tensor(1.5 * total_cost)
    max_fail = torch.tensor(0.1)
    pen_cost = (cost > max_cost)
    pen_risk = (p_fail > max_fail)
    cost_log = (1 + torch.log(1 + (cost / max_cost))) + penalty * pen_cost
    risk_log = (1 + torch.log(1 + (p_fail / max_fail))) + penalty * pen_risk
    uti = -(cost_log * risk_log).view(-1, 1)
    return uti


def fmeca2(reward):
    cost = torch.abs(reward[:, 0])
    p_fail = (1 - torch.exp(reward[:, 1]))
    max_factor = torch.tensor(6)
    rate = torch.tensor(10)
    max_cost = torch.tensor(2 * total_cost)
    max_fail = torch.tensor(0.2)
    penalty = torch.tensor(4)
    pen_cost = (cost > max_cost)
    pen_risk = (p_fail > max_fail)
    cost_log = max_factor * -torch.log10(1 / rate) * torch.log10(1 + (cost / max_cost) * 10) + penalty * pen_cost
    cost_log = torch.clamp(cost_log, min=1)
    risk_log = max_factor * -torch.log10(1 / rate) * torch.log10(1 + (p_fail / max_fail) * 10) + penalty * pen_risk
    risk_log = torch.clamp(risk_log, min=1)
    uti = -(cost_log * risk_log).view(-1, 1)
    return uti


def main():
    args = parse_arguments()
    if args.clip_grad_norm == 0:
        clip_grad_norm = None
    else:
        clip_grad_norm = args.clip_grad_norm
    env = BayesianObservation(MaintenanceEnv(ncomp, ndeterioration, ntypes, nstcomp, naglobal, nacomp, nobs, nfail, P,
                                             O, C_glo, C_rep, comp_setup, f_modes, start_S, total_cost,
                                             ep_length=args.episode_length))
    pnet = PNet(ncomp, nstcomp, nacomp, naglobal, objectives=2, use_accrued_reward=args.no_accrued_reward)
    # exit()
    vnet = VNet(ncomp, nstcomp, c=args.c, objectives=2, use_accrued_reward=args.no_accrued_reward)
    agent = CPPO(pnet, vnet, env, utility=fmeca2, obj_names=['cost', 'risk'], log_run=args.no_log,
                 v_min=(args.v_min_cost, args.v_min_risk), v_max=(args.v_max_cost, args.v_max_risk),
                 clip_grad_norm=clip_grad_norm, c=args.c, device=args.device,
                 v_coef=args.v_coef, e_coef=args.e_coef, lr_critic=args.lr_critic, do_eval_every=1000,
                 lr_policy=args.lr_policy, use_accrued_reward=args.no_accrued_reward, gamma=args.gamma,
                 save_folder=args.save_folder, name=args.name, num_steps=args.num_steps, eval_only=args.do_eval,
                 project_name="CPPO_quay_wall")

    def signal_handler(sig, frame):
        print('You pressed Ctrl+C!')
        agent.close_wandb()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    if not args.do_eval:
        agent.train(training_steps=args.num_steps)
    else:
        agent.load_model(args.path_pnet, args.path_vnet)
        cost_array, risk_array, uti_array, scoring_table = agent.do_eval(5)
        print("Cost: ", np.mean(cost_array))
        print("Risk: ", np.mean(risk_array))
        print("Utility: ", np.mean(uti_array))
        df_dict = {'cost': cost_array, 'risk': risk_array, 'utility': uti_array}
        df = pd.DataFrame(df_dict)
        path_file = os.path.join(args.save_eval_folder, args.name + '_eval.csv')
        path_file_scoring = os.path.join(args.save_eval_folder, args.name + '_scoring.csv')
        df.to_csv(path_file)
        scoring_table.to_csv(path_file_scoring, index=False)


if __name__ == '__main__':
    main()
