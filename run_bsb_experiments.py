import numpy as np
import torch

import pandas as pd
from modcmac_code.environments.Maintenance_Gym import MaintenanceEnv
from modcmac_code.environments.BeliefObservation import BayesianObservation
from modcmac_code.agents.bsb_agent import BSB

ncomp = 13  # number of components
ndeterioration = 50  # number of deterioration modes
ntypes = 3  # number of component types
nstcomp = 5  # number of states per component
naglobal = 2  # number of actions global (inspect X purpose)
# npurpose = 3
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

for i in range(ntypes):
    for j in range(nstcomp):
        if np.sum(P_start[i, j, :]) != 1:
            print('P_start type {} row {} does not sum to 1 with val {}'.format(i, j, np.sum(P_start[i, j, :])))
        P_start[i, j, :] = P_start[i, j, :] / np.sum(P_start[i, j, :])
        if np.sum(P_end[i, j, :]) != 1:
            print('P_end type {} row {} does not sum to 1 with val {}'.format(i, j, np.sum(P_end[i, j, :])))
        P_end[i, j, :] = P_end[i, j, :] / np.sum(P_end[i, j, :])

# P_start = np.zeros((ntypes, nstcomp, nstcomp))
# P_start[0] = np.array([[0.983,  0.0089, 0.0055, 0.0026, 0.],
#                        [0.,     0.9861, 0.0086, 0.0053, 0.],
#                        [0.,     0.,     0.9916, 0.0084, 0.],
#                        [0.,     0.,     0.,     1.,     0.],
#                        [0.,     0.,     0.,     0.,     1.]])
# P_start[1] = np.array([[0.9702, 0.0153, 0.0097, 0.0048, 0.],
#                        [0.,     0.9754, 0.0149, 0.0097, 0.],
#                        [0.,     0.,     0.9851, 0.0149, 0.],
#                        [0.,     0.,     0.,     1.,     0.],
#                        [0.,     0.,     0.,     0.,     1.]])
#
# P_start[2] = np.array([[0.9878, 0.0065, 0.0039, 0.0018, 0.],
#                        [0.,     0.9901, 0.0061, 0.0038, 0.],
#                        [0.,     0.,     0.994,  0.006,  0.],
#                        [0.,     0.,     0.,     1.,     0.],
#                        [0.,     0.,     0.,     0.,     1.]])
#
#
# P_end = np.zeros((ntypes, nstcomp, nstcomp))
# P_end[0] = np.array([[0.9449, 0.0279, 0.0182, 0.009,  0.],
#                      [0.,     0.9541, 0.0278, 0.0181, 0.],
#                      [0.,     0.,     0.9721, 0.0279, 0.],
#                      [0.,     0.,     0.,     1.,     0.],
#                      [0.,     0.,     0.,     0.,     1.]])
#
# P_end[1] = np.array([[0.9347, 0.033,  0.0215, 0.0108, 0.],
#                      [0.,     0.9456, 0.0328, 0.0216, 0.],
#                      [0.,     0.,     0.9669, 0.0331, 0.],
#                      [0.,     0.,     0.,     1.,     0.],
#                      [0.,     0.,     0.,     0.,     1.]])
#
# P_end[2] = np.array([[0.9848, 0.008,  0.0048, 0.0024, 0.],
#                      [0.,     0.9876, 0.0076, 0.0048, 0.],
#                      [0.,     0.,     0.9925, 0.0075, 0.],
#                      [0.,     0.,     0.,     1.,     0.],
#                      [0.,     0.,     0.,     0.,     1.]])

"""
Check if each row in P_start and P_end sums to 1
"""
for i in range(ntypes):
    for j in range(nstcomp):
        if np.sum(P_start[i, j, :]) != 1:
            print('P_start type {} row {} does not sum to 1 with val {}'.format(i, j, np.sum(P_start[i, j, :])))
        if np.sum(P_end[i, j, :]) != 1:
            print('P_end type {} row {} does not sum to 1 with val {}'.format(i, j, np.sum(P_end[i, j, :])))

P = np.zeros((ndeterioration, P_start.shape[0], P_start.shape[1], P_start.shape[2]))
for i in range(ndeterioration):
    P[i, :, :] = P_start + (P_end - P_start) * i / (ndeterioration - 1)

"""
F: failure probability matrix, with dimensions (ntypes, nstcomp)

F is the probability of failure for each component type given the current state, if failed the component stays failed
until replaced
"""
F = np.zeros((ntypes, nstcomp))
F[0] = np.array([0.0008, 0.0046, 0.0123, 0.0259, 1])
F[1] = np.array([0.0012, 0.0073, 0.0154, 0.0324, 1])
F[2] = np.array([0.0019, 0.0067, 0.0115, 0.0177, 1])

"""
Observation matrix
O_no: observation matrix for the no-inspection action
O_in: observation matrix for the inspection action
O is the observation matrix for the inspect, no-inspect and replace action
"""
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
inspect_per = 0.015

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
Sets the type of each componenent. The first 9 components are wooden poles (type 0), 
the next 3 are wooden kesp (type 1) and the last one is a wooden floor (type 2).
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


def fmeca_utility(reward):
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


NUM_RUNS = 1000
NUM_STEPS = 50
# Save path for the deterministic BSB results
SAVE_PATH_DET = None
# Save path for the sampled BSB results
SAVE_PATH_SAMP = None
env = BayesianObservation(MaintenanceEnv(ncomp, ndeterioration, ntypes, nstcomp, naglobal, nacomp, nobs, nfail, P,
                                         O, C_glo, C_rep, comp_setup, f_modes, start_S, total_cost))
policy_det = BSB(env, fmeca_utility, sample=False)
cost_det, risk_det, utility_det = policy_det.do_test(NUM_RUNS, NUM_STEPS)

policy_samp = BSB(env, fmeca_utility, sample=True)
cost_samp, risk_samp, utility_samp = policy_samp.do_test(NUM_RUNS, NUM_STEPS)

if SAVE_PATH_DET is not None:
    df_dict = {'cost': cost_det, 'risk': risk_det, 'utility': utility_det}
    df = pd.DataFrame(df_dict)
    df.to_csv(SAVE_PATH_DET)

if SAVE_PATH_SAMP is not None:
    df_dict = {'cost': cost_samp, 'risk': risk_samp, 'utility': utility_samp}
    df = pd.DataFrame(df_dict)
    df.to_csv(SAVE_PATH_SAMP)
