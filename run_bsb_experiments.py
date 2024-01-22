import numpy as np
import torch

import pandas as pd
from modcmac_code.environments.Maintenance_Gym import MaintenanceEnv
from modcmac_code.environments.BeliefObservation import BayesianObservation
from modcmac_code.agents.bsb_agent import BSB
import gymnasium as gym


def fmeca2(reward):
    cost = torch.abs(reward[:, 0])
    p_fail = (1 - torch.exp(reward[:, 1]))
    max_factor = torch.tensor(6)
    rate = torch.tensor(10)
    max_cost = torch.tensor(2)
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
env = gym.make("Maintenance-quay-wall-v0")
env = BayesianObservation(env)
policy_det = BSB(env, fmeca2, sample=False)
cost_det, risk_det, utility_det = policy_det.do_test(NUM_RUNS, NUM_STEPS)

policy_samp = BSB(env, fmeca2, sample=True)
cost_samp, risk_samp, utility_samp = policy_samp.do_test(NUM_RUNS, NUM_STEPS)

if SAVE_PATH_DET is not None:
    df_dict = {'cost': cost_det, 'risk': risk_det, 'utility': utility_det}
    df = pd.DataFrame(df_dict)
    df.to_csv(SAVE_PATH_DET)

if SAVE_PATH_SAMP is not None:
    df_dict = {'cost': cost_samp, 'risk': risk_samp, 'utility': utility_samp}
    df = pd.DataFrame(df_dict)
    df.to_csv(SAVE_PATH_SAMP)
