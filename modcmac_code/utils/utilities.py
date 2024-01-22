import torch


def fmeca2(reward):
    cost = torch.abs(reward[:, 0])
    p_fail = (1 - torch.exp(reward[:, 1]))
    max_factor = torch.tensor(6)
    rate = torch.tensor(10)
    max_cost = torch.tensor(4)
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


def fmeca2_round(reward):
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
    cost_log = torch.round(torch.clamp(cost_log, min=1), decimals=0)
    risk_log = max_factor * -torch.log10(1 / rate) * torch.log10(1 + (p_fail / max_fail) * 10) + penalty * pen_risk
    risk_log = torch.round(torch.clamp(risk_log, min=1), decimals=0)
    uti = -(cost_log * risk_log).view(-1, 1)
    return uti


def other_uti(reward):
    cost = torch.abs(reward[:, 0])
    p_fail = (1 - torch.exp(reward[:, 1]))
    p_tensor = torch.zeros_like(p_fail)
    p_tensor_extra = torch.zeros_like(p_fail)
    p_tensor_extra[p_fail >= 0.1] = 1
    p_tensor_extra[p_fail >= 0.2] = 2
    p_tensor = p_tensor + 5

    p_tensor[p_fail < 0.2] = 3
    p_tensor[p_fail < 0.1] = 1
    uti = -((cost + p_tensor_extra) * p_tensor).view(-1, 1)
    return uti


def other_uti_simple_env(reward):
    cost = torch.abs(reward[:, 0])
    p_fail = (1 - torch.exp(reward[:, 1]))
    p_tensor = torch.zeros_like(p_fail)
    p_tensor_extra = torch.zeros_like(p_fail)
    p_tensor_extra[p_fail >= 0.2] = 1
    p_tensor_extra[p_fail >= 0.3] = 2
    p_tensor = p_tensor + 5

    p_tensor[p_fail < 0.3] = 3
    p_tensor[p_fail < 0.2] = 1
    uti = -((cost + p_tensor_extra) * p_tensor).view(-1, 1)
    return uti


def other_uti_smooth(reward):
    cost = torch.abs(reward[:, 0])
    p_fail = (1 - torch.exp(reward[:, 1]))
    uti = torch.zeros_like(cost)
    uti[p_fail < 0.1] = -(cost[p_fail < 0.1] + (p_fail[p_fail < 0.1]))
    uti[p_fail >= 0.1] = -(cost[p_fail >= 0.1] + torch.pow(1 + p_fail[p_fail >= 0.1], 2) + 4)

    return uti.view(-1, 1)
