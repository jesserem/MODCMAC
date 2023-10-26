import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import torch
import mo_gymnasium as mo_gym
from modcmac_code.environments.BeliefObservation import BayesianObservation
import numpy as np
import wandb
import yaml
from time import sleep
from modcmac_code.agents.modcmac import MODCMAC
from modcmac_code.networks.model import PNet, VNet
from mo_gymnasium.utils import MORecordEpisodeStatistics

from modcmac_code.utils.utils import reset_wandb_env, seed_everything


@dataclass
class WorkerInitData:
    sweep_id: str
    seed: int
    config: dict
    worker_num: int


@dataclass
class WorkerDoneData:
    utility: float


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-entity", type=str, help="Wandb entity to use for the sweep", required=False)
    parser.add_argument("--project-name", type=str, help="Project name to use for the sweep",
                        default="modcmac_quay_wall_hp_sweep")

    parser.add_argument("--sweep-count", type=int, help="Number of trials to do in the sweep worker", default=1)
    parser.add_argument("--num-seeds", type=int, help="Number of seeds to use for the sweep", default=1)

    parser.add_argument(
        "--seed", type=int, help="Random seed to start from, seeds will be in [seed, seed+num-seeds)", default=10
    )

    parser.add_argument("--config-name", type=str, help="Name of the config to use for the sweep.")

    args = parser.parse_args()

    if not args.config_name:
        args.config_name = f"{args.algo}.yaml"
    elif not args.config_name.endswith(".yaml"):
        args.config_name += ".yaml"

    return args


# def train(worker_data: WorkerInitData) -> WorkerDoneData:
#     # Reset the wandb environment variables
#
#     reset_wandb_env()
#
#     seed = worker_data.seed
#     group = worker_data.sweep_id
#     config = worker_data.config
#     worker_num = worker_data.worker_num
#
#     # Set the seed
#     seed_everything(seed)
#
#     print(f"Worker {worker_num}: Seed {seed}.")
#     name = f"hp_tune_{worker_num}"
#     env = mo_gym.make("Maintenance-quay-wall-v0")
#     env = BayesianObservation(env)
#     pnet = PNet(env.observation_space, env.action_space, use_accrued_reward=True, objectives=2)
#     vnet = VNet(env.observation_space, c=config["c"], use_accrued_reward=True, objectives=2)
#     # print("HIJ KOMT HIER")
#     algo = MODCMAC(pnet, vnet, env, utility=fmeca2, obj_names=['cost', 'risk'], log_run=True,
#                    v_min=(config["v_min_cost"], config["v_min_risk"]), v_max=(0, 0),
#                    clip_grad_norm=config["clip_grad_norm"], c=config["c"], device="cpu",
#                    n_step_update=config["n_step_update"], v_coef=config["v_coef"], e_coef=config["e_coef"],
#                    lr_critic=config["lr_critic"], do_eval_every=1000, lr_policy=config["lr_policy"],
#                    use_accrued_reward=True, gamma=config["gamma"], seed=seed, save_folder="./model_tuning",
#                    name=name, num_steps=1_000_000, eval_only=False, project_name="modcmac_quay_wall_hp_sweep")
#
#     # Launch the agent training
#
#     print(f"Worker {worker_num}: Seed {seed}. Training agent...")
#     algo.train(training_steps=1_000_000)
#
#     # Get the hypervolume from the wandb run
#     utility = wandb.run.summary["evaluation/Utility"]
#     print(f"Worker {worker_num}: Seed {seed}. Utility: {utility}")
#
#     return utility

def train():
    # Reset the wandb environment variables
    configs = {
        "c": 11,
        "gamma": 0.975,
        "v_coef": 0.5,
        "e_coef": 0.01,
        "lr_critic": 0.001,
        "lr_policy": 0.0001,
        "clip_grad_norm": 10,
        "v_min_cost": -8,
        "v_min_risk": -0.3,
        "n_step_update": 32,
        "normalize_advantage": True,
        "use_lr_scheduler": True,
    }
    wandb.init(config=configs, project="modcmac_quay_wall_hp_sweep")

    reset_wandb_env()
    config = wandb.config

    name = f"hp_tune"
    env = mo_gym.make("Maintenance-quay-wall-v0")
    env = BayesianObservation(env)
    pnet = PNet(env.observation_space, env.action_space, use_accrued_reward=True, objectives=2)
    vnet = VNet(env.observation_space, c=config["c"], use_accrued_reward=True, objectives=2)
    # print("HIJ KOMT HIER")
    algo = MODCMAC(pnet, vnet, env, utility=fmeca2, obj_names=['cost', 'risk'], log_run=True,
                   v_min=(config["v_min_cost"], config["v_min_risk"]), v_max=(0, 0),
                   clip_grad_norm=config["clip_grad_norm"], c=config["c"], device="cpu",
                   n_step_update=config["n_step_update"], v_coef=config["v_coef"], e_coef=config["e_coef"],
                   lr_critic=config["lr_critic"], do_eval_every=1000, lr_policy=config["lr_policy"],
                   use_accrued_reward=True, gamma=config["gamma"], save_folder="./model_tuning",
                   normalize_advantage=config["normalize_advantage"], use_lr_scheduler=config["use_lr_scheduler"],
                   name=name, num_steps=1_000_000, eval_only=False, project_name="modcmac_quay_wall_hp_sweep")

    # Launch the agent training

    print(f"Starting")
    algo.train(training_steps=1_000_000)

    # Get the hypervolume from the wandb run
    utility = wandb.run.summary["evaluation/Utility"]
    # print(f"Worker {worker_num}: Seed {seed}. Utility: {utility}")

    # return utility


# def main():
#     # Get the sweep id
#     sweep_run = wandb.init()
#
#     # Spin up workers before calling wandb.init()
#     # Workers will be blocked on a queue waiting to start
#     with ProcessPoolExecutor(max_workers=args.num_seeds) as executor:
#         futures = []
#         for num in range(args.num_seeds):
#             # print("Spinning up worker {}".format(num))
#             seed = seeds[num]
#             # print(dict(sweep_run.config))
#             futures.append(
#                 executor.submit(
#                     train, WorkerInitData(sweep_id=sweep_id, seed=seed, config=dict(sweep_run.config), worker_num=num)
#                 )
#             )
#             # sleep(60)
#
#         # Get results from workers
#         results = [future.result() for future in futures]
#
#     # Get the hypervolume from the results
#     utility_metrics = [result.utility for result in results]
#     print(f"Utility of the sweep {sweep_id}: {utility_metrics}")
#
#     # Compute the average hypervolume
#     average_utility = sum(utility_metrics) / len(utility_metrics)
#     print(f"Average utility of the sweep {sweep_id}: {average_utility}")
#
#     # Log the average hypervolume to the sweep run
#     sweep_run.log(dict(avg_utility=average_utility))
#     wandb.finish()


args = parse_args()

# Create an array of seeds to use for the sweep
seeds = [args.seed + i for i in range(args.num_seeds)]

# Load the sweep config
config_file = os.path.join(os.path.dirname(__file__), "configs", args.config_name)

# Set up the default hyperparameters
with open(config_file) as file:
    sweep_config = yaml.load(file, Loader=yaml.FullLoader)

# print()

# Set up the sweep
sweep_id = wandb.sweep(sweep=sweep_config, entity=args.wandb_entity, project=args.project_name)
# print(sweep_id)
# main()
#
# # Run the sweep agent
wandb.agent(sweep_id, function=train(), count=args.sweep_count)
