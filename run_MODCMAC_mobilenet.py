import numpy as np
import torch
import os
from modcmac_code.agents.modcmac import MODCMAC
from modcmac_code.environments.MOMobileNetWrapper import MOMobileNetWrapper
from modcmac_code.networks.model import VNetMobile, PNetMobile
import gymnasium as gym
import mobile_env
from modcmac_code.environments.MOMobileNetWrapper import MOMobileNetWrapper
import pandas as pd
import argparse
import signal
import sys


def parse_arguments():
    parser = argparse.ArgumentParser(description="MO_DCMAC parameters")
    parser.add_argument("--v_min", type=float, nargs="+", default=[-3.0], help="V_min.")
    parser.add_argument("--v_max", type=float, nargs="+", default=[0.0], help="Maximum value of cost.")
    parser.add_argument("--clip_grad_norm", type=float, default=10, help="Gradient norm clipping value.")
    parser.add_argument("--c", type=int, default=11, help="Number of bins for critic.")
    parser.add_argument("--n_step_update", type=int, default=50, help="Number of steps for update.")
    parser.add_argument("--v_coef", type=float, default=0.5, help="Coefficient for value function.")
    parser.add_argument("--e_coef", type=float, default=0.01, help="Coefficient for entropy.")
    parser.add_argument("--lr_critic", type=float, default=0.0005, help="Learning rate for critic.")
    parser.add_argument("--lr_policy", type=float, default=0.0005, help="Learning rate for policy.")
    parser.add_argument("--no_accrued_reward", action='store_false', help="Flag to use accrued reward.")
    parser.add_argument("--gamma", type=float, default=0.999, help="Discount factor.")
    parser.add_argument("--episodes", type=int, default=500_000, help="Number of training episodes.")
    parser.add_argument("--name", type=str, default="MO_DCMAC", help="Name of the experiment.")
    parser.add_argument("--device", type=str, default="cpu", help="Device.", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--save_folder", type=str, default="./models", help="Folder to save models.")
    parser.add_argument("--do_eval", action='store_true', help="Flag to do evaluation.")
    parser.add_argument("--path_pnet", type=str, default=None, help="Path to policy network.")
    parser.add_argument("--path_vnet", type=str, default=None, help="Path to value network.")
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


def test_utility(reward):
    return reward.view(-1, 1)


def main():
    args = parse_arguments()
    if args.clip_grad_norm == 0:
        clip_grad_norm = None
    else:
        clip_grad_norm = args.clip_grad_norm
    env = MOMobileNetWrapper(gym.make('mobile-small-central-v0'))
    pnet = PNetMobile(input_size=env.observation_space.shape[0], n_actions=env.action_space.nvec.max(),
                      n_heads=env.action_space.shape[0], use_accrued_reward=args.no_accrued_reward)
    vnet = VNetMobile(input_size=env.observation_space.shape[0], objectives=1,
                      use_accrued_reward=args.no_accrued_reward)
    agent = MODCMAC(pnet, vnet, env, utility=test_utility, v_min=args.v_min, v_max=args.v_max,
                    clip_grad_norm=clip_grad_norm, c=args.c, device=args.device, n_step_update=args.n_step_update,
                    v_coef=args.v_coef, e_coef=args.e_coef, lr_critic=args.lr_critic,
                    lr_policy=args.lr_policy, use_accrued_reward=args.no_accrued_reward, gamma=args.gamma,
                    save_folder=args.save_folder, name=args.name, num_episodes=args.episodes, eval_only=args.do_eval)

    def signal_handler(sig, frame):
        print('You pressed Ctrl+C!')
        agent.close_wandb()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    if not args.do_eval:
        agent.train(episodes=args.episodes)
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
