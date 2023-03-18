# training script for augmented training and running research experiments
import os
import numpy as np
import gym
import argparse
import procgen
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3.ppo.ppo_research_1 import PPO_ResearchMethod1
from stable_baselines3.ppo.ppo_research_2 import PPO_ResearchMethod2
from stable_baselines3.ppo.ppo_research_3 import PPO_ResearchMethod3
from stable_baselines3.common.vec_env import VecExtractDictObs, VecMonitor

from models.impala import *
from utils.method_utils import *
import random
import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    
    # agent args
    # https://openreview.net/attachment?id=mux7gn3g_3&name=supplementary_material
    parser.add_argument('--n-steps', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--n-epochs', type=int, default=3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--clip-range', type=float, default=0.2)
    parser.add_argument('--clip-range-vf', type=bool, default=None)
    parser.add_argument('--ent-coef', type=float, default=0.01)
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--target-kl', type=float, default=0.01)
    parser.add_argument('--normalize-advantage', type=bool, default=True)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=5e-4)
    
    parser.add_argument('--device', type=int, default=0)
    
    # env args
    parser.add_argument('--env-name', type=str, default='coinrun')
    parser.add_argument('--num-envs', type=int, default=64)
    parser.add_argument('--num-levels', type=int, default=500)
    parser.add_argument('--start-level', type=int, default=0)
    parser.add_argument('--distribution-mode', type=str, default='hard')
    parser.add_argument('--max-global-steps', type=int, default=100000000)
    
    # saving/logging args
    parser.add_argument('--log', type=bool, default=False)
    parser.add_argument('--logging-path', type=str, default=None)
    parser.add_argument('--save-path', type=str, default=None)
    parser.add_argument('--csv-path', type=str, default=None)
    parser.add_argument('--checkpoint-path', type=str, default=None)
    parser.add_argument('--hard-levels-path', type=str, default=None)
    parser.add_argument('--finetuned-weights-path', type=str, default=None)
    parser.add_argument('--load-existing-irm', type=str, default=None)
    
    # research method args
    parser.add_argument('--research-method', type=int, default=1) # 1, 2, or 3 for now 
    
    args = parser.parse_args()
    return args



def main(args):
    env = procgen.ProcgenEnv(
        num_envs=args.num_envs, 
        env_name=args.env_name, 
        num_levels=args.num_levels, 
        start_level=args.start_level,
        distribution_mode=args.distribution_mode
    )
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.NormalizeReward(env, gamma=args.gamma)
    env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    env.is_vector_env = True
    env = VecMonitor(env)
    
    device = torch.device('cuda:' + str(args.device))
    model = SB_Impala(env.observation_space['rgb'], env.action_space.n, args.lr)
    
    with open(args.hard_levels_path + '.pkl', 'rb') as f:
        hard_levels = pickle.load(f)
        
    if args.load_existing_irm:
        with open(args.load_existing_irm + '.pkl', 'rb') as f:
            irm = pickle.load(f)
    else:
        assert args.finetuned_weights_path is not None
        if args.research_method == 1:
            irm = init_research_method_1(
                finetune_weights_path=args.finetuned_weights_path,
                hard_levels=hard_levels,
                model=model,
                device=device
            )
        elif args.research_method == 2 or args.research_method == 3:
            irm = init_research_method_2(
                finetune_weights_path=args.finetuned_weights_path,
                hard_levels=hard_levels,
                model=model,
                device=device
            )
            
    # pop the empty dictionaries (levels that the finetuned agents don't know how to play or failed to learn how to play)
    popkeys = [i for level, info in irm.items() if not info]
    for i in popkeys:
        irm.pop(i)
    level_num_runs = {i : 0 for i in irm}
    level_learned = {i : 0 for i in irm}
    
    if args.research_method == 1:
        agent = PPO_ResearchMethod1(
            policy=model,
            custom_policy=True,
            env=env,
            csv_path=args.csv_path,
            checkpoint_path=args.checkpoint_path,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            clip_range_vf=args.clip_range_vf,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            target_kl=args.target_kl,
            normalize_advantage=args.normalize_advantage,
            max_grad_norm=args.max_grad_norm,
            tensorboard_log=args.logging_path,
            device=device
        )
    elif args.research_method == 2:
        agent = PPO_ResearchMethod2(
            policy=model,
            custom_policy=True,
            env=env,
            csv_path=args.csv_path,
            checkpoint_path=args.checkpoint_path,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            clip_range_vf=args.clip_range_vf,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            target_kl=args.target_kl,
            normalize_advantage=args.normalize_advantage,
            max_grad_norm=args.max_grad_norm,
            tensorboard_log=args.logging_path,
            device=device
        )
    elif args.research_method == 3:
        # research method 3 is easy + hard levels so we need an additional dict for training
        init_easy_level_dict = init_easy_levels_dict(hard_levels, num_levels=args.num_levels, env_name=args.env_name)
        agent = PPO_ResearchMethod3(
            policy=model,
            custom_policy=True,
            env=env,
            csv_path=args.csv_path,
            checkpoint_path=args.checkpoint_path,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            clip_range_vf=args.clip_range_vf,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            target_kl=args.target_kl,
            normalize_advantage=args.normalize_advantage,
            max_grad_norm=args.max_grad_norm,
            tensorboard_log=args.logging_path,
            device=device
        )
    else:
        raise ValueError('only research methods 1, 2, and 3 implemented for now')
    
    # add methods manually cuz faster
    agent.init_research_method = irm
    agent.level_num_runs = level_num_runs
    agent.level_learned = level_learned
    agent.init_easy_level_dict = init_easy_level_dict
    
    agent.learn(total_timesteps=args.max_global_steps)
    
    torch.save({
        'model': agent.policy.state_dict(),
        'optimizer': agent.policy.optimizer.state_dict()
    }, args.save_path + '_final.pth')
    

if __name__ == '__main__':
    args = parse_args()
    main(args)