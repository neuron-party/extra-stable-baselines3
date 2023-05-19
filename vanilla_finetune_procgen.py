import os
import numpy as np
import argparse
import gym
import procgen
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
import copy

from stable_baselines3.common.vec_env import VecExtractDictObs, VecMonitor
# from stable_baselines3.ppo.ppo_supervised_finetuning import PPOSupervisedFinetuning
from stable_baselines3.ppo.ppo_vanilla_finetuning import PPOVanillaFinetuning

# from networks import *
# from supervised_utils import *
# from method_utils import *
from models.impala import *
from utils.supervised_utils import *
from utils.method_utils import *


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
    
    # finetuning env args
    parser.add_argument('--env-name', type=str, default='coinrun')
    parser.add_argument('--num-envs', type=int, default=64)
    parser.add_argument('--num-levels', type=int, default=1)
    parser.add_argument('--distribution-mode', type=str, default='hard')
    parser.add_argument('--max-global-steps', type=int, default=2000000)
    parser.add_argument('--start-level', type=int, default=0)
    
    # saving/logging args
    parser.add_argument('--log', type=bool, default=False)
    parser.add_argument('--logging-path', type=str, default=None)
    parser.add_argument('--save-path', type=str, default='agent.pth')
    parser.add_argument('--pretrained-weights-path', type=str, default='weights/')
    parser.add_argument('--hard-levels-path', type=str, default=None)
    parser.add_argument('--trajectory-buffer-save-path', type=str, default=None)
    
    args = parser.parse_args()
    return args


def main(args):
    device = torch.device('cuda:' + str(args.device))
    
    weights = torch.load(args.pretrained_weights_path + '.pth')
        
    with open(args.hard_levels_path + '.pkl', 'rb') as f:
        hard_levels = pickle.load(f)
    
    if args.trajectory_buffer_save_path:
        trajectory_buffer_save_path = {}
    
    for hard_level in hard_levels:
        env = procgen.ProcgenEnv(
            num_envs=args.num_envs, 
            env_name=args.env_name, 
            num_levels=args.num_levels, # can change this to 1
            start_level=hard_level,
            distribution_mode=args.distribution_mode
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.NormalizeReward(env, gamma=args.gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        env.is_vector_env = True
        env = VecMonitor(env)
    
        model = SB_Impala(env.observation_space['rgb'], env.action_space.n, args.lr)
    
        agent = PPOVanillaFinetuning(
            policy=model,
            custom_policy=True,
            env=env,
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
        
        agent.policy.load_state_dict(weights['model'])
        agent.learn(total_timesteps=args.max_global_steps)
    
        torch.save({
            'model': agent.policy.state_dict(),
            'optimizer': agent.policy.optimizer.state_dict()
        }, args.save_path + '_' + str(hard_level) + '_finetune.pth')
        
        if args.trajectory_buffer_save_path:
            trajectory_buffer_save_path[hard_level] = {
                'array buffer': copy.deepcopy(agent.good_trajectory_buffer),
                'bytes buffer': copy.deepcopy(agent.good_trajectory_bytes_buffer)
            }
    
    if args.trajectory_buffer_save_path:
        with open(args.trajectory_buffer_save_path + '.pkl', 'wb') as f:
            pickle.dump(trajectory_buffer_save_path, f)
    

if __name__ == '__main__':
    args = parse_args()
    main(args)