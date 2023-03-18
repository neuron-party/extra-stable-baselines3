# evaluate average returns on specific levels 
import os
import gym
import procgen
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import argparse

from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.vec_env import VecExtractDictObs, VecMonitor

from networks import *



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained-weights-path', type=str, default='weights/')
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num-eval-levels', type=int, default=1000)
    parser.add_argument('--start-level', type=int, default=0)
    parser.add_argument('--distribution-mode', type=str, default='hard')
    parser.add_argument('--num-envs', type=int, default=64) # should most likely set this to less than 
    parser.add_argument('--save-path', type=str, default='metrics/')
    parser.add_argument('--env-name', type=str, default='coinrun')
    
    args = parser.parse_args()
    return args
    
def initialize_env(num_envs, env_name, num_levels, start_level, distribution_mode): # prob change this, don't need all the extra wrappers for evaluating
    env = procgen.ProcgenEnv(
        num_envs=num_envs,
        env_name=env_name,
        num_levels=1,
        start_level=start_level,
        distribution_mode=distribution_mode
    )
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.NormalizeReward(env, 0.99)
    env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    env.is_vector_env = True
    env = VecMonitor(env)
    return env

def rollout_trajectories(model, env, device, n=100):
    returns = []
    state = env.reset()
    while len(returns) < n:
        state = {'rgb': torch.tensor(state['rgb'], device=device).permute(0, 3, 1, 2)}
        action, _, _ = model(state)
        action = action.detach().cpu().numpy()
        next_state, reward, done, info = env.step(action)
        state = next_state
        
        for item in info:
            if 'episode' in item.keys():
                returns.append(item['episode']['r']) # return
    return returns


def main(args):
    weights = torch.load(args.pretrained_weights_path + '.pth')
    device = torch.device('cuda:' + str(args.device))
    
    eval_level_returns = []
    
    for i in range(args.num_eval_levels):
        env = initialize_env(num_envs=args.num_envs, env_name=args.env_name, num_levels=1, start_level=args.start_level + i, distribution_mode=args.distribution_mode)
        model = SB_Impala(env.observation_space['rgb'], env.action_space.n, 5e-4)
        model.load_state_dict(weights['model'])
        model = model.to(device)
        model.eval()
        average_returns = rollout_trajectories(model=model, env=env, device=device, n=args.n)
        
        eval_level_returns.append(average_returns)
        print(f'Level: {args.start_level + i}, Average Return: {np.mean(average_returns)}')
        
    with open(args.save_path + '.pkl', 'wb') as f:
        pickle.dump(eval_level_returns, f)
        

if __name__ == '__main__':
    args = parse_args()
    main(args)