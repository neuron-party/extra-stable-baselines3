# evaluate average returns on specific levels 
# separate script for evaluating ood levels since licheng wants it done faster XD
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

# from networks import *
from models.impala import *
from utils.method_utils import *



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained-weights-path', type=str, default='weights/')
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--distribution-mode', type=str, default='hard')
    parser.add_argument('--num-envs', type=int, default=64) # should most likely set this to less than 
    parser.add_argument('--save-path', type=str, default='metrics/')
    parser.add_argument('--env-name', type=str, default='coinrun')
    
    args = parser.parse_args()
    return args
    
def initialize_env2(num_envs, env_name, num_levels, start_level, distribution_mode): # prob change this, don't need all the extra wrappers for evaluating
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


def main(args):
    # set a seed for torch so that theres less variance in the results
    torch.manual_seed(0)
    
    weights = torch.load(args.pretrained_weights_path + '.pth', map_location='cpu')
    device = torch.device('cuda:' + str(args.device))
    
    env = initialize_env2( # don't need to specify num_levels or start_level since we're doing it manually
        num_envs=args.num_envs, 
        env_name=args.env_name, 
        num_levels=0, 
        start_level=0, 
        distribution_mode=args.distribution_mode
    )
    
    level_returns = {}
    level_counter = {}
    
    init_all_level_dict = init_specific_all_levels_dict([i for i in range(500, 10000)], args.env_name)
    model = SB_Impala(env.observation_space['rgb'], env.action_space.n, 5e-4)
    model.load_state_dict(weights['model'])
    model = model.to(device)
    model.eval()
    
    ood_levels = list(reversed([i for i in range(500, 10000)]))
    state_bytes = env.venv.env.env.env.env.get_state()
    # env.venv.venv.env.env.env.env.set_state(current_state_bytes)
    
    for i in range(args.num_envs):
        level = ood_levels.pop()
        state_bytes[i] = init_all_level_dict[level][0]
        
    env.venv.env.env.env.env.set_state(state_bytes)
    state = env.reset()
    while ood_levels:
    # while len(level_returns) < 9500:
        with torch.no_grad():
            state = {'rgb': torch.tensor(state['rgb'], device=device).permute(0, 3, 1, 2)}
            action, _, _ = model(state)
            action = action.detach().cpu().numpy()
            next_state, reward, done, info = env.step(action)
            
            # get state bytes
            state_bytes = env.venv.env.env.env.env.get_state()
            
            for i, d in enumerate(done):
                if d:
                    try:
                        assert 'episode' in info[i].keys()
                        level = ood_levels.pop()
                        state_bytes[i] = init_all_level_dict[level][0]

                        curr_level = info[i]['prev_level_seed']
                        assert curr_level not in level_counter and curr_level not in level_returns
                        level_counter[curr_level] = 1
                        level_returns[curr_level] = info[i]['episode']['r']

                        print(f'Level: {curr_level}, Return: {info[i]["episode"]["r"]}')
                    except:
                        continue
                    
            env.venv.env.env.env.env.set_state(state_bytes)
            state = env.reset()
            
    with open(args.save_path + '.pkl', 'wb') as f:
        pickle.dump(level_returns, f)
        

if __name__ == '__main__':
    args = parse_args()
    main(args)