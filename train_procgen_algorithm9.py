import os
import copy
import numpy as np
import gym
import argparse
import procgen
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3.ppo.ppo_research_9 import PPO_ResearchMethod9
from stable_baselines3.common.vec_env import VecExtractDictObs, VecMonitor

from networks import *
from method_utils import *
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
    parser.add_argument('--save-path', type=str, default='agent.pth')
    parser.add_argument('--csv-path', type=str, default=None)
    parser.add_argument('--checkpoint-path', type=str, default=None)
    
    parser.add_argument('--hard-levels-path', type=str, default=None)
    parser.add_argument('--load-existing-alt', type=str, default=None) # alt: all-level-trajectories
    parser.add_argument('--load-unplayable-levels', type=str, default=None)
    parser.add_argument('--finetuned-weights-path', type=str, default=None)
    parser.add_argument('--pretrained-weights-path', type=str, default=None)
    
    parser.add_argument('--max-steps-multiplier', type=int, default=2)
    parser.add_argument('--p', type=float, default=0.5)
    
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
    
    # all OOD tests have been on levels 500-10000 so far, so i'll keep the testing evaluation consistent with this too
    test_env = procgen.ProcgenEnv(
        num_envs=1, 
        env_name=args.env_name, 
        num_levels=9500, 
        start_level=500,
        distribution_mode=args.distribution_mode
    )
    test_env = gym.wrappers.RecordEpisodeStatistics(test_env)
    test_env = gym.wrappers.NormalizeReward(test_env, gamma=args.gamma)
    test_env = gym.wrappers.TransformReward(test_env, lambda reward: np.clip(reward, -10, 10))
    test_env.is_vector_env = True
    test_env = VecMonitor(test_env)
    
    weights = torch.load(args.pretrained_weights_path + '.pth')
    model.load_state_dict(weights['model'])
    
    with open(args.hard_levels_path + '.pkl', 'rb') as f:
        hard_levels = pickle.load(f)
    
    if args.load_existing_alt:
        with open(args.load_existing_alt + '.pkl', 'rb') as f:
            all_level_trajectories = pickle.load(f)
    else:
        all_level_trajectories = init_all_level_trajectories() # implement later, just always load for now to save time
        
    if args.load_unplayable_levels:
        with open(args.load_unplayable_levels + '.pkl', 'rb') as f:
            unplayable_levels = pickle.load(f)
    
    # there should be no overlap in hard_levels and unplayable_levels
    intersection = [value for value in hard_levels if value in unplayable_levels]
    assert len(intersection) == 0
    
    init_research_method = {i : j for i, j in all_level_trajectories.items() if i in hard_levels}
    init_all_level_dict = init_all_levels_dict(args.num_levels, args.env_name)
    init_easy_level_dict = {i : j for i, j in init_all_level_dict.items() if i not in hard_levels}
    # all_level_trajectories, unplayable_levels
    level_average_returns = {}
    for i in range(args.num_levels):
        if i in hard_levels:
            level_average_returns[i] = 0.0
        else:
            level_average_returns[i] = 10.0
            
            
    # make self.env_max_steps dict 
    # noticed that some levels have max_steps of < 10, maybe we should set a higher floor? ask li-cheng
    env_max_steps = {}
    for i in all_level_trajectories:
        lengths = [len(all_level_trajectories[i][j]['trajectory bytes']) for j in all_level_trajectories[i].keys() if j != 'trajectory boundaries']
        max_length = max(lengths)
        max_steps = args.max_steps_multiplier * max_length
        env_max_steps[i] = max_steps

    for i in unplayable_levels:
        env_max_steps[i] = 1000
    

    agent = PPO_ResearchMethod9(
        policy=model,
        custom_policy=True,
        env=env,
        csv_path=args.csv_path,
        p=args.p,
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
    
    # add methods manually cuz faster
    agent.init_research_method = init_research_method.copy()
    agent.init_easy_level_dict = init_easy_level_dict.copy()
    agent.init_all_level_trajectories = all_level_trajectories.copy()
    agent.init_all_level_dict = init_all_level_dict.copy()
    agent.level_average_returns = level_average_returns.copy()
    agent.unplayable_levels = unplayable_levels.copy()
    agent.num_initial_hard_levels = len(hard_levels)
    agent.env_max_steps = env_max_steps
    
    # testing env / evaluation methods
    agent.test_env = test_env
    agent.test_logging_checkpoint = 30
    agent.num_test_trajectories = 20
    
    agent.learn(total_timesteps=args.max_global_steps)
    
    torch.save({
        'model': agent.policy.state_dict(),
        'optimizer': agent.policy.optimizer.state_dict()
    }, args.save_path + '_final.pth')
    

if __name__ == '__main__':
    args = parse_args()
    main(args)