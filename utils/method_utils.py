import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
import gym
import procgen
import tqdm
from stable_baselines3.common.vec_env import VecExtractDictObs, VecMonitor


def initialize_env(num_envs, env_name, num_levels, start_level, distribution_mode):
    env = procgen.ProcgenEnv(
        num_envs=num_envs,
        env_name=env_name,
        num_levels=1,
        start_level=start_level,
        distribution_mode=distribution_mode
    )
    return env


def initialize_full_env(num_envs, env_name, num_levels, start_level, distribution_mode):
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


def init_easy_levels_dict(hard_levels, num_levels, env_name):
    init_state_bytes = {}
    for level in range(num_levels):
        if level not in hard_levels:
            env = initialize_env(
                num_envs=1,
                env_name=env_name,
                num_levels=1,
                start_level=level,
                distribution_mode='hard'
            )
            _ = env.reset()
            b = env.env.get_state()
            init_state_bytes[level] = b
    return init_state_bytes

def init_all_levels_dict(num_levels, env_name):
    init_state_bytes = {}
    for level in range(num_levels):
        env = initialize_env(
            num_envs=1,
            env_name=env_name,
            num_levels=1,
            start_level=level,
            distribution_mode='hard'
        )
        _ = env.reset()
        b = env.env.get_state()
        init_state_bytes[level] = b
    return init_state_bytes


def init_research_method_1(finetune_weights_path, hard_levels, model, device):
    '''
    return a dictionary with key: hard level, value: dictionary
    where each inner dictionary has
        key: hard_level_trajectory_n
        values: average return during training, boundary sections to start from, byte info for setting the state
        
    finetune_weights_path should be formatted as path_ with matching levels
    i.e coinrun_1000_finetune_level_ 
    '''
    
    init = {i : {} for i in hard_levels}
    for hard_level in tqdm.tqdm(hard_levels):
        finetune_path = finetune_weights_path + str(hard_level) + '.pth'
        try:
            finetune_weights = torch.load(finetune_path)
        except:
            continue # i think i forgot to train 1 level
        model.load_state_dict(finetune_weights['model'])
        model = model.to(device)
        model.eval()
        env = initialize_env(
            num_envs=1,
            env_name='coinrun',
            num_levels=1,
            start_level=hard_level,
            distribution_mode='hard'
        )
        
        init[hard_level]['training average reward'] = 0.0
        
        with torch.no_grad():
            for i in range(10): # hard coded for now, this is the number of trajectories to generate for a level
                sum_reward = tries = 0
                while sum_reward != 10 and tries < 3: # ensure that the trajectory is a good one within 3 tries or else just give up on it (or else it gets stuck)
                    done = False
                    state = env.reset()
                    trajectory = [state]
                    trajectory_bytes = [env.env.get_state()]
                    sum_reward = 0
                    while not done:
                        state = {'rgb': torch.tensor(state['rgb'], device=device).permute(0, 3, 1, 2)}
                        action, _, _ = model(state)
                        action = action.detach().cpu().numpy()
                        next_state, reward, done, info = env.step(action)
                        state = next_state
                        sum_reward += reward
                        trajectory.append(state)
                        trajectory_bytes.append(env.env.get_state())

                    assert len(trajectory) == len(trajectory_bytes)
                    tries += 1
                    
                    init[hard_level][str(hard_level) + '_' + str(i)] = {
                        'trajectory bytes': trajectory_bytes,
                        'trajectory length': len(trajectory),
                        'trajectory boundaries': list(np.linspace(0, len(trajectory), 5).astype(np.int64)[:4]), # hardcoding 4 boundaries for now, but can change later
                        'finetuned agent reward': sum_reward # just as a informational metric, don't change
                    }
        
    return init





def init_research_method_2(finetune_weights_path, hard_levels, model, device):
    init = {i : {} for i in hard_levels}
    for hard_level in tqdm.tqdm(hard_levels):
        finetune_path = finetune_weights_path + str(hard_level) + '.pth'
        try:
            finetune_weights = torch.load(finetune_path)
        except:
            continue # i think i forgot to train 1 level
        model.load_state_dict(finetune_weights['model'])
        model = model.to(device)
        model.eval()
        env = initialize_env(
            num_envs=1,
            env_name='coinrun',
            num_levels=1,
            start_level=hard_level,
            distribution_mode='hard'
        )
        
        init[hard_level]['training average reward'] = 0.0
        
        low, high = 0, np.inf
        with torch.no_grad():
            for i in range(10): # hard coded for now, this is the number of trajectories to generate for a level
                sum_reward = tries = 0
                while sum_reward != 10 and tries < 3: # ensure that the trajectory is a good one within 3 tries or else just give up on it (or else it gets stuck)
                    done = False
                    state = env.reset()
                    trajectory = [state]
                    trajectory_bytes = [env.env.get_state()]
                    sum_reward = 0
                    while not done:
                        state = {'rgb': torch.tensor(state['rgb'], device=device).permute(0, 3, 1, 2)}
                        action, _, _ = model(state)
                        action = action.detach().cpu().numpy()
                        next_state, reward, done, info = env.step(action)
                        state = next_state
                        sum_reward += reward
                        trajectory.append(state)
                        trajectory_bytes.append(env.env.get_state())

                    assert len(trajectory) == len(trajectory_bytes)
                    tries += 1
                    
                    init[hard_level][str(hard_level) + '_' + str(i)] = {
                        'trajectory bytes': trajectory_bytes,
                        'finetuned agent reward': sum_reward # just as a informational metric, don't change
                    }
                    
                high = min(high, len(trajectory))
        boundaries = list(np.linspace(0, high, 5).astype(np.int64))[:4]
        init[hard_level]['trajectory boundaries'] = boundaries
        
    return init


def init_all_level_trajectories(
    easy_levels, 
    hard_levels, 
    pretrained_weights_path, 
    finetuned_weights_path, 
    env_name,
    model, 
    device,
    verbose=True
):
    '''
    keys: levels
    
    values: dictionary
        keys: trajectories level_1, level_2, ..., level_10
        values: dictionary
            key: trajectory state bytes
            value: list of state bytes for env.set_state()
            
            key: return:
            value: scalar; return of that run
            
        key: trajectory boundaries
        value: list of boundaries
        
    note to self: want to format this dictionary in a way such that we can directly transfer values to the hard level dict 
    i.e hard_level_dict[level] = all_level_dict[level]
    '''
    d = {i: {} for i in range(500)} # hard coded for now XD
    
    # easy level trajectories
    weights = torch.load(pretrained_weights_path + '.pth')
    model.load_state_dict(weights['model'])
    model = model.to(device)
    model.eval()
    for level in tqdm.tqdm(easy_levels):
        env = initialize_env(
            num_envs=1,
            env_name=env_name,
            num_levels=1,
            start_level=level,
            distribution_mode='hard'
        )
        low, high = 0, np.inf
        with torch.no_grad():
            for i in range(5): # number of trajectories to generate for a level
                sum_reward = tries = 0
                while sum_reward != 10 and tries < 3:
                    done = False
                    state = env.reset()
                    trajectory = [state]
                    trajectory_bytes = [env.env.get_state()]
                    sum_reward = 0
                    while not done:
                        state = {'rgb': torch.tensor(state['rgb'], device=device).permute(0, 3, 1, 2)}
                        action, _, _ = model(state)
                        action = action.detach().cpu().numpy()
                        next_state, reward, done, info = env.step(action)
                        state = next_state
                        sum_reward += reward
                        trajectory.append(state)
                        trajectory_bytes.append(env.env.get_state())

                    assert len(trajectory) == len(trajectory_bytes)
                    tries += 1
                    
                d[level][str(level) + '_' + str(i)] = {
                    'trajectory bytes': trajectory_bytes,
                    'return': sum_reward # just as a informational metric, don't change
                }
                high = min(high, len(trajectory))
                if verbose:
                    print(f'Level: {level}, Trajectory: {str(level) + "_" + str(i)}, Return: {sum_reward}')
                    
            boundaries = list(np.linspace(0, high, 5).astype(np.int64))[:4]
            d[level]['trajectory boundaries'] = boundaries
        
        
    # hard level trajectories
    for level in tqdm.tqdm(hard_levels):
        finetune_path = finetuned_weights_path + '_' + str(level) + '_finetune.pth'
        weights = torch.load(finetune_path)
        model.load_state_dict(weights['model'])
        model = model.to(device)
        model.eval()
        env = initialize_env(
            num_envs=1,
            env_name=env_name,
            num_levels=1,
            start_level=level,
            distribution_mode='hard'
        )
        low, high = 0, np.inf
        with torch.no_grad():
            for i in range(5): # number of trajectories to generate for a level
                sum_reward = tries = 0
                while sum_reward != 10 and tries < 3:
                    done = False
                    state = env.reset()
                    trajectory = [state]
                    trajectory_bytes = [env.env.get_state()]
                    sum_reward = 0
                    while not done:
                        state = {'rgb': torch.tensor(state['rgb'], device=device).permute(0, 3, 1, 2)}
                        action, _, _ = model(state)
                        action = action.detach().cpu().numpy()
                        next_state, reward, done, info = env.step(action)
                        state = next_state
                        sum_reward += reward
                        trajectory.append(state)
                        trajectory_bytes.append(env.env.get_state())

                    assert len(trajectory) == len(trajectory_bytes)
                    tries += 1
                    
                d[level][str(level) + '_' + str(i)] = {
                    'trajectory bytes': trajectory_bytes,
                    'return': sum_reward # just as a informational metric, don't change
                }
                high = min(high, len(trajectory))    
                if verbose:
                    print(f'Level: {level}, Trajectory: {str(level) + "_" + str(i)}, Return: {sum_reward}')
                        
            boundaries = list(np.linspace(0, high, 5).astype(np.int64))[:4]
            d[level]['trajectory boundaries'] = boundaries
            
    return d


def add_boundary_sampling(d):
    '''
    turn [0, b1, b2, b3, b4] -> [[0], [0, b1], [b1, b2], ...]
    '''
    d = d.copy()
    
    for level in d:
        boundaries = d[level]['trajectory boundaries']
        new_boundaries = []
        for i in range(len(boundaries) - 1):
            new_boundaries.append([boundaries[i], boundaries[i + 1]])
        new_boundaries.insert(0, [0])
        new_boundaries.append([boundaries[-1]])
        
        d[level]['trajectory boundaries'] = new_boundaries
    
    return d


def postprocess_dict(d):
    d = d.copy()
    
    unplayable_levels = []
    
    for level in d:
        trajectory_returns = [d[level][i]['return'] for i in d[level].keys() if i != 'trajectory boundaries']
        if sum(np.array(trajectory_returns) == 10) == 0:
            unplayable_levels.append(level)
        else:
            trajectory_popkeys = []
            for i in d[level]:
                if i != 'trajectory boundaries':
                    if d[level][i]['return'] == 0:
                        trajectory_popkeys.append(i)
            for tpk in trajectory_popkeys:
                d[level].pop(tpk)
    
    return d, unplayable_levels