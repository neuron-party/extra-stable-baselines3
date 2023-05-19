import numpy as np
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms.adversarial.gail import GAIL
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env

import os
import numpy as np
import argparse
import procgen
import torch
import copy

from stable_baselines3.common.vec_env import VecExtractDictObs, VecMonitor
from stable_baselines3.ppo.ppo import PPO

from models.impala import *
from gym.wrappers import TimeLimit

import numpy as np
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper


def parse_args():
    parser = argparse.ArgumentParser()
    
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
    
    parser.add_argument('--log', type=bool, default=False)
    parser.add_argument('--logging-path', type=str, default=None)
    
    parser.add_argument('--num-envs', type=int, default=64)
    parser.add_argument('--env-name', type=str, default=None)
    parser.add_argument('--num-levels', type=int, default=500)
    parser.add_argument('--start-level', type=int, default=0)
    parser.add_argument('--distribution-mode', type=str, default='hard')
    
    parser.add_argument('--learner-weights-path', type=str, default=None)
    parser.add_argument('--expert-weights-path', type=str, default=None)
    parser.add_argument('--num-demonstrations', type=int, default=2500) # 5 per level for 500 training levels
    parser.add_argument('--n-batches', type=int, default=1220) # corresponds to ~ 20M training steps, 610 ~ 10M training steps
    parser.add_argument('--device', type=int, default=0)
    
    parser.add_argument('--gym-seed', type=int, default=0)
    parser.add_argument('--torch-seed', type=int, default=0)
    parser.add_argument('--rng-seed', type=int, default=0)
    
    parser.add_argument('--learner-save-path', type=str, default=None)
    
    args = parser.parse_args()
    return args
    
    
    
def main(args):
    rng = np.random.default_rng(args.rng_seed) # arg parse this
    env = procgen.ProcgenEnv(
        num_envs=args.num_envs, 
        env_name=args.env_name, 
        num_levels=args.num_levels, 
        start_level=args.start_level,
        distribution_mode=args.distribution_mode,
        rand_seed=args.gym_seed
    )

    device = torch.device('cuda:' + str(args.device))
    expert_model = SB_Impala_BC(env.observation_space['rgb'], 15, args.lr)
    env = VecExtractDictObs(env, "rgb")
    env.observation_space = env.observation_space
    
    agent = PPO(
        policy=expert_model,
        custom_policy=True,
        env=env,
        n_steps=256,
        batch_size=2048,
        n_epochs=3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.01,
        vf_coef=0.5,
        target_kl=0.01,
        normalize_advantage=True,
        max_grad_norm=0.5,
        tensorboard_log=args.logging_path,
        device=device,
        checkpoint_path=None,
        checkpoints_remaining=None
    )
    
    agent.observation_space = env.observation_space
    agent.action_space = env.action_space
    agent.policy.device = device
    
    expert_weights = torch.load(args.expert_weights_path + '.pth')
    agent.policy.load_state_dict(expert_weights['model'])
    
    rollouts = rollout.rollout(
        agent,
        env,
        rollout.make_sample_until(min_timesteps=None, min_episodes=args.num_demonstrations),
        rng=rng,
        unwrap=False
    )
    
    transitions = rollout.flatten_trajectories(rollouts)
    
    learner_model = SB_Impala_BC(env.observation_space, 15, args.lr)
    
    if args.learner_weights_path: # otherwise from scratch
        learner_weights = torch.load(args.learner_weights_path + '.pth')
        learner_model.load_state_dict(learner_weights['model'])
    
    learner_model.observation_space = env.observation_space
    learner_model.action_space = env.action_space
    learner_model.device = device
    
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        rng=rng,
        batch_size=64 * 256,
        minibatch_size=2048,
        policy=learner_model,
        device=device
        # custom_logger=args.logging_path
    )
    
    bc_trainer.train(n_batches=args.n_batches)

    torch.save({'model': bc_trainer.policy.state_dict()}, args.learner_save_path + '.pth')

if __name__ == '__main__':
    args = parse_args()
    main(args)