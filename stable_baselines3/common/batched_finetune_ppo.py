# different training loops for research

import sys
import time
import copy
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
import pickle

import numpy as np
import torch as th # WTF 
import torch
import torch.nn.functional as F
from gym import spaces

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv

import random
import pandas as pd
import pickle

SelfBatchedFinetune = TypeVar("SelfBatchedFinetune", bound="BatchedFinetune")

# helper function
def maximum_length_index(l):
    lengths = [len(i) for i in l]
    maxlen = 0
    index = 0
    for a, b in enumerate(lengths):
        if b > maxlen:
            maxlen = b
            index = a
    return index


class BatchedFinetune(BaseAlgorithm):
    """
    PPO training loop for batched finetuning levels
    
    If any level ever has a successful trajectory, save it, even if the agent is not able to be finetuned correctly
    However, the trajectory may be extremely inefficient so I'm not sure if the agent will be able to learn with it using these kinds of demonstrations
    
    Needed methods:
    1. self.finetune_level_init_dict
    2. self.successful_trajectories
    """
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        custom_policy,
        finetune_level_list,
        trajectory_dict_path,
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        tensorboard_log: Optional[str] = None,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[Tuple[spaces.Space, ...]] = None,
    ):
        super().__init__(
            policy=policy,
            custom_policy=custom_policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            support_multi_env=True,
            seed=seed,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces,
        )

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer = None
        
        # using custom neural networks, not the given ones 
        self.custom_policy = custom_policy # bool
        self.policy = policy
        self.finetune_level_list = finetune_level_list
        self.trajectory_dict_path = trajectory_dict_path
        
        self.trajectory_buffer = [[] for i in range(self.n_envs)]
        self.trajectory_dict = {i : [] for i in self.finetune_level_list}
        self.finetune_level_play_num = {i : 0 for i in self.finetune_level_list}


        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        buffer_cls = DictRolloutBuffer if isinstance(self.observation_space, spaces.Dict) else RolloutBuffer

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        
        if not self.custom_policy: # wtf is this policy_class?????
            self.policy = self.policy_class(  # pytype:disable=not-instantiable
                self.observation_space,
                self.action_space,
                self.lr_schedule,
                use_sde=self.use_sde,
                **self.policy_kwargs  # pytype:disable=not-instantiable
            )
            
        self.policy = self.policy.to(self.device)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        if not self.custom_policy:
            self.policy.set_training_mode(False)
        else:
            self.policy.eval()

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
            
            # custom wandb logging 
            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs)
            
            # add all frames to trajectory buffer
            for i, state in enumerate(self._last_obs['rgb']):
                assert isinstance(state, np.ndarray)
                self.trajectory_buffer[i].append(state.copy())
            
            self._last_obs = new_obs
            self._last_episode_starts = dones
            
            for idx, item in enumerate(infos):
                if 'episode' in item.keys():
                    level = item['prev_level_seed']
                    
                    if level not in self.trajectory_dict.keys(): # because of initialization, its fine if we miss the first 10k steps or something XD
                        print('MISTAKE')
                        continue
                    
                    if item['episode']['r'] == 10:
                        if len(self.trajectory_dict[level]) >= 100:
                            replacement_index = maximum_length_index(self.trajectory_dict[level])
                            self.trajectory_dict[level][replacement_index] = self.trajectory_buffer[idx].copy()
                        else:
                            self.trajectory_dict[level].append(self.trajectory_buffer[idx].copy()) # list of np arrays
                            
                    self.trajectory_buffer[idx] = [] # reset list, end of a trajectory
                    self.finetune_level_play_num[level] += 1
                
            # init finetune level dict
            
            # set environments
            for idx, done in enumerate(dones):
                if done:
                    random_finetune_level = random.sample(self.finetune_level_init_dict.keys(), 1)[0]
                    state_bytes = self.finetune_level_init_dict[random_finetune_level]
                    current_state_bytes = env.venv.venv.env.env.env.env.get_state()
                    current_state_bytes[idx] = state_bytes[0] # need to access the list
                    env.venv.venv.env.env.env.env.set_state(current_state_bytes)
                    modified_starting_states = env.reset() # gonna throw warning message but its ok
                    self._last_obs = modified_starting_states.copy() # overwrite for sb3 compatability

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True
    

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def learn(
        self: SelfBatchedFinetune,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfBatchedFinetune:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())
        
        iters = 0
        
        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            # !!! TRAINING LOGGING !!!
            if log_interval is not None and iteration % log_interval == 0:
                time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)
                
                # pd.DataFrame(self.finetune_level_play_num, index=[0]).to_csv(self.trajectory_dict_path + '_playnum.csv')
                print(f'FINETUNE LEVEL PLAY NUM : {self.finetune_level_play_num}')
                
            # save trajectory dict path often
            with open(self.trajectory_dict_path + '.pkl', 'wb') as f:
                pickle.dump(self.trajectory_dict, f)
            
            self.train()
            
            iters += 1

        callback.on_training_end()

        return self

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, [] 