# different training loops for research

import sys
import time
import copy
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
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

SelfResearchMethod5 = TypeVar("SelfResearchMethod5", bound="ResearchMethod5")


class ResearchMethod5(BaseAlgorithm):
    """
    boundary sampling, revised training average tracker, (not implemented yet) early stopping, dynamic p value
    
    the irm dictionary value 'trajectory boundaries' is now a list of lists with overlapping boundaries
    
    ADD EARLY STOPPING IF THIS IS STILL BAD (last thing to implement and try)
    """
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        custom_policy,
        csv_path,
        env: Union[GymEnv, str],
        p,
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
        self.csv_path = csv_path
        self.p = p
        self.initial_p = p

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
            self._last_obs = new_obs
            self._last_episode_starts = dones
            
            
            # needed variables (can easily initialize these outside the class then just set them in)
            # self.init_research_method
            # self.level_num_runs
            # self.level_learned
            # self.init_easy_level_dict
            
            # update specific level returns
            for item in infos:
                if 'episode' in item.keys():
                    level = item['prev_level_seed']
                    
                    if level not in self.init_research_method:
                        print(f'Finished playing normal level: {level}')
                    else:
                        self.init_research_method[level]['training average reward'] = (self.init_research_method[level]['training average reward'] * 0.9) + (item['episode']['r'] * 0.1)
            
            # set environments
            for idx, done in enumerate(dones):
                if done:
                    if np.random.random() <= self.p: # basically the same exact code, but we are only augmenting training with probability p (some parameter to set later)
                        hard_level = random.sample(self.init_research_method.keys(), 1)[0]
                        average_return = self.init_research_method[hard_level]['training average reward']

                        if average_return >= 5.0:
                            if len(self.init_research_method[hard_level]['trajectory boundaries']) > 1: # > 0
                                self.init_research_method[hard_level]['trajectory boundaries'].pop()
                                self.init_research_method[hard_level]['training average reward'] = 0.0
                            else:
                                # maybe enforce a higher threshold for learning a level? check if average return >= 8.0 or 9.0 or something here?
                                self.level_learned[hard_level] = 1
                                self.init_easy_level_dict[hard_level] = self.init_all_level_dict[hard_level]
                                self.init_research_method.pop(hard_level)
                                self.p -= self.initial_p / self.num_initial_hard_levels
                                
                                print(f'Learned a hard level. Probability sampling parameter p is now at {self.p}')
                                
                                # resample the hard level or else will get a key error
                                hard_level = random.sample(self.init_research_method.keys(), 1)[0]
                                
                        # better readability
                        sampling_list = list(self.init_research_method[hard_level].keys())
                        hard_level_trajectory = random.sample(sampling_list[1:len(sampling_list)-1], 1)[0]
                        
                        if len(self.init_research_method[hard_level]['trajectory boundaries'][-1]) == 1:
                            starting_boundary = 0
                            
                            # before (for jumper_rm5) it was [0] or [last boundary]; in the case of [last boundary] then sample from [last_boundary, len(trajectory)] 
                        
                        elif len(self.init_research_method[hard_level]['trajectory boundaries'][-1]) == 2:
                            starting_boundary = np.random.randint(
                                low=self.init_research_method[hard_level]['trajectory boundaries'][-1][0],
                                high=self.init_research_method[hard_level]['trajectory boundaries'][-1][1]
                            )
                        else:
                            raise ValueError('incorrect boundaries formatting')
                        
                        state_bytes = self.init_research_method[hard_level][hard_level_trajectory]['trajectory bytes'][starting_boundary]

                        current_state_bytes = env.venv.venv.env.env.env.env.get_state()
                        current_state_bytes[idx] = state_bytes[0] # need to access the list
                        env.venv.venv.env.env.env.env.set_state(current_state_bytes)

                        modified_starting_states = env.reset() # gonna throw warning message but its ok
                        self._last_obs = modified_starting_states.copy() # overwrite for sb3 compatability
                        
                    else: # for some reason it stops going back to play regular states so im just gonna manually enforce it now.
                        random_easy_level = random.sample(list(self.init_easy_level_dict.keys()), 1)[0] 
                        current_state_bytes = env.venv.venv.env.env.env.env.get_state()
                        current_state_bytes[idx] = self.init_easy_level_dict[random_easy_level][0]
                        env.venv.venv.env.env.env.env.set_state(current_state_bytes)
                        modified_starting_states = env.reset() 
                        self._last_obs = modified_starting_states.copy() 

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
        self: SelfResearchMethod5,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfResearchMethod5:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
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
                
                # do some extra logging here
                # pd.DataFrame({i : yuh[i]['training average reward'] for i in yuh}, index=[0])
                pd.DataFrame({i : self.init_research_method[i]['training average reward'] for i in self.init_research_method}, index=[0]).to_csv(self.csv_path + '.csv')
                pd.DataFrame(self.level_learned, index=[0]).to_csv(self.csv_path + '_level_learned.csv')
                
                
            self.train()

        callback.on_training_end()

        return self

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, [] 