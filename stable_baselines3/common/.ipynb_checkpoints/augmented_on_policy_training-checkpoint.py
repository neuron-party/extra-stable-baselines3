import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gym import spaces
import gym
import procgen
import random
import pandas as pd

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv

# tbh no idea what this does lol
SelfNaiveGeneralizationAugmentation = TypeVar("SelfNaiveGeneralizationAugmentation", bound="NaiveGeneralizationAugmentation")


def initialize_env(num_envs, env_name, num_levels, start_level, distribution_mode):
    env = procgen.ProcgenEnv(
        num_envs=num_envs,
        env_name=env_name,
        num_levels=1,
        start_level=start_level,
        distribution_mode=distribution_mode
    )
    return env



class NaiveGeneralizationAugmentation(BaseAlgorithm):
    """
    Modified version of OnPolicyAlgorithm from stable baselines 3 for researching different training methods
    
    Modified methods:
        collect_rollouts()
        
    Additional methods:
        self.init_level_dict
        self.level_average_returns
        self.level_num_runs
    """

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        custom_policy,
        env: Union[GymEnv, str],
        csv_path,
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
        
        # !!! START
        # level_init_dict, level_average_returns, and level_num_runs
        self.level_average_returns = {i:0 for i in range(env.venv.env.env.env.env.options['num_levels'])}
        self.level_num_runs = {i:0 for i in range(env.venv.env.env.env.env.options['num_levels'])}
        
        state_ids = []
        for i in range(env.venv.env.env.env.env.options['num_levels']):
            # num_envs, env_name, num_levels, start_level, distribution_mode
            dummy_env = initialize_env(
                num_envs=1,
                env_name=env.venv.env.env.env.env.options['env_name'],
                num_levels=1,
                start_level=i,
                distribution_mode='hard' # hardcoding this cuz too annoying to dynamic 
            )
            _ = dummy_env.reset()
            state_ids.append(dummy_env.env.get_state())
            
        self.level_init_dict = {i:state for i, state in zip(range(env.venv.env.env.env.env.options['num_levels']), state_ids)}
        self.csv_path = csv_path
        # !!! END
        
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
            # pretty sure I can remove this stuff, will never use this .use_sde method ever lol
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
            
            # i think setting the new obs here is incorrect for our method, since one of these envs may get replaced, the initial state will not be the same as 
            # the state of the augmented environment
            # we need to take advantage of the manual reset ignored env.reset() method here to extract the modified environment observations
            # the way that stable baselines 3 uses dones is as a 'beginning of episode signal' i.e 
                # self._last_episode_starts = True when a new trajectory starts; don't really understand this format but need to adhere to it when making modifications
            self._last_obs = new_obs
            self._last_episode_starts = dones
            
            # !!! start
            # on termination, with a 50% chance replace that environment with the lowest return environment
            # this implementation is not the exact same as the one i made in comp-ml
            # differences: 
                # instead of returning the same lowest level repeatedly (until it 'gets it out of the way'),
                # instead return a list of all levels that have the same lowest score and randomly sample among these
                # i.e if 50 different levels have 0 reward, randomly sample from these 50 each time instead of having it play
                # one of these repeatedly until the reward is no longer 0
                
                # added env.reset() for correct resetting of the initial state array
            for item in infos:
                if 'episode' in item.keys():
                    level = item['prev_level_seed']
                    
                    # if level not in self.level_average_returns:
                    #     self.level_average_returns[level] = item['episode']['r']
                    #     self.level_num_runs[level] = 1
                    # else:
                    self.level_num_runs[level] += 1
                    running_average = self.level_average_returns[level] - (self.level_average_returns[level] / self.level_num_runs[level])
                    running_average = running_average + (item['episode']['r'] / self.level_num_runs[level])
                    self.level_average_returns[level] = running_average
                        
            if dones.any():
                if np.random.random() <= 0.5:
                    current_state_ids = env.venv.venv.env.env.env.env.get_state()
                    for idx, d in enumerate(dones):
                        if d:
                            invert = {}
                            for k, v in self.level_average_returns.items():
                                if v in invert:
                                    invert[v].append(k)
                                else:
                                    invert[v] = [k]
                            lowest_return_level = sorted(invert.items(), key=lambda x: x[0])[0][1]
                            lowest_return_level = random.sample(lowest_return_level, 1)[0]
                            current_state_ids[idx] = self.level_init_dict[lowest_return_level][0]
                            env.venv.venv.env.env.env.env.set_state(current_state_ids)
                            
                            correct_reset_states = env.reset()
                            self._last_obs = correct_reset_states
                            
                            # dummy_states = env.reset()
                            # self._last_obs = dummy_states
                            # i think with no break, it sets every single finished state to the bad state where basically every single level is the bad state
                            break
                            
                    # only resets the env we care about, leaves the other ones alone
                    # don't need to change the dones or rewards signal since all we're doing is replacing an environment
                    # self._last_obs = env.reset() 
                    
            # import pdb; pdb.set_trace()
            # !!! end
            
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
        self: SelfNaiveGeneralizationAugmentation,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfNaiveGeneralizationAugmentation:
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
                
                # !!! START
                # additional csv logging cuz its cool
                pd.DataFrame(self.level_average_returns, index=[0]).to_csv(self.csv_path + '_level_average_returns.csv')
                pd.DataFrame(self.level_num_runs, index=[0]).to_csv(self.csv_path + '_level_num_runs.csv')
                print('working')
                # !!! END

            self.train()

        callback.on_training_end()

        return self

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []