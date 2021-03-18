import math
import random

from copy import deepcopy

import numpy as np

import gym

from rejoin_rta.utils.util import draw_from_rand_bounds_dict
from rejoin_rta.environments.managers import RewardManager, StatusManager, ObservationManager


class BaseEnv(gym.Env):
    def __init__(self, config):
        # save config
        self.config = config

        if 'verbose' in config:
            self.verbose = config['verbose']
        else:
            self.verbose = False

        self.observation_manager = ObservationManager(self.config["observation"])
        self.reward_manager = RewardManager(config=self.config["reward"])
        self.status_manager = StatusManager(config=self.config["status"])

        self._setup_env_objs()
        self._setup_action_space()
        self._setup_obs_space()

        self.timestep = 1  # TODO

        self.reset()

    def seed(self, seed=None):
        np.random.seed(seed)
        # note that python random should not be used (use numpy random instead)
        # Setting seed just to be safe in case it is accidentally used
        random.seed(seed)

        return [seed]
    
    def step(self, action):
        self._step_sim(action)

        self.status_dict = self._generate_constraint_status()

        reward = self._generate_reward()
        obs = self._generate_obs()
        info = self._generate_info()

        # determine if done
        if self.status_dict['success'] or self.status_dict['failure']:
            done = True
        else:
            done = False

        return obs, reward, done, info

    def _step_sim(self, action):
        raise NotImplementedError

    def reset(self):
        # apply random initilization to environment objects
        init_dict = self.config['init']

        successful_init = False
        while not successful_init:
            init_dict_draw = draw_from_rand_bounds_dict(init_dict)
            for obj_key, obj_init_dict in init_dict_draw.items():
                self.env_objs[obj_key].reset(**obj_init_dict)

            # TODO check if initialization is safe
            successful_init = True

        # reset processor objects
        self.reward_manager.reset(env_objs=self.env_objs)
        self.observation_manager.reset(env_objs=self.env_objs)
        self.status_manager.reset(env_objs=self.env_objs)

        # reset status dict
        self.status_dict = self.status_manager.status

        # generate reset state observations
        obs = self._generate_obs()

        if self.verbose:
            print("env reset with params {}".format(self._generate_info()))

        return obs

    def _setup_env_objs(self):
        self.env_objs = {}
        self.agent = None
        raise NotImplementedError

    def _setup_obs_space(self):
        self.observation_space = self.observation_manager.observation_space

    def _setup_action_space(self):
        self.action_space = self.agent.action_space
        
    def _generate_obs(self):
        # TODO: Handle multiple observations
        self.observation_manager.step(
            env_objs=self.env_objs,
            timestep=self.timestep,
            status=deepcopy(self.status_dict),
            old_status=deepcopy(self.status_dict)
        )
        return self.observation_manager.obs

    def _generate_reward(self):
        self.reward_manager.step(
            env_objs=self.env_objs,
            timestep=self.timestep,
            status=deepcopy(self.status_dict),
            old_status=deepcopy(self.status_dict)
        )
        return self.reward_manager.step_value

    def _generate_constraint_status(self):
        self.status_manager.step(
            env_objs=self.env_objs,
            timestep=self.timestep,
            status=deepcopy(self.status_dict),
            old_status=deepcopy(self.status_dict)
        )
        return self.status_manager.status

    def _generate_info(self):
        info = {
            'failure': self.status_dict['failure'],
            'success': self.status_dict['success'],
            'status': self.status_dict,
            'reward': self.reward_manager._generate_info(),
        }

        return info
