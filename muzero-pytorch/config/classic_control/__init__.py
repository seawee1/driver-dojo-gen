from typing import Callable

import gym
import torch

from core.config import BaseMuZeroConfig, DiscreteSupport
from .env_wrapper import ClassicControlWrapper
from .model import MuZeroNet


class ClassicControlConfig(BaseMuZeroConfig):
    def __init__(self):
        super(ClassicControlConfig, self).__init__(
            training_steps=1000000,
            test_interval=100,
            test_episodes=20,
            checkpoint_interval=20,
            max_moves=500,
            discount=0.997,
            dirichlet_alpha=0.25,
            num_simulations=50,
            batch_size=128,
            td_steps=5,
            lr_init=0.05,
            lr_decay_rate=0.01,
            lr_decay_steps=10000,
            window_size=1000,
            value_loss_coeff=1,
            value_support=DiscreteSupport(0, 80),  # Extracted from PPO runs
            reward_support=DiscreteSupport(-10, 10))
        self.args = None

    def visit_softmax_temperature_fn(self, num_moves, trained_steps):
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25

    def set_game(self, env_name, args, save_video=False, save_path=None, video_callable=None):
        self.env_name = env_name
        game = self.new_game(args)
        self.obs_shape = game.reset().shape[0]
        self.action_space_size = game.action_space_size

    def get_uniform_network(self):
        return MuZeroNet(self.obs_shape, self.action_space_size, self.reward_support.size, self.value_support.size,
                         self.inverse_value_transform, self.inverse_reward_transform)

    def new_game(self, args=None, save_video=False, save_path=None, episode_trigger: Callable[[int], bool] = None, uid=None):
        from driver_dojo.core.env import DriverDojoEnv  # MINE
        import os, sys
        sys.path.append(os.getcwd() + '/..')
        from env_config import get_env_config
        if self.args is None:
            self.args = args
        env_config = get_env_config(self.args)
        env_config.scenario.generation_threading = False  # Pickle issue due to threading
        env = DriverDojoEnv(_config=env_config)
        #env = gym.make(self.env_name, new_step_api=True)

        if save_video:
            assert save_path is not None, 'save_path cannot be None if saving video'
            from gym.wrappers import RecordVideo
            env = RecordVideo(env, video_folder=save_path, episode_trigger=episode_trigger,
                              name_prefix=f"rl-video-{uid}", new_step_api=True)
        return ClassicControlWrapper(env, discount=self.discount, k=4)

    def scalar_reward_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)

    def scalar_value_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)


muzero_config = ClassicControlConfig()
