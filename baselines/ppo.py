import argparse
import gym

import ray
from ray import tune, air
from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.air import RunConfig

from driver_dojo.core.config import Config
from driver_dojo.core.env import DriverDojoEnv
from driver_dojo.core.types import *
from callbacks import CustomCallback

parser = argparse.ArgumentParser()
parser.add_argument("--num-cpus", type=int, default=24)
parser.add_argument("--num-gpus", type=int, default=1)
parser.add_argument("--env-seed", type=int, default=0)
parser.add_argument("--env-seed-offset", type=int, default=0)
parser.add_argument("--task", type=str, default="1_1_1")
parser.add_argument("--no-traffic", action="store_true")
parser.add_argument("--timesteps", type=int, default=1000000000)
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--exp_path", type=str, default=None)
parser.add_argument('--num-test', type=int, default=1000)

if __name__ == '__main__':
    args = parser.parse_args()
    num_maps, num_traffic, num_tasks = [int(x) for x in args.task.split('_')]

    c = Config()
    c.actions.space = 'Discretized'
    c.vehicle.dynamics_model = 'KS'
    c.simulation.dt = 0.2
    c.simulation.max_time_steps = 500
    c.scenario.traffic_init = True
    c.scenario.traffic_init_spread = 30.0
    c.scenario.traffic_spawn = True
    c.scenario.traffic_spawn_period = 1.0
    c.scenario.behavior_dist = False
    c.scenario.ego_init = True
    c.scenario.seed_offset = args.env_seed_offset
    c.vehicle.v_max = 13.34
    c.scenario.name = 'Intersection'
    c.scenario.kwargs['crossing_style'] = 'Minor'
    c.scenario.tasks = ['L']
    c.scenario.num_maps = num_maps
    c.scenario.num_traffic = num_traffic
    c.scenario.num_tasks = num_tasks

    if args.no_traffic:
        c.scenario.traffic_spawn = False
        c.scenario.traffic_init = False

    def env_creator(x):
        env = DriverDojoEnv(_config=c)
        return env

    register_env('custom_env', env_creator)

    ray.init()
    config = PPOConfig()
    config = config.training(
        gamma=0.99,
        lambda_=0.95,
        train_batch_size=8192,
        sgd_minibatch_size=256,
        lr=0.00005,
        clip_param=0.2,
        num_sgd_iter=20,
        kl_coeff=0.2,
    )
    config = config.environment(
        env='custom_env',
        disable_env_checking=True
    )
    config = config.framework(
        framework='torch',
    )
    config = config.rollouts(
        num_rollout_workers=args.num_cpus - 1,
        num_envs_per_worker=1,
        rollout_fragment_length='auto',
        horizon=500,
    )
    config = config.resources(
        num_gpus=args.num_gpus,
    )
    config = config.callbacks(CustomCallback)
    config.model['framestack'] = True
    config.model['fcnet_hiddens'] = [512, 512]

    if not args.as_test:
        tuner = tune.Tuner(
            "PPO",
            run_config=RunConfig(
                name=f"PPO_custom_env_{num_maps}_{num_traffic}_{num_tasks}_{args.env_seed}_{args.env_seed_offset}",
                stop=dict(
                    timesteps_total=args.timesteps,
                ),
                verbose=1,
                checkpoint_config=air.CheckpointConfig(
                    checkpoint_frequency=1, checkpoint_at_end=True,
                ),
            ),
            param_space=config.to_dict(),
            tune_config=tune.TuneConfig(
                metric='custom_metrics/cost_mean',
                mode='min'
            )
        )
        results = tuner.fit()
    else:
        assert args.exp_path is not None
        tuner = tune.Tuner.restore(path=args.exp_path)
        results = tuner.get_results()

    checkpoint = results.get_best_result().checkpoint
    # Create new Algorithm and restore its state from the last checkpoint.
    algo = Algorithm.from_checkpoint(checkpoint)

    for phase in ['train', 'test']:
        # Create the env to do inference in.
        if phase == 'test':
            c.scenario.test_seeding = True
        env = env_creator(None)
        obs = env.reset()

        rewards = []
        infos = []
        episode_reward = 0
        for it in range(args.num_test):
            print("Episode", it)
            done = False
            while not done:
                a = algo.compute_single_action(
                    observation=obs,
                    explore=False,
                    policy_id="default_policy",
                )
                # Send the computed action `a` to the env.
                obs, reward, done, info = env.step(a)
                episode_reward += reward
                if done:
                    print(f"Episode done: Total reward = {episode_reward}")
                    obs = env.reset()
                    rewards.append(episode_reward)
                    infos.append(info)
                    episode_reward = 0.0

        result_dict = dict()
        import numpy as np
        for k, v in infos[0].items():
            x = []
            for info in infos:
                x.append(info[k])
            if k not in ['road_seed', 'traffic_seed']:
                result_dict[k] = float(np.mean(x))
            else:
                result_dict[k] = list(x)
        result_dict['reward'] = np.mean(rewards)

        print(phase)
        print(result_dict)
        import os
        import yaml
        with open(os.path.join(results.get_best_result().log_dir, f'results_{phase}.yaml'), 'w') as f:
            yaml.dump(result_dict, f)


    algo.stop()

    ray.shutdown()
