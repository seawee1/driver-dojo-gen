import argparse
import gym

import ray
from ray import tune, air
from ray.rllib.algorithms import Algorithm
from ray.tune.registry import register_env
from ray.air import RunConfig

from driver_dojo.core.config import Config
from driver_dojo.core.env import DriverDojoEnv
from driver_dojo.core.types import *

from env_config import get_env_config

parser = argparse.ArgumentParser()
parser.add_argument("--algo", type=str, default="PPO")
parser.add_argument("--num-cpus", type=int, default=24)
parser.add_argument("--num-gpus", type=int, default=1)
parser.add_argument("--env-seed", type=int, default=0)
parser.add_argument("--env-seed-offset", type=int, default=0)
parser.add_argument("--task", type=str, default="1_1_1")
parser.add_argument("--no-traffic", action="store_true")
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--exp_path", type=str, default=None)
parser.add_argument('--num-test', type=int, default=1000)

if __name__ == '__main__':
    args = parser.parse_args()
    num_maps, num_traffic, num_tasks = [int(x) for x in args.task.split('_')]

    c = get_env_config(args)

    def env_creator(x):
        env = DriverDojoEnv(_config=c)
        return env

    register_env('custom_env', env_creator)

    ray.init()
    if args.algo == 'PPO':
        from baselines.ppo_config import get_config
        config = get_config(args)
    else:
        raise ValueError("Algo not implemented!")

    if args.as_test:
        assert args.exp_path is not None
        tuner = tune.Tuner.restore(path=args.exp_path)
        results = tuner.get_results()
    else:
        tuner = tune.Tuner(
            args.algo,
            run_config=RunConfig(
                name=f"{args.algo}_custom_env_{num_maps}_{num_traffic}_{num_tasks}_{args.env_seed}_{args.env_seed_offset}",
                stop=dict(
                    timesteps_total=1000000000,
                ),
                verbose=1,
                checkpoint_config=air.CheckpointConfig(
                    checkpoint_frequency=1,
                    checkpoint_at_end=True,
                    num_to_keep=10,
                ),
            ),
            param_space=config.to_dict(),
            tune_config=tune.TuneConfig(
                metric='custom_metrics/cost_mean',
                mode='min'
            )
        )
        results = tuner.fit()

    # Evaluate
    checkpoint = results.get_best_result().checkpoint
    algo = Algorithm.from_checkpoint(checkpoint)

    for phase in ['train', 'test']:
        # Create the env to do inference in.
        if phase == 'test':
            c.scenario.test_seeding = True
        else:
            c.scenario.test_seeding = False
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
                result_dict[k+'_std'] = float(np.std(x))
            else:
                result_dict[k] = [int(i) for i in x]
        result_dict['reward'] = float(np.mean(rewards))
        result_dict['reward_std'] = float(np.std(rewards))

        print(phase)
        print(result_dict)
        import os
        import yaml
        with open(os.path.join(results.get_best_result().log_dir, f'results_{phase}.yaml'), 'w') as f:
            yaml.dump(result_dict, f)

    algo.stop()
    ray.shutdown()
