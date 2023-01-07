from ray.rllib.algorithms.ppo import PPOConfig
from callbacks import CustomCallback


def get_config(args):
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

    return config
