import argparse
import logging.config
import os

import numpy as np
import ray
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter

from core.test import test
from core.train import train
from core.utils import init_logger, make_results_dir

if __name__ == '__main__':
    # Lets gather arguments
    parser = argparse.ArgumentParser(description='MuZero Pytorch Implementation')
    parser.add_argument('--env', required=True, help='Name of the environment')
    parser.add_argument('--result_dir', default=os.path.join(os.getcwd(), 'results'),
                        help="Directory Path to store results (default: %(default)s)")
    parser.add_argument('--case', required=True, choices=['atari', 'classic_control', 'box2d'],
                        help="It's used for switching between different domains(default: %(default)s)")
    parser.add_argument('--opr', required=True, choices=['train', 'test'])
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='no cuda usage (default: %(default)s)')
    parser.add_argument('--no_mps', action='store_true', default=False,
                        help='no mps (Metal Performance Shaders) usage (default: %(default)s)')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='If enabled, logs additional values  '
                             '(gradients, target value, reward distribution, etc.) (default: %(default)s)')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Renders the environment (default: %(default)s)')
    parser.add_argument('--force', action='store_true', default=False,
                        help='Overrides past results (default: %(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='seed (default: %(default)s)')
    parser.add_argument('--value_loss_coeff', type=float, default=None,
                        help='scale for value loss (default: %(default)s)')
    parser.add_argument('--revisit_policy_search_rate', type=float, default=None,
                        help='Rate at which target policy is re-estimated (default: %(default)s)')
    parser.add_argument('--use_max_priority', action='store_true', default=False,
                        help='Forces max priority assignment for new incoming data in replay buffer '
                             '(only valid if --use_priority is enabled) (default: %(default)s)')
    parser.add_argument('--use_priority', action='store_true', default=False,
                        help='Uses priority for data sampling in replay buffer. '
                             'Also, priority for new data is calculated based on loss (default: False)')
    parser.add_argument('--use_target_model', action='store_true', default=False,
                        help='Use target model for bootstrap value estimation (default: %(default)s)')
    parser.add_argument('--num_actors', type=int, default=32,
                        help='Number of actors running concurrently (default: %(default)s)')
    parser.add_argument('--test_episodes', type=int, default=10,
                        help='Evaluation episode count (default: %(default)s)')
    parser.add_argument('--use_wandb', action='store_true', default=False,
                        help='logs console and tensorboard date on wandb(https://wandb.ai) (default: %(default)s)')
    parser.add_argument("--env-seed", type=int, default=0)  # For DriverDojoEnv
    parser.add_argument("--env-seed-offset", type=int, default=0)
    parser.add_argument("--task", type=str, default="1_1_1")
    parser.add_argument("--no-traffic", action="store_true")

    # Process arguments
    args = parser.parse_args()
    args.device = 'cuda' if (not args.no_cuda) and torch.cuda.is_available() else (
        'mps' if (not args.no_mps) and torch.backends.mps.is_available() else 'cpu')
    assert args.revisit_policy_search_rate is None or 0 <= args.revisit_policy_search_rate <= 1, \
        ' Revisit policy search rate should be in [0,1]'

    # seeding random iterators
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # import corresponding configuration , neural networks and envs
    if args.case == 'atari':
        # from config.atari import muzero_config
        raise NotImplementedError('Atari Configuration is not implemented and one can use classic_control'
                                  ' configuration as a template for the same.')
    elif args.case == 'box2d':
        from config.classic_control import muzero_config  # just using same config as classic_control for now
    elif args.case == 'classic_control':
        from config.classic_control import muzero_config
    else:
        raise Exception('Invalid --case option')

    # set config as per arguments
    exp_path = muzero_config.set_config(args)
    exp_path, log_base_path = make_results_dir(exp_path, args)

    # set-up logger
    init_logger(log_base_path)

    try:
        if args.opr == 'train':
            ray.init()
            if args.use_wandb:
                wandb.init(project="muzero-pytorch", sync_tensorboard=True,  config=muzero_config.get_hparams())
            summary_writer = SummaryWriter(exp_path, flush_secs=10)
            train(muzero_config, summary_writer)
            ray.shutdown()
        elif args.opr == 'test':
            assert os.path.exists(muzero_config.model_path), 'model not found at {}'.format(muzero_config.model_path)
            model = muzero_config.get_uniform_network().to('cpu')
            model.load_state_dict(torch.load(muzero_config.model_path, map_location=torch.device('cpu')))
            test_score = test(muzero_config, model, args.test_episodes, device='cpu', render=args.render,
                              save_video=True)
            logging.getLogger('test').info('Test Score: {}'.format(test_score))
        else:
            raise Exception('Please select a valid operation(--opr) to be performed')
    except Exception as e:
        logging.getLogger('root').error(e, exc_info=True)
