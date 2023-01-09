import yaml
import numpy as np
from yaml.loader import SafeLoader
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('base_path', type=str)
parser.add_argument('in_path', type=str)
parser.add_argument('out_path', type=str)
parser.add_argument('nums', type=str)

args = parser.parse_args()


def get_dir_name(x):
    # return f'PPO_custom_env_{x}_1_1_0_0'
    return f"res{x}"


result_dict = dict()
result_dict['n'] = []
for i, n in enumerate([x for x in args.num.split(',')]):
    for k, v in result_dict:
        if i > 0 and len(v) != i:
            result_dict[k].append(np.NAN)

    path = os.path.join(args.base_path, get_dir_name(n), args.in_path)
    if not os.path.isfile(path):
        print(f"Could not find {path}! Skipping...")
        continue
    result_dict['n'].append(n)
    with open(path, 'r') as f:
        res_yaml = yaml.load(f, Loader=SafeLoader)
        for k, v in res_yaml.items():
            if k not in result_dict:
                result_dict[k] = [v]
            else:
                result_dict[k].append(v)

if '.yaml' not in args.out_path:
    args.out_path += '.yaml'
with open(args.out_path, 'w') as f:
    yaml.dump(result_dict, f)
