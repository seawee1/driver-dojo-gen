import yaml
from yaml.loader import SafeLoader
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('base_path', type=str)
parser.add_argument('in_path', type=str)
parser.add_argument('out_path', type=str)
parser.add_argument('nums', type=list)

args = parser.parse_args()


def get_dir_name(x):
    # return f'PPO_custom_env_{x}_1_1_0_0'
    return f"res{x}"


result_dict = dict()
result_dict['n'] = []
for n in args.num:
    result_dict['n'].append(n)
    path = os.path.join(args.base_path, get_dir_name(n), args.in_path)
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