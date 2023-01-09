import yaml
from yaml.loader import SafeLoader
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

base_path = '.'

def get_path(x, y, z):
    return os.path.join(x, f'{y}_results_{z}.yml')

part = 'traf1'
out_base = 'baselines_traf'
plot_x = 'n'
plot_y = 'reached_goal'
legend_map = {
    'train': 'Train',
    'test_maps': 'Novel maps',
    'test_traffic': 'Novel traffic'
}


# part1 = 'maps1'
# part2 = 'traf1'
# for p in ['train', 'test_maps', 'test_traffic']:
#     with open(get_path(base_path, part1, p), 'r') as f:
#         y_src = yaml.load(f, SafeLoader)
#         with open(get_path(base_path, part2, p), 'r') as g:
#             y_trg = yaml.load(g, SafeLoader)
#             for k, v in y_src.items():
#                 y_trg[k] = [y_src[k][0]] + y_trg[k]
#             with open(get_path(base_path, part2, p+'_new'), 'w') as h:
#                 yaml.dump(y_trg, h)
#
# exit()

data = {}
datas = []

for p in ['train', 'test_maps', 'test_traffic']:
    with open(get_path(base_path, part, p)) as f:
        data_yaml = yaml.load(f, SafeLoader)
        datas.append(data_yaml)
        for k, v in data_yaml.items():
            data[k] = data[k] + v if k in data else v
        name = 'Partition'
        if name not in data:
            data[name] = []
        data[name] += [legend_map[p]] * len(data_yaml['n'])

df = pd.DataFrame(data)
# df['Partition'] = [legend_map[p]] * len(data_yaml['n'])
ax = sns.lineplot(df, x=plot_x, y=plot_y + '_mean', hue='Partition')
# ax.set(xscale='log')
ax.set(xlabel='# Traffic scenarios', ylabel='Success rate')
ax.set(xticks=data_yaml['n'])
ax.set(xticklabels=data_yaml['n'])
for i in range(len(datas)):
    d = datas[i]
    ax.fill_between(d[plot_x], y1=np.array(d[plot_y + '_mean']) - np.array(d[plot_y + '_std']), y2=np.array(d[plot_y + '_mean']) + np.array(d[plot_y + '_std']), alpha=.2)
ax.set_xlim([data_yaml['n'][0], data_yaml['n'][-1]])
ax.set_ylim([0, 1.005])

plt.tight_layout()
plt.savefig(f'{out_base}.png')
plt.savefig(f'{out_base}.svg')
#plt.show()
