import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os

from src.lib.consts import BOUND_STRATEGIES

sns.set_style("whitegrid")
plt.rcParams['text.usetex'] = True  # Let TeX do the typsetting
plt.rcParams['font.family'] = 'serif'  # ... for regular text
plt.rcParams['font.sans-serif'] = ['Times']  # Choose a nice font here

parser = argparse.ArgumentParser()
parser.add_argument("--dis2_results_fname", type=str, required=True)
parser.add_argument("--plot_dir", type=str, default="plots/")
args = parser.parse_args()

data = pd.read_pickle(args.dis2_results_fname)
data = data[~data['train_method'].isin(['DANN', 'CDANN'])]
ns, nt = data['n_val_source'].to_numpy(), data['n_val_target'].to_numpy()
eps = np.sqrt((4 * nt + ns) * np.log(100) / (2 * nt * ns))
data['lower_bound'] = np.maximum(data['h_val_acc'] - data['max_ts_agree_diff'] - eps, 0)

fig, axs = plt.subplots(2, 3, figsize=(12, 8))
for i, strategy in enumerate(BOUND_STRATEGIES):
    if strategy in ['logits']:
        continue

    ax = axs.flatten()[i]
    k = int(strategy[3:])
    strat_data = data[data['bound_strategy'] == strategy]
    indices = strat_data.groupby(['dataset', 'shift', 'train_method'])['lower_bound'].idxmax()
    preds = strat_data.loc[indices][['lower_bound', 'trg_accuracy']].to_numpy()

    ax.scatter(preds[:, 1], preds[:, 0], s=16)
    title = 'Full Features' if i == 0 else f'Top 1/{k} PCs'
    ax.set_title(title, fontsize=23)
    ax.plot([0, 1], label='y = x', color='black', linestyle='--', linewidth=1.)
    ax.set_xlabel('Target Accuracy', fontsize=17)
    ax.set_ylabel('Prediction', fontsize=17)
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.set_xlim(0.2, 1)
    ax.set_ylim(-.01, 1)

plt.tight_layout()

if not os.path.isdir(args.plot_dir):
    os.mkdir(args.plot_dir)
name = os.path.join(args.plot_dir, 'dis2_reduce_PCs')
plt.savefig(f'{name}.pdf', format='pdf', transparent=True, bbox_inches='tight')
plt.savefig(f'{name}.png', format='png', transparent=False, bbox_inches='tight')