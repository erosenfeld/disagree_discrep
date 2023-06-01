import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os

sns.set_style("whitegrid")
plt.rcParams['text.usetex'] = True  # Let TeX do the typsetting
plt.rcParams['font.family'] = 'serif'  # ... for regular text
plt.rcParams['font.sans-serif'] = ['Times']  # Choose a nice font here

parser = argparse.ArgumentParser()
parser.add_argument("--dis2_results_fname", type=str, required=True)
parser.add_argument("--plot_dir", type=str, default="plots/")
args = parser.parse_args()

dis2_data = pd.read_pickle(args.dis2_results_fname)
dis2_data = dis2_data[~dis2_data['train_method'].isin(['DANN', 'CDANN'])]
ns, nt = dis2_data['n_val_source'].to_numpy(),  dis2_data['n_val_target'].to_numpy()
eps = np.sqrt((4 * nt + ns) * np.log(100) / (2 * nt * ns))
dis2_data['lower_bound'] = np.maximum(dis2_data['h_val_acc'] - dis2_data['max_ts_agree_diff'] - eps, 0)

fig, axs = plt.subplots(1, 4, figsize=(16, 4))

min_ratios = [0.95, 0.9, 0.8]
for i, ax in enumerate(axs.flatten()):
    if i == 0:
        data = dis2_data[dis2_data['bound_strategy'] == 'logits']
        indices = data.groupby(['dataset', 'shift', 'train_method'])['lower_bound'].idxmax()
    else:
        data = dis2_data[(dis2_data['min_ratio'] > min_ratios[i-1]) | (dis2_data['bound_strategy'] == 'logits')]
        indices = data.groupby(['dataset', 'shift', 'train_method'])['lower_bound'].idxmax()

    preds = data.loc[indices][['lower_bound', 'trg_accuracy']].to_numpy()
    MAE = np.abs(preds[:, 0] - preds[:, 1]).mean()
    coverage = (preds[:, 0] < preds[:, 1]).mean()

    ax.scatter(preds[:, 1], preds[:, 0], s=16)
    ax.plot([0, 1], label='y = x', color='black', linestyle='--', linewidth=1.)
    ax.set_xlabel('Target Accuracy', fontsize=18)
    ax.set_ylabel('Accuracy Prediction', fontsize=18)
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.set_xlim(0.2, 1)
    ax.set_ylim(0, 1)
    title = 'Logits' if i == 0 else rf'Ratio$>${min_ratios[i-1]}'
    ax.set_title(f'{title} ({MAE:.2f}/{coverage:.2f})', fontsize=22)

plt.tight_layout()

if not os.path.isdir(args.plot_dir):
    os.mkdir(args.plot_dir)
name = os.path.join(args.plot_dir, 'dis2_decrease_min_ratio')
plt.savefig(f'{name}.pdf', format='pdf', transparent=True, bbox_inches='tight')
plt.savefig(f'{name}.png', format='png', transparent=False, bbox_inches='tight')