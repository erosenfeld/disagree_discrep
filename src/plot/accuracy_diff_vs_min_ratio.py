import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dis2_results_fname", type=str, required=True)
parser.add_argument("--plot_dir", type=str, default="plots/")
args = parser.parse_args()

sns.set_style("whitegrid")
plt.rcParams['text.usetex'] = True  # Let TeX do the typsetting
plt.rcParams['font.family'] = 'serif'  # ... for regular text
plt.rcParams['font.sans-serif'] = ['Times']  # Choose a nice font here

valid_points = invalid_points = None
data = pd.read_pickle(args.dis2_results_fname)
data = data[~data['train_method'].isin(['DANN', 'CDANN'])]

ns, nt = data['n_val_source'].to_numpy(),  data['n_val_target'].to_numpy()
eps = np.sqrt((4 * nt + ns) * np.log(100) / (2 * nt * ns))
data['lower_bound'] = data['h_val_acc'] - data['max_ts_agree_diff'] - eps
acc_diffs = data['trg_accuracy'].to_numpy() - np.maximum(data['lower_bound'].to_numpy(), 0)
valid_inds = (acc_diffs >= 0)

score_name = 'min_ratio'
valid_points = np.stack([data[valid_inds][score_name], acc_diffs[valid_inds]], axis=1)
invalid_points = np.stack([data[~valid_inds][score_name], acc_diffs[~valid_inds]], axis=1)

plt.figure(figsize=(6, 4))

plt.scatter(valid_points[:, 0], valid_points[:, 1], s=5, label='Valid Bound')
plt.scatter(invalid_points[:, 0], invalid_points[:, 1], marker='x', s=5, label='Invalid Bound')
plt.xlabel(r'Cumulative $\ell_1$ Ratio', fontsize=16)
plt.ylabel('Accuracy Minus Bound', fontsize=16)
plt.xlim(0.3, 1.01)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.tight_layout()

if not os.path.isdir(args.plot_dir):
    os.mkdir(args.plot_dir)
name = os.path.join(args.plot_dir, 'accuracy_diff_vs_min_ratio')
plt.savefig(f'{name}.pdf', format='pdf', transparent=True, bbox_inches='tight')
plt.savefig(f'{name}.png', format='png', transparent=False, bbox_inches='tight')
