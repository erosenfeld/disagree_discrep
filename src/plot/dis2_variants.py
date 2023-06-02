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

data = pd.read_pickle(args.dis2_results_fname)
data = data[~data['train_method'].isin(['DANN', 'CDANN'])]
ns, nt = data['n_val_source'].to_numpy(), data['n_val_target'].to_numpy()
eps = np.sqrt((4 * nt + ns) * np.log(100) / (2 * nt * ns))
data['lower_bound'] = np.maximum(data['h_val_acc'] - data['max_ts_agree_diff'] - eps, 0)

orig_data = data[data['bound_strategy'] == 'PCA1']
orig_indices = orig_data.groupby(['dataset', 'shift', 'train_method'])['lower_bound'].idxmax()
orig_preds = orig_data.loc[orig_indices][['lower_bound', 'trg_accuracy']].to_numpy()

logit_data = data[data['bound_strategy'] == 'logits']
logit_indices = logit_data.groupby(['dataset', 'shift', 'train_method'])['lower_bound'].idxmax()
logit_preds = logit_data.loc[logit_indices][['lower_bound', 'trg_accuracy']].to_numpy()

logit_data['lower_bound2'] = logit_data['h_val_acc'] - logit_data['max_ts_agree_diff']
improved_indices = logit_data.groupby(['dataset', 'shift', 'train_method'])['lower_bound2'].idxmax()
improved_preds = logit_data.loc[improved_indices][['lower_bound2', 'trg_accuracy']].to_numpy()

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
all_preds = [orig_preds, logit_preds, improved_preds]
names = ['Features', 'Logits', 'Logits w/o $\delta$']

for i, ax in enumerate(axs.flatten()):
    preds = all_preds[i]
    ax.scatter(preds[:, 1], preds[:, 0], s=16)
    ax.set_title(names[i], fontsize=22)
    ax.plot([0, 1], label='y = x', color='black', linestyle='--', linewidth=1.)
    ax.set_xlabel('Target Accuracy', fontsize=18)
    ax.set_ylabel('Prediction', fontsize=18)
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.set_xlim(0.2, 1)
    ax.set_ylim(0, 1)

plt.tight_layout()

if not os.path.isdir(args.plot_dir):
    os.mkdir(args.plot_dir)
name = os.path.join(args.plot_dir, 'dis2_variants')
plt.savefig(f'{name}.pdf', format='pdf', transparent=True, bbox_inches='tight')
plt.savefig(f'{name}.png', format='png', transparent=False, bbox_inches='tight')