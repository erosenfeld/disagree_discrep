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
parser.add_argument("--other_results_fname", type=str, required=True)
parser.add_argument("--DA", action='store_true')
parser.add_argument("--plot_dir", type=str, default="plots/")
args = parser.parse_args()

dis2data = pd.read_pickle(args.dis2_results_fname)
dis2data = dis2data[np.logical_xor(args.DA, ~dis2data['train_method'].isin(['DANN', 'CDANN']))]

logit_dis2_data = dis2data[dis2data['bound_strategy'].isin(['logits'])]
ns, nt = logit_dis2_data['n_val_source'].to_numpy(), logit_dis2_data['n_val_target'].to_numpy()
eps = np.sqrt((4 * nt + ns) * np.log(100) / (2 * nt * ns))
logit_dis2_data['lower_bound'] = logit_dis2_data['h_val_acc'] - logit_dis2_data['max_ts_agree_diff'] - eps
logit_dis2_indices = logit_dis2_data.groupby(['dataset', 'shift', 'train_method'])['lower_bound'].idxmax()
logit_dis2_preds = logit_dis2_data.loc[logit_dis2_indices][['lower_bound', 'trg_accuracy']].to_numpy()

other_data = pd.read_pickle(args.other_results_fname)
other_data = other_data[np.logical_xor(args.DA, ~other_data['train_method'].isin(['DANN', 'CDANN']))]

ATC_preds = other_data[(other_data['prediction_method'] == 'ATC_NE') & (other_data['temperature'] == 'source')]
ATC_preds = ATC_preds[['lower_bound', 'trg_accuracy']].to_numpy()
COT_preds = other_data[(other_data['prediction_method'] == 'COT') & (other_data['temperature'] == 'source')]
COT_preds = COT_preds[['lower_bound', 'trg_accuracy']].to_numpy()
AC_preds = other_data[(other_data['prediction_method'] == 'AC') & (other_data['temperature'] == 'source')]
AC_preds = AC_preds[['lower_bound', 'trg_accuracy']].to_numpy()

plt.figure(figsize=(8, 4))
plt.scatter(logit_dis2_preds[:, 1], logit_dis2_preds[:, 0], s=25, label=r'\textsc{Dis}$^2$ (ours)', zorder=2, marker='*')
plt.scatter(ATC_preds[:, 1], ATC_preds[:, 0], s=16, label='ATC', zorder=1, alpha=0.4)
plt.scatter(COT_preds[:, 1], COT_preds[:, 0], s=16, label='COT', zorder=1, alpha=0.4)
plt.scatter(AC_preds[:, 1], AC_preds[:, 0], s=16, label='AC', zorder=1, alpha=0.4)
plt.xlabel('Target Accuracy', fontsize=19)
plt.ylabel('Target Accuracy Prediction', fontsize=19)
plt.xlim(0.2, 1)
plt.ylim(0, 1)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.plot([0, 1], label='y = x', color='black', linestyle='--', linewidth=1., zorder=13)
plt.legend(loc='upper left', fontsize=16, bbox_to_anchor=(1, 1))
plt.tight_layout()

if not os.path.isdir(args.plot_dir):
    os.mkdir(args.plot_dir)
name = os.path.join(args.plot_dir, 'compare_methods_acc_vs_pred' + ('_DA' if args.DA else ''))
plt.savefig(f'{name}.pdf', format='pdf', transparent=True, bbox_inches='tight')
plt.savefig(f'{name}.png', format='png', transparent=False, bbox_inches='tight')