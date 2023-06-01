import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
import argparse
import os

sns.set_style("whitegrid")
plt.rcParams['text.usetex'] = True  # Let TeX do the typsetting
plt.rcParams['font.family'] = 'serif'  # ... for regular text
plt.rcParams['font.sans-serif'] = ['Times']  # Choose a nice font here

parser = argparse.ArgumentParser()
parser.add_argument("--dis2_results_fname", type=str, required=True)
parser.add_argument("--dis2_dbatloss_results_fname", type=str, required=True)
parser.add_argument("--dis2_negxentloss_results_fname", type=str)
parser.add_argument("--plot_dir", type=str, default="plots/")
args = parser.parse_args()

loss_fnames = {
    'D-BAT': args.dis2_dbatloss_results_fname,
    r'$\ell_{\textrm{dis}}$ (ours)': args.dis2_results_fname,
    'Negative Cross-Entropy': args.dis2_negxentloss_results_fname,
}

vals1, vals2 = pd.read_pickle(args.dis2_results_fname), pd.read_pickle(args.dis2_dbatloss_results_fname)
print(ttest_rel(
    vals1[vals1['bound_strategy'] == 'PCA1']['max_ts_agree_diff'],
    vals2[vals2['bound_strategy'] == 'PCA1']['max_ts_agree_diff'], alternative='greater'
))

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 4))
for loss, fname in loss_fnames.items():
    if fname is None:
        continue
    data = pd.read_pickle(fname)
    data = data[data['bound_strategy'] == 'PCA1']
    train_stderr = data["max_tr_agree_diff"].std() / np.sqrt(len(data))
    test_stderr = data["max_ts_agree_diff"].std() / np.sqrt(len(data))
    print(f'{loss}: train avg = {data["max_tr_agree_diff"].mean():.4f} (+- {train_stderr:.4f}),\t'
          f'test avg = {data["max_ts_agree_diff"].mean():.4f} (+- {test_stderr:.4f})')

    ax1.hist(data['max_tr_agree_diff'], bins=10, alpha=0.5, label=loss)
    ax1.set_xlabel('Disagreement Discrepancy', fontsize=17)
    ax1.set_ylabel('Count', fontsize=17)
    ax1.set_title('Train', fontsize=20)
    ax1.xaxis.set_tick_params(labelsize=14)
    ax1.yaxis.set_tick_params(labelsize=14)

    ax2.hist(data['max_ts_agree_diff'], bins=10, alpha=0.5, label=loss)
    ax2.set_xlabel('Disagreement Discrepancy', fontsize=17)
    ax2.set_title('Test', fontsize=20)
    ax2.xaxis.set_tick_params(labelsize=14)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc='upper right', fontsize=16)
plt.tight_layout()

if not os.path.isdir(args.plot_dir):
    os.mkdir(args.plot_dir)
name = os.path.join(args.plot_dir, 'compare_disagree_losses')
plt.savefig(f'{name}.pdf', format='pdf', transparent=True, bbox_inches='tight')
plt.savefig(f'{name}.png', format='png', transparent=False, bbox_inches='tight')