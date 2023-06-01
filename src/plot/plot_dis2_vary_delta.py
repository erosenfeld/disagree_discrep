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


def get_frac_violated_for_delta(bounds, accs, ns, nt, delta):
    epsilon = np.sqrt((4 * nt + ns) * np.log(1. / delta) / (2 * nt * ns))
    return (bounds - epsilon > accs).mean()


data = pd.read_pickle(args.dis2_results_fname)
data = data[(~data['train_method'].isin(['DANN', 'CDANN'])) & (data['bound_strategy'] == 'logits')]
data['lower_bound'] = data['h_val_acc'] - data['max_ts_agree_diff']
indices = data.groupby(['dataset', 'shift', 'train_method'])['lower_bound'].idxmax()
preds = data.loc[indices][['lower_bound', 'trg_accuracy']].to_numpy()
ns, nt = data.loc[indices]['n_val_source'].to_numpy(), data.loc[indices]['n_val_target'].to_numpy()

fig = plt.figure(figsize=(4.5, 4))
deltas = [.2, .1, .05, .01, .005]
frac_violated = [get_frac_violated_for_delta(preds[:, 0], preds[:, 1], ns, nt, delta) for delta in deltas]

plt.plot(deltas, frac_violated, marker='^')
plt.plot([0, 1], linestyle='--', color='black', label='y = x', linewidth=1.)
plt.xlim(0, max(deltas)+.01)
plt.ylim(-.01, max(deltas)+.01)
plt.xlabel(r'Violation Probability $\delta$', fontsize=20)
plt.ylabel(r'Observed Violation Rate', fontsize=20)
plt.tick_params(labelsize=14)
plt.gca().invert_xaxis()
plt.legend(fontsize=20)
plt.tight_layout()

if not os.path.isdir(args.plot_dir):
    os.mkdir(args.plot_dir)
name = os.path.join(args.plot_dir, 'dis2_vary_delta')
plt.savefig(f'{name}.pdf', format='pdf', transparent=True, bbox_inches='tight')
plt.savefig(f'{name}.png', format='png', transparent=False, bbox_inches='tight')