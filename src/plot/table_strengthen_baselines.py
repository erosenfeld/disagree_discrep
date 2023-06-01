import numpy as np
import pandas as pd
import argparse

from src.lib.consts import DATASETS
from src.eval.other_methods import PREDICT_METHOD_FUNCS

DESIRED_COVERAGES = [.9, .95, .99]

parser = argparse.ArgumentParser()
parser.add_argument("--dis2_results_fname", type=str)
parser.add_argument("--other_results_fname", type=str)
args = parser.parse_args()

def add_for_desired_coverage(accuracies, bounds, coverage):
    differences = accuracies - bounds
    add_for_coverage = min(np.quantile(differences, 1. - coverage, method='lower'), 0)
    assert (differences - add_for_coverage >= 0).mean() >= coverage
    return -add_for_coverage


def mult_for_desired_coverage(accuracies, bounds, coverage):
    ratios = accuracies / bounds
    mult_for_coverage = min(np.quantile(ratios - 1e-8, 1. - coverage, method='lower'), 1)
    assert (accuracies - mult_for_coverage * bounds >= 0).mean() >= coverage
    return mult_for_coverage


results_df = pd.read_pickle(args.other_results_fname)
results_df = results_df[~results_df['train_method'].isin(['DANN', 'CDANN'])]
dis2_df = pd.read_pickle(args.dis2_results_fname)
dis2_df = dis2_df[(dis2_df['bound_strategy'] == 'logits') & ~dis2_df['train_method'].isin(['DANN', 'CDANN'])]
adjusts = ['shift', 'scale']

for method in list(PREDICT_METHOD_FUNCS.keys()) + ['\\disdis (w/o $\delta$)']:
    if method == 'ATC':
        continue

    if 'disdis' in method:
        data = dis2_df
        data['lower_bound'] = dis2_df['h_val_acc'] - dis2_df['max_ts_agree_diff']
    else:
        data = results_df[(results_df['prediction_method'] == method) & (results_df['temperature'] == 'source')]

    diffs = data['trg_accuracy'] - data['lower_bound']
    valid_bounds = (diffs > 0)
    coverage = valid_bounds.mean()
    mean_err = diffs.abs().mean()
    invalid_results = diffs[~valid_bounds]
    print('\midrule')
    print(f'{method} & none && \multicolumn{{3}}{{c}}{{{mean_err:.3f}}} && \multicolumn{{3}}{{c}}{{{coverage:.3f}}} && '
          f'\multicolumn{{3}}{{c}}{{{-invalid_results.mean():.3f}}} \\\\')
    results = {adjust: {alpha: [] for alpha in DESIRED_COVERAGES} for adjust in adjusts}

    for alpha in DESIRED_COVERAGES:
        MAE_sum = {k: 0 for k in adjusts}
        coverage_sum = {k: 0 for k in adjusts}
        CAO_sum = {k: 0 for k in adjusts}
        CAO_cnts = {k: 0 for k in adjusts}
        cnt = 0
        for ds in DATASETS:
            tr_shifts = data[~(data['dataset'] == ds)]
            shift = add_for_desired_coverage(tr_shifts['trg_accuracy'], tr_shifts['lower_bound'], alpha)
            scale = mult_for_desired_coverage(tr_shifts['trg_accuracy'], tr_shifts['lower_bound'], alpha)

            ts_shifts = data[data['dataset'] == ds]
            cnt += len(ts_shifts)
            orig_bounds = ts_shifts['lower_bound'].to_numpy()
            adjusted_bounds = {
                'shift': orig_bounds - shift,
                'scale': orig_bounds * scale
            }
            for adjust in adjusts:
                diffs = ts_shifts['trg_accuracy'] - adjusted_bounds[adjust]
                valid_bounds = (diffs >= 0)
                MAE_sum[adjust] += diffs.abs().sum()
                coverage_sum[adjust] += valid_bounds.sum()
                CAO_sum[adjust] -= diffs[~valid_bounds].sum()
                CAO_cnts[adjust] += (~valid_bounds).sum()

        for adjust in adjusts:
            results[adjust][alpha] = [MAE_sum[adjust] / cnt,
                                      coverage_sum[adjust] / cnt,
                                      CAO_sum[adjust] / CAO_cnts[adjust]]
    for adjust in adjusts:
        print(f'& {adjust}')
        for i in range(3):
            print('&& ' + ' & '.join([f'{results[adjust][alpha][i]:.3f}' for alpha in DESIRED_COVERAGES]))
        print('\\\\')


ns, nt = dis2_df['n_val_source'].to_numpy(), dis2_df['n_val_target'].to_numpy()
print('\\midrule')
for neg_power in [2, 3]:
    eps = np.sqrt((4 * nt + ns) * np.log(10 ** neg_power) / (2 * nt * ns))
    diffs = dis2_df['trg_accuracy'] - (dis2_df['h_val_acc'] - dis2_df['max_ts_agree_diff'] - eps)
    print(f'\\disdis ($\delta=10^{{-{neg_power}}}$) & none && \\multicolumn{{3}}{{c}}{{{diffs.abs().mean():.3f}}} && '
          f'\\multicolumn{{3}}{{c}}{{{(diffs >= 0).mean():.3f}}} && '
          f'\\multicolumn{{3}}{{c}}{{{-np.nan_to_num(diffs[(diffs < 0)].mean()):.3f}}} \\\\')
print('\\bottomrule')
