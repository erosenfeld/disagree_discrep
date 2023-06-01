import torch
import numpy as np
import pandas as pd
import argparse
import os
import itertools

from src.lib.validation import validation_scores_and_bounds_from_split_data
from src.lib.consts import DATASET_SHIFTS, TRAIN_METHODS, BOUND_STRATEGIES
from src.lib.utils import load_shift_data, np_to_device

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--critic_repeats", type=int, default=30)
parser.add_argument("--val_frac", type=float, default=0.5)
parser.add_argument("--loss_type", type=str, default='disagreement')
parser.add_argument("--feats_dir", type=str, required=True)
parser.add_argument("--results_dir", type=str, required=True)
args = parser.parse_args()

np.random.seed(0)
torch.manual_seed(0)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('DEVICE:', device)
filename = os.path.join(
    args.results_dir,
    f'dis2_{args.epochs}epochs_{args.critic_repeats}repeats_valfrac{args.val_frac:0.2f}.pkl'
)
total_evals = 0
violations = {strategy: 0 for strategy in BOUND_STRATEGIES}
results_df = pd.DataFrame()
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

for dataset in DATASET_SHIFTS.keys():
    for shift, method in itertools.product(DATASET_SHIFTS[dataset], TRAIN_METHODS):

        try:
            (source_feats, source_labels), (target_feats, target_labels), \
                (h1_source_out, h1_target_out) = load_shift_data(args.feats_dir, dataset, shift, method, device)

            n_source, n_target = len(source_labels), len(target_labels)
            n_val_source, n_val_target = int(n_source * args.val_frac), int(n_target * args.val_frac)
            orig_src_acc = (h1_source_out.argmax(1) == source_labels).float().mean().item()
            orig_trg_acc = (h1_target_out.argmax(1) == target_labels).float().mean().item()

        except FileNotFoundError:
            print(f'Couldn\'t load {dataset} {method}_{shift}\n')
            continue

        df_entries = {}
        s_inds, s_val_inds = torch.arange(n_source)[:-n_val_source], torch.arange(n_source)[-n_val_source:]
        t_inds, t_val_inds = torch.arange(n_target)[:-n_val_target], torch.arange(n_target)[-n_val_target:]

        shfl_src_inds, shfl_trg_inds = torch.randperm(n_source), torch.randperm(n_target)
        shfl_source_feats, shfl_target_feats = source_feats[shfl_src_inds], target_feats[shfl_trg_inds]
        shfl_source_labels, shfl_target_labels = source_labels[shfl_src_inds], target_labels[shfl_trg_inds]
        shfl_source_logits, shfl_target_logits = h1_source_out[shfl_src_inds], h1_target_out[shfl_trg_inds]

        _, _, Vt = np.linalg.svd(shfl_source_feats[s_inds].cpu())
        PCA_components = np_to_device(Vt.T, device)

        for strategy in BOUND_STRATEGIES:
            if strategy == 'logits':
                eval_source_feats = shfl_source_logits
                eval_target_feats = shfl_target_logits
            else:
                dim_reduce = int(strategy[3:])
                k = shfl_source_feats.shape[1] // dim_reduce
                if k < 4:
                    continue
                eval_source_feats = torch.matmul(shfl_source_feats, PCA_components[:, :k])
                eval_target_feats = torch.matmul(shfl_target_feats,  PCA_components[:, :k])

            critic_stats, (h_val_acc, h_full_acc), epsilon =\
                validation_scores_and_bounds_from_split_data(
                    eval_source_feats[s_inds], eval_source_feats[s_val_inds],
                    shfl_source_logits[s_inds], shfl_source_logits[s_val_inds],
                    shfl_source_labels[s_inds], shfl_source_labels[s_val_inds],
                    eval_target_feats[t_inds], eval_target_feats[t_val_inds],
                    shfl_target_logits[t_inds], shfl_target_logits[t_val_inds],
                    shfl_target_labels[t_inds], shfl_target_labels[t_val_inds],
                    num_repeats=args.critic_repeats, num_epochs=args.epochs, loss_type=args.loss_type
                )

            best_critic_ind = critic_stats['ts_agree_diffs'].argmax() // args.epochs
            max_agree_diff = critic_stats['ts_agree_diffs'].max()
            lower_bound = max((h_val_acc - max_agree_diff - epsilon).item(), 0)
            print(f'dataset: {dataset}\tshift: {shift}\tmethod: {method}\tstrategy: {strategy}')
            print(f'dim: {shfl_source_feats.shape[1]}\t{n_source=}\t{n_target=}')
            print(f'ACCS\tsource: {orig_src_acc:.4f}\ttarget: {orig_trg_acc:.4f}\t'
                  f"lower bound: {lower_bound:.4f}")

            max_train_agree_diff = critic_stats['tr_agree_diffs'].max().item()
            max_test_agree_diff = critic_stats['ts_agree_diffs'].max().item()
            inds = torch.arange(args.critic_repeats)

            test_src_agrees = critic_stats['ts_src_agree']
            argmaxes = test_src_agrees.argmax(1)
            l1_diffs = torch.cat([test_src_agrees[:, 0:1], torch.diff(test_src_agrees, dim=1).abs()], dim=1)
            cum_diffs = torch.cumsum(l1_diffs, dim=1)
            ratios = test_src_agrees[inds, argmaxes] / cum_diffs[torch.arange(args.critic_repeats), argmaxes]

            test_trg_agrees = critic_stats['ts_trg_agree']
            trg_argmaxes = test_trg_agrees.argmax(1)
            trg_l1_diffs = torch.cat([test_trg_agrees[:, 0:1], torch.diff(test_trg_agrees, dim=1).abs()], dim=1)
            trg_cum_diffs = torch.cumsum(trg_l1_diffs, dim=1)
            trg_ratios = test_trg_agrees[inds, trg_argmaxes] / trg_cum_diffs[torch.arange(args.critic_repeats), trg_argmaxes]

            ratios[ratios != ratios] = 0
            trg_ratios[trg_ratios != trg_ratios] = 0

            print(f'src ratio\tmin:{ratios.min().item()}\tmax: {ratios.max().item()}\tmean: {ratios.mean().item()}')
            print(f'trg ratio\tmin:{trg_ratios.min().item()}\tmax: {trg_ratios.max().item()}\tmean: {trg_ratios.mean().item()}')

            entry = {
                'min_ratio': ratios.min().item(),
                'mean_ratio': ratios.mean().item(),
                'max_ratio': ratios.max().item(),
                'min_trg_ratio': trg_ratios.min().item(),
                'mean_trg_ratio': trg_ratios.mean().item(),
                'max_trg_ratio': trg_ratios.max().item(),
                'max_src_agree': test_src_agrees.max().item(),
                'max_trg_agree': test_trg_agrees.max().item(),
                'bound_valid': lower_bound <= orig_trg_acc,
                'src_accuracy': orig_src_acc,
                'trg_accuracy': orig_trg_acc,
                'lower_bound': lower_bound,
                'dataset': dataset,
                'shift': str(shift),
                'train_method': method,
                'bound_strategy': strategy,
                'max_tr_agree_diff': max_train_agree_diff,
                'max_ts_agree_diff': max_test_agree_diff,
                'n_source': n_source,
                'n_val_source': n_val_source,
                'n_target': n_target,
                'n_val_target': n_val_target,
                'h_val_acc': h_val_acc,
                'h_full_acc': h_full_acc,
                'epsilon': epsilon.item(),
                'dim': shfl_source_feats.shape[1]
            }
            df_entries[strategy] = entry

        method_df = pd.DataFrame(df_entries.values())
        print(method_df)
        results_df = pd.concat([results_df, method_df], ignore_index=True)

        total_evals += 1
        for strategy in df_entries.keys():
            violations[strategy] += (df_entries[strategy]['lower_bound'] > orig_trg_acc)

        results_df.to_pickle(filename)

    print(f'\n{dataset} TOTAL VIOLATIONS:\t{violations} / {total_evals}')
