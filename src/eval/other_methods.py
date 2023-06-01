import torch
import numpy as np
import pandas as pd
import ot
import os
import itertools
import argparse

from src.lib.consts import DATASET_SHIFTS, TRAIN_METHODS
from ..lib.utils import calibration_temp, negentropy, load_shift_data


# Average Threshold Confidence
def predict_ATC_maxconf(source_logits, source_labels, target_logits):
    source_scores = torch.softmax(source_logits, dim=1).amax(1)
    target_scores = torch.softmax(target_logits, dim=1).amax(1)
    sorted_source_scores, _ = torch.sort(source_scores)
    threshold = sorted_source_scores[-(source_logits.argmax(1) == source_labels).sum()]
    estimate = (target_scores > threshold).float().mean().item()
    return estimate


def predict_ATC_negent(source_logits, source_labels, target_logits):
    source_scores = negentropy(source_logits)
    target_scores = negentropy(target_logits)
    sorted_source_scores, _ = torch.sort(source_scores)
    threshold = sorted_source_scores[-(source_logits.argmax(1) == source_labels).sum()]
    estimate = (target_scores > threshold).float().mean().item()
    return estimate


# Confidence Optimal Transport
def predict_COT(source_logits, source_labels, target_logits):
    num_classes = source_logits.shape[1]
    source_label_dist = torch.nn.functional.one_hot(source_labels).float().mean(0)
    target_probs = torch.softmax(target_logits, dim=1)
    cost_matrix = torch.stack([(target_probs - onehot).abs().sum(1)
                               for onehot in torch.eye(num_classes, device=source_logits.device)], dim=1) / 2
    ot_plan = ot.emd(torch.ones(len(target_probs)) / len(target_probs), source_label_dist, cost_matrix)
    ot_cost = torch.sum(ot_plan * cost_matrix.cpu()).item()

    s_conf = torch.softmax(source_logits, dim=1).amax(1).mean().item()
    s_acc = (source_logits.argmax(1) == source_labels).float().mean().item()
    conf_gap = s_conf - s_acc
    err_est = ot_cost + conf_gap
    return 1. - err_est


# Average Confidence
def predict_AC(source_logits, source_labels, target_logits):
    return torch.softmax(target_logits, dim=1).amax(1).mean().item()


# Difference of Confidences
def predict_DOC(source_logits, source_labels, target_logits):
    avg_source_conf = torch.softmax(source_logits, dim=1).amax(1).mean().item()
    avg_target_conf = torch.softmax(target_logits, dim=1).amax(1).mean().item()
    source_acc = (source_logits.argmax(1) == source_labels).float().mean().item()

    return source_acc + (avg_target_conf - avg_source_conf)


PREDICT_METHOD_FUNCS = {
    'ATC': predict_ATC_maxconf,
    'ATC_NE': predict_ATC_negent,
    'COT': predict_COT,
    'AC': predict_AC,
    'DOC': predict_DOC,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--feats_dir", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    args = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('DEVICE:', device)
    filename = os.path.join(args.results_dir, 'other_methods.pkl')
    results = pd.DataFrame()
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    for dataset in DATASET_SHIFTS.keys():
        for shift, method in itertools.product(DATASET_SHIFTS[dataset], TRAIN_METHODS):

            try:
                (source_feats, source_labels), (target_feats, target_labels), \
                    (h1_source_out, h1_target_out) = load_shift_data(args.feats_dir, dataset, shift, method, device)
                orig_src_acc = (h1_source_out.argmax(1) == source_labels).float().mean().item()
                orig_trg_acc = (h1_target_out.argmax(1) == target_labels).float().mean().item()

            except FileNotFoundError:
                print(f'Couldn\'t load {dataset} {method}_{shift}\n')
                continue

            print(f'\ndataset: {dataset}\tshift: {shift}\tmethod: {method}')
            source_temp = calibration_temp(h1_source_out.detach(), source_labels.detach())
            temperatures = {
                'none': 1.0,
                'source': source_temp,
            }

            for temp_type, temp in temperatures.items():
                scaled_source_logits = h1_source_out * temp
                scaled_target_logits = h1_target_out * temp

                for method_name, method_func in PREDICT_METHOD_FUNCS.items():
                    estimate = method_func(scaled_source_logits, source_labels, scaled_target_logits)
                    entry = pd.DataFrame({
                        'bound_valid': estimate <= orig_trg_acc,
                        'trg_accuracy': orig_trg_acc,
                        'lower_bound': estimate,
                        'dataset': dataset,
                        'shift': str(shift),
                        'train_method': method,
                        'src_accuracy': orig_src_acc,
                        'prediction_method': method_name,
                        'temperature': temp_type,
                    }, index=[0])
                    results = pd.concat([results, entry], ignore_index=True)

    results.to_pickle(filename)
