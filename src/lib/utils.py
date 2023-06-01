import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def np_to_device(data, device):
    return torch.from_numpy(data).to(device)


def create_dataloader(feats, logits, labels, shuffle=True, batch_size=256):
    return DataLoader(TensorDataset(feats, logits, labels), shuffle=shuffle, batch_size=batch_size)


def calibration_temp(logits, labels):
    log_temp = torch.nn.Parameter(torch.tensor([0.], device=logits.device))
    temp_opt = torch.optim.LBFGS([log_temp], lr=.1, max_iter=50, tolerance_change=5e-5)

    def closure_fn():
        loss = F.cross_entropy(logits * torch.exp(log_temp), labels)
        temp_opt.zero_grad()
        loss.backward()
        return loss

    temp_opt.step(closure_fn)
    return torch.exp(log_temp).item()


def negentropy(logits):
    return (torch.softmax(logits, dim=1) * torch.log_softmax(logits, dim=1)).sum(1)


def load_shift_data(feats_dir, dataset, shift, method, device):
    shift_dir = os.path.join(feats_dir, dataset, f'{method}_{shift}_100.0')
    feats = np.load(os.path.join(shift_dir, 'model_feats.npy'), allow_pickle=True).item()
    labels = np.load(os.path.join(shift_dir, 'ytrue.npy'), allow_pickle=True).item()

    source_feats, source_labels = np_to_device(feats['source'], device), np_to_device(labels['source'], device)
    target_feats, target_labels = np_to_device(feats['target'], device), np_to_device(labels['target'], device)

    lin = torch.nn.Linear(source_feats.shape[1], source_labels.max() + 1, device=device)
    lin.load_state_dict(torch.load(os.path.join(shift_dir, 'linear.pth'), map_location=device))
    h1_source_out = (source_feats @ lin.weight.T + lin.bias).detach()
    h1_target_out = (target_feats @ lin.weight.T + lin.bias).detach()

    return (source_feats, source_labels), (target_feats, target_labels), (h1_source_out, h1_target_out)