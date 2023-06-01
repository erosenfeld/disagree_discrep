import torch
from torch import nn
import numpy as np

from .critic import get_critic_source_loss, get_critic_target_loss, train_multiple_critics, eval_multiple_critics
from .utils import create_dataloader


def validation_scores_and_bounds_from_loaders(
        source_loader, source_val_loader,
        target_loader, target_val_loader,
        num_repeats=30, num_epochs=100, critic=None, delta=0.01, loss_type='disagreement'
):

    feats, logits, _ = next(iter(source_loader))
    dim = feats.shape[1]
    num_classes = logits.shape[1]
    device = logits.device
    critic_stats = {}
    n_source, n_target = len(source_loader.dataset), len(target_loader.dataset)
    n_val_source, n_val_target = len(source_val_loader.dataset), len(target_val_loader.dataset)

    if critic is None:
        critic_weights, critic_biases, critic_stats = \
            train_multiple_critics(source_loader, target_loader, source_val_loader, target_val_loader,
                                   num_repeats=num_repeats, num_epochs=num_epochs, loss_type=loss_type)
        train_agree_diffs = critic_stats['tr_src_agree'] - critic_stats['tr_trg_agree']
        test_agree_diffs = critic_stats['ts_src_agree'] - critic_stats['ts_trg_agree']
        best_critic_ind = test_agree_diffs.argmax() // num_epochs

        critic = nn.Linear(dim, num_classes, device=device)
        critic.weight.data = critic_weights[best_critic_ind].T
        critic.bias.data = critic_biases[best_critic_ind]

    else:
        source_agrees, target_agrees, _, _, _ = eval_multiple_critics(
            critic.weight.T.unsqueeze(0), critic.bias.unsqueeze(0),
            source_loader, target_loader, loss_type=loss_type)
        source_val_agrees, target_val_agrees, _, _, _ = eval_multiple_critics(
            critic.weight.T.unsqueeze(0), critic.bias.unsqueeze(0),
            source_val_loader, target_val_loader, loss_type=loss_type)
        critic_stats['tr_src_agree'], critic_stats['tr_trg_agree'] = source_agrees, target_agrees
        critic_stats['ts_src_agree'], critic_stats['ts_trg_agree'] = source_val_agrees, target_val_agrees
        train_agree_diffs = (source_agrees - target_agrees).unsqueeze(0)
        test_agree_diffs = (source_val_agrees - target_val_agrees).unsqueeze(0)

    with torch.no_grad():
        h_acc = h_val_acc = 0.
        critic_source_val_loss = 0.
        for feats, h_logits, labels in source_val_loader:
            h_val_acc += (h_logits.argmax(1) == labels).sum().item()
            if not critic_stats:
                critic_source_val_loss += get_critic_source_loss(critic(feats), h_logits.argmax(1)).sum().item()
        h_val_acc /= n_val_source
        critic_source_val_loss = critic_source_val_loss / n_val_source

        if 'ts_src_loss' in critic_stats:
            critic_source_val_loss = critic_stats['ts_src_loss'][best_critic_ind, -1].item()
            critic_target_val_loss = critic_stats['ts_trg_loss'][best_critic_ind, -1].item()
        else:
            critic_target_val_loss = 0.
            for feats, h_logits, labels in target_val_loader:
                critic_logits = critic(feats)
                h_pred_labels = h_logits.argmax(1)
                critic_target_val_loss += get_critic_target_loss(
                    critic_logits, h_pred_labels, loss_type=loss_type).sum().item()
            critic_target_val_loss = critic_target_val_loss / n_val_target

        for feats, h_logits, labels in source_loader:
            h_acc += (h_logits.argmax(1) == labels).sum().item()
        h_acc /= n_source
        h_full_acc = (h_acc * n_source + h_val_acc * n_val_source) / (n_source + n_val_source)

        epsilon = np.sqrt((n_val_source + 4 * n_val_target) * np.log(1. / delta) / (2 * n_val_target * n_val_source))
        critic_stats['tr_agree_diffs'] = train_agree_diffs
        critic_stats['ts_agree_diffs'] = test_agree_diffs
        critic_stats['src_val_loss'] = critic_source_val_loss
        critic_stats['trg_val_loss'] = critic_target_val_loss

    return critic_stats, (h_val_acc, h_full_acc), epsilon


def validation_scores_and_bounds_from_split_data(
        source_feats, source_val_feats, source_logits, source_val_logits, source_labels, source_val_labels,
        target_feats, target_val_feats, target_logits, target_val_logits, target_labels, target_val_labels,
        num_repeats=30, num_epochs=100, critic=None, delta=0.01, loss_type='disagreement'
):

    source_loader = create_dataloader(source_feats, source_logits, source_labels)
    target_loader = create_dataloader(target_feats, target_logits, target_labels)
                    
    source_val_loader = create_dataloader(source_val_feats, source_val_logits, source_val_labels, shuffle=False)
    target_val_loader = create_dataloader(target_val_feats, target_val_logits, target_val_labels, shuffle=False)

    return validation_scores_and_bounds_from_loaders(
        source_loader, source_val_loader, target_loader, target_val_loader,
        num_repeats=num_repeats, num_epochs=num_epochs, critic=critic, delta=delta, loss_type=loss_type
    )