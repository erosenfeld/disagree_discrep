import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_critic_source_loss(source_logits, source_pred_labels):
    return F.cross_entropy(source_logits, source_pred_labels, reduction='none')


def get_critic_target_loss(target_logits, target_pred_labels, loss_type='disagreement'):
    valid_loss_types = ['disagreement', 'DBAT', 'negative_xent']
    assert loss_type in valid_loss_types, f'Invalid loss type. Choose one of {valid_loss_types}.'

    if loss_type == 'negative_xent':
        return -F.cross_entropy(target_logits, target_pred_labels, reduction='none')
    else:
        logit_margin = get_critic_target_margin(target_logits, target_pred_labels, loss_type=loss_type)
        return torch.log1p(torch.exp(logit_margin))


def get_critic_target_margin(target_logits, target_pred_labels, loss_type='disagreement'):
    agree_logits = target_logits.gather(1, target_pred_labels.unsqueeze(1))[:, 0]

    if loss_type == 'disagreement':
        return (agree_logits.unsqueeze(1) - target_logits).sum(1) / (target_logits.shape[1] - 1)
    else:
        logsumexp_vals = target_logits.clone()
        logsumexp_vals[torch.arange(len(target_logits)), target_pred_labels] = float('-inf')
        return agree_logits - torch.logsumexp(logsumexp_vals, dim=1)

def train_multiple_critics(
        source_loader, target_loader, source_val_loader, target_val_loader,
        num_repeats=30, num_epochs=100, loss_type='disagreement'
):
    feats, logits, _ = next(iter(source_loader))
    device = feats.device
    dim, n_classes = feats.shape[1], logits.shape[1]

    stdv = 1. / math.sqrt(dim)
    critic_weights = nn.Parameter(torch.randn((num_repeats, dim, n_classes), device=device))
    critic_biases = nn.Parameter(torch.empty((num_repeats, 1, n_classes), device=device))
    critic_weights.data.uniform_(-stdv, stdv)
    critic_biases.data.uniform_(-stdv, stdv)

    opt = torch.optim.AdamW(
        [critic_weights, critic_biases],
        lr=3e-3, weight_decay=5e-4,
    )

    stat_names = [
        'tr_src_agree', 'ts_src_agree', 'tr_trg_agree', 'ts_trg_agree',
        'tr_src_loss', 'ts_src_loss', 'tr_trg_loss', 'ts_trg_loss',
    ]
    stats = {stat_name: [] for stat_name in stat_names}

    for epoch in range(num_epochs):
        epoch_stats = {stat_name: torch.zeros(num_repeats, device=device) for stat_name in stat_names}
        cnt = 0
        for source_batch, target_batch in zip(source_loader, target_loader):
            source_feats, h_source_logits, _ = source_batch
            target_feats, h_target_logits, _ = target_batch

            source_pred_labels = h_source_logits.argmax(1)
            target_pred_labels = h_target_logits.argmax(1)

            critic_source_logits = torch.einsum('nd,rdk->rnk', source_feats, critic_weights) + critic_biases
            critic_target_logits = torch.einsum('nd,rdk->rnk', target_feats, critic_weights) + critic_biases

            src_loss = get_critic_source_loss(torch.flatten(critic_source_logits, end_dim=1),
                                              source_pred_labels.repeat(num_repeats))
            trg_loss = get_critic_target_loss(torch.flatten(critic_target_logits, end_dim=1),
                                              target_pred_labels.repeat(num_repeats), loss_type=loss_type)
            loss = num_repeats * (src_loss.mean() + trg_loss.mean()) / 2

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_stats['tr_src_agree'] += (critic_source_logits.argmax(2) == source_pred_labels).sum(1)
            epoch_stats['tr_trg_agree'] += (critic_target_logits.argmax(2) == target_pred_labels).sum(1)
            epoch_stats['tr_src_loss'] += src_loss.reshape(num_repeats, -1).sum(1)
            epoch_stats['tr_trg_loss'] += trg_loss.reshape(num_repeats, -1).sum(1)
            cnt += len(source_pred_labels)

        stats['tr_src_agree'].append(epoch_stats['tr_src_agree'] / cnt)
        stats['tr_trg_agree'].append(epoch_stats['tr_trg_agree'] / cnt)
        stats['tr_src_loss'].append(epoch_stats['tr_src_loss'] / cnt)
        stats['tr_trg_loss'].append(epoch_stats['tr_trg_loss'] / cnt)

        ts_src_agree, ts_trg_agree, ts_src_loss, ts_trg_loss = eval_multiple_critics(
            critic_weights, critic_biases, source_val_loader, target_val_loader, loss_type=loss_type)
        stats['ts_src_agree'].append(ts_src_agree)
        stats['ts_trg_agree'].append(ts_trg_agree)
        stats['ts_src_loss'].append(ts_src_loss)
        stats['ts_trg_loss'].append(ts_trg_loss)

    return critic_weights, critic_biases, {k: torch.stack(v, dim=1).cpu().detach() for k, v in stats.items()}


def eval_multiple_critics(
        critic_weights, critic_biases,
        source_loader, target_loader,
        loss_type='disagreement'
):
    device = critic_weights.device
    num_repeats = critic_weights.shape[0]

    with torch.no_grad():
        n_val_source, n_val_target = len(source_loader.dataset), len(target_loader.dataset)
        source_agrees = torch.zeros(num_repeats, device=device)
        target_agrees = torch.zeros(num_repeats, device=device)
        source_losses = torch.zeros(num_repeats, device=device)
        target_losses = torch.zeros(num_repeats, device=device)

        for feats, h_logits, _ in source_loader:
            critic_logits = torch.einsum('nd,rdk->rnk', feats, critic_weights) + critic_biases
            source_agrees += (critic_logits.argmax(2) == h_logits.argmax(1)).sum(1)
            source_losses += get_critic_source_loss(
                torch.flatten(critic_logits, end_dim=1),
                h_logits.argmax(1).repeat(num_repeats)
            ).reshape(num_repeats, -1).sum(1)

        for feats, h_logits, _ in target_loader:
            critic_logits = torch.einsum('nd,rdk->rnk', feats, critic_weights) + critic_biases
            target_agrees += (critic_logits.argmax(2) == h_logits.argmax(1)).sum(1)
            target_losses += get_critic_target_loss(
                torch.flatten(critic_logits, end_dim=1),
                h_logits.argmax(1).repeat(num_repeats), loss_type=loss_type
            ).reshape(num_repeats, -1).sum(1)

        source_agrees /= n_val_source
        source_losses /= n_val_source
        target_agrees /= n_val_target
        target_losses /= n_val_target

    return source_agrees, target_agrees, source_losses, target_losses
