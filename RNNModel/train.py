import torch
import numpy as np
from utils import f1_metrics_window

_pos_weight_tensor = None
def masked_bce_with_logits(logits, targets, mask):
    global _pos_weight_tensor
    if _pos_weight_tensor is None:
        _pos_weight_tensor = torch.tensor([3.0], device=logits.device)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction='none', pos_weight=_pos_weight_tensor
    )
    mask3 = mask.expand_as(bce)
    loss = (bce * mask3).sum() / mask3.sum()
    return loss


@torch.no_grad()
def validate(model, loader, device, threshold=0.1, window=2):
    model.eval()
    all_logits_flat, all_true_flat = [], []
    total_loss = 0.0
    num_batches = 0

    for x, y, lengths, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        logits, _ = model(x)
        logits = logits[:, :, :-6]
        y = y[:, :, :-6]

        loss = masked_bce_with_logits(logits, y, mask)
        total_loss += loss.item()
        num_batches += 1

        mask_np = mask.detach().cpu().numpy().astype(bool).squeeze(-1)
        logits_np = logits.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()

        all_logits_flat.append(logits_np[mask_np])
        all_true_flat.append(y_np[mask_np])

    all_logits_flat = np.concatenate(all_logits_flat, axis=0)
    all_true_flat = np.concatenate(all_true_flat, axis=0)
    all_logits = all_logits_flat[None, :, :]
    all_true = all_true_flat[None, :, :]

    probs = 1.0 / (1.0 + np.exp(-all_logits))
    metrics = f1_metrics_window(all_true, probs, threshold=threshold, window=window)
    avg_loss = total_loss / max(1, num_batches)
    return avg_loss, metrics, probs, all_true


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    num_batches = 0
    for x, y, lengths, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        optimizer.zero_grad()
        logits, _ = model(x)
        logits = logits[:, :, :-6]
        y = y[:, :, :-6]
        loss = masked_bce_with_logits(logits, y, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    return total_loss / max(1, num_batches)


