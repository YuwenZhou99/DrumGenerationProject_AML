import numpy as np
import torch
import os
from typing import Dict

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def compress_windows(y: np.ndarray, window: int = 2):
    """
        Vectorized window-OR compression without mask/hop.
        y: (N, L, k) binary or 0/1 float array.
        window: integer window size (e.g., 4).
        Returns:
          y_comp: (N, Lw, k) binary array where Lw = L // window (tail truncated)
        """
    y = np.asarray(y)
    if y.ndim != 3:
        raise ValueError("y must have shape (N, L, k)")
    N, L, k = y.shape
    Lw = L // window
    if Lw == 0:
        # no full windows -> return empty arrays
        return np.zeros((N, 0, k), dtype=int)
    # truncate tail so that length = Lw * window
    y_trunc = y[:, : Lw * window, :]  # (N, Lw*window, k)
    # reshape to (N, Lw, window, k) then OR across axis=2
    y_reshaped = y_trunc.reshape(N, Lw, window, k)
    y_comp = (y_reshaped.max(axis=2) > 0).astype(int)  # (N, Lw, k)
    return y_comp

def f1_metrics_window(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.3, window: int = 2) -> Dict[str, float]:
    """
    Compute micro/macro F1 after compressing every non-overlapping `window` steps by OR.
    y_true, y_prob: (N, L, k)
    threshold: binarize y_prob >= threshold
    window: window size in time steps
    Returns dict: micro_f1, macro_f1, precision, recall
    """
    probs = np.asarray(y_prob)
    true = np.asarray(y_true)
    if probs.shape != true.shape:
        raise ValueError("y_true and y_prob must have same shape (N,L,k)")

    # Binarize and compress
    pred_bin = (probs >= threshold).astype(int)
    true_bin = (true >= 0.5).astype(int)   # assume ground truth is 0/1; keep conservative threshold

    pred_comp = compress_windows(pred_bin, window=window)  # (N, Lw, k)
    true_comp = compress_windows(true_bin, window=window)  # (N, Lw, k)

    # If no windows, return zeros
    if pred_comp.size == 0 or true_comp.size == 0:
        return {"micro_f1": 0.0, "macro_f1": 0.0, "precision": 0.0, "recall": 0.0}

    # flatten across windows and channels
    pred_flat = pred_comp.reshape(-1)   # shape (N * Lw * k,)
    true_flat = true_comp.reshape(-1)

    # micro metrics
    tp = int(((true_flat == 1) & (pred_flat == 1)).sum())
    fp = int(((true_flat == 0) & (pred_flat == 1)).sum())
    fn = int(((true_flat == 1) & (pred_flat == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    micro_f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    # per-channel (macro)
    Nw = pred_comp.shape[0] * pred_comp.shape[1]  # number of windows total per channel
    k = pred_comp.shape[2]
    f1s = []
    pred_ch = pred_comp.reshape(-1, k)   # (N*Lw, k)
    true_ch = true_comp.reshape(-1, k)
    for ch in range(k):
        yt = true_ch[:, ch].astype(int)
        yp = pred_ch[:, ch].astype(int)
        tp_c = int(((yt == 1) & (yp == 1)).sum())
        fp_c = int(((yt == 0) & (yp == 1)).sum())
        fn_c = int(((yt == 1) & (yp == 0)).sum())
        prec_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
        rec_c  = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
        f1_c = (2 * prec_c * rec_c / (prec_c + rec_c)) if (prec_c + rec_c) > 0 else 0.0
        f1s.append(f1_c)
    macro_f1 = float(np.mean(f1s))
    return {"micro_f1": micro_f1, "macro_f1": macro_f1, "precision": prec, "recall": rec}

def save_model(path: str, model: torch.nn.Module, optimizer=None, epoch: int = 0, meta: dict = None):
    payload = {"model_state": model.state_dict(), "epoch": epoch}
    if optimizer is not None:
        payload["opt_state"] = optimizer.state_dict()
    if meta is not None:
        payload["meta"] = meta
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(payload, path)

def load_model(path: str, model: torch.nn.Module, map_location='cpu'):
    ck = torch.load(path, map_location=map_location)
    model.load_state_dict(ck["model_state"])
    return ck

