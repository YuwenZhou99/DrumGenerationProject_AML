import os
import numpy as np
from typing import Dict
import torch
from torch import nn

def compress_windows(y: np.ndarray, window: int = 2):
    y = np.asarray(y)
    N, L, k = y.shape
    Lw = L // window
    if Lw == 0:
        return np.zeros((N, 0, k), dtype=int)
    y_trunc = y[:, :Lw * window, :]
    y_reshaped = y_trunc.reshape(N, Lw, window, k)
    y_comp = (y_reshaped.max(axis=2) > 0).astype(int)
    return y_comp


def f1_metrics_window(y_true: np.ndarray, y_prob: np.ndarray,
                      threshold: float = 0.3, window: int = 2) -> Dict[str, float]:
    probs = np.asarray(y_prob)
    true = np.asarray(y_true)
    pred_bin = (probs >= threshold).astype(int)
    true_bin = (true >= 0.5).astype(int)
    pred_comp = compress_windows(pred_bin, window=window)
    true_comp = compress_windows(true_bin, window=window)
    if pred_comp.size == 0 or true_comp.size == 0:
        return {"micro_f1": 0.0, "macro_f1": 0.0, "precision": 0.0, "recall": 0.0}
    pred_flat = pred_comp.reshape(-1)
    true_flat = true_comp.reshape(-1)
    tp = int(((true_flat == 1) & (pred_flat == 1)).sum())
    fp = int(((true_flat == 0) & (pred_flat == 1)).sum())
    fn = int(((true_flat == 1) & (pred_flat == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    micro_f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    f1s = []
    pred_ch = pred_comp.reshape(-1, pred_comp.shape[2])
    true_ch = true_comp.reshape(-1, true_comp.shape[2])
    for ch in range(pred_comp.shape[2]):
        yt = true_ch[:, ch].astype(int)
        yp = pred_ch[:, ch].astype(int)
        tp_c = int(((yt == 1) & (yp == 1)).sum())
        fp_c = int(((yt == 0) & (yp == 1)).sum())
        fn_c = int(((yt == 1) & (yp == 0)).sum())
        prec_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
        rec_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
        f1_c = (2 * prec_c * rec_c / (prec_c + rec_c)) if (prec_c + rec_c) > 0 else 0.0
        f1s.append(f1_c)
    macro_f1 = float(np.mean(f1s))
    return {"micro_f1": micro_f1, "macro_f1": macro_f1, "precision": prec, "recall": rec}


def save_model(path: str, model: nn.Module, optimizer=None, epoch: int = 0, meta: dict = None):
    payload = {"model_state": model.state_dict(), "epoch": epoch}
    if optimizer is not None:
        payload["opt_state"] = optimizer.state_dict()
    if meta is not None:
        payload["meta"] = meta
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(payload, path)


def load_model(path: str, model: nn.Module, map_location="cpu"):
    ck = torch.load(path, map_location=map_location)
    model.load_state_dict(ck["model_state"], strict=False)
    return model