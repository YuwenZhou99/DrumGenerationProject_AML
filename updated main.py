# Kaggle DrumRNN Training Notebook
# ==================================================
# This notebook merges your training script, dataset.py, RNNModel.py, and utils functions.

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from typing import Dict, List
import pandas as pd

# ==================================================
# dataset.py content
# ==================================================
def detect_drum_columns_from_df(df: pd.DataFrame) -> List[str]:
    cols = list(df.columns)
    if 'Sub_Beat' in cols:
        idx = cols.index('Sub_Beat')
        return cols[idx+1:]
    cand = [c for c in cols if any(ch.isdigit() for ch in str(c))]
    if cand:
        return cand
    exclude = {'Time_Slot', 'Measure', 'Beat', 'Sub_Beat', 'Cymbal'}
    return [c for c in cols if c not in exclude]

def build_master_columns(folder: str):
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.csv')])
    if not files:
        raise RuntimeError(f"No .csv in {folder}")
    sets = []
    for fname in files:
        path = os.path.join(folder, fname)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[build_master_columns] failed to read {fname}: {e}")
            continue
        cols = detect_drum_columns_from_df(df)
        sets.append(set(cols))
    master = sorted(list(set().union(*sets)))
    return master

def load_csv_to_grid_with_master(path: str, master_columns: List[str]) -> np.ndarray:
    df = pd.read_csv(path)
    cols_present = list(df.columns)
    arr_cols = []
    for c in master_columns:
        if c in cols_present:
            coldata = df[c].fillna(0).astype(int).values
            arr_cols.append((c, coldata))
        else:
            arr_cols.append((c, np.zeros((len(df),), dtype=int)))
    stacked = np.stack([col for (_, col) in arr_cols], axis=1)
    stacked = (stacked != 0).astype(int)
    return stacked

def load_folder_as_sequences(folder: str) -> List[np.ndarray]:
    master_cols = build_master_columns(folder)
    seqs = []
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.csv')])
    for fname in files:
        path = os.path.join(folder, fname)
        try:
            arr = load_csv_to_grid_with_master(path, master_cols)
            seqs.append(arr)
        except Exception as e:
            print(f"[load_folder] skip {fname} due to error: {e}")
    return seqs

class ShortDrumDataset(torch.utils.data.Dataset):
    def __init__(self, seq_list: List[np.ndarray]):
        assert isinstance(seq_list, list)
        self.seq_list = [s.astype('float32') for s in seq_list]

    def __len__(self):
        return len(self.seq_list)

    def __getitem__(self, idx):
        seq = self.seq_list[idx]
        return torch.from_numpy(seq[:-1]), torch.from_numpy(seq[1:]), seq.shape[0]-1

def collate_pad(batch):
    xs, ys, lengths = zip(*batch)
    k = xs[0].shape[1]
    Lmax = max(x.shape[0] for x in xs)
    B = len(xs)
    x_pad = torch.zeros((B, Lmax, k), dtype=xs[0].dtype)
    y_pad = torch.zeros((B, Lmax, k), dtype=ys[0].dtype)
    mask = torch.zeros((B, Lmax, 1), dtype=torch.float32)
    for i, (x, y, l) in enumerate(batch):
        x_pad[i, :x.shape[0], :] = x
        y_pad[i, :y.shape[0], :] = y
        mask[i, :x.shape[0], 0] = 1.0
    lengths = torch.tensor(lengths, dtype=torch.long)
    return x_pad, y_pad, lengths, mask

# ==================================================
# RNNModel.py content
# ==================================================
class DrumRNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, hidden=None):
        out, hidden = self.lstm(x, hidden)
        logits = self.fc(out)
        return logits, hidden

# ==================================================
# utils functions
# ==================================================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def compress_windows(y: np.ndarray, window: int = 4):
    y = np.asarray(y)
    N, L, k = y.shape
    Lw = L // window
    if Lw == 0:
        return np.zeros((N, 0, k), dtype=int)
    y_trunc = y[:, : Lw * window, :]
    y_reshaped = y_trunc.reshape(N, Lw, window, k)
    y_comp = (y_reshaped.max(axis=2) > 0).astype(int)
    return y_comp

def f1_metrics_window(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.3, window: int = 4) -> Dict[str, float]:
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
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
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
        rec_c  = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
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

# ==================================================
# Training script
# ==================================================
def masked_bce_with_logits(logits, targets, mask):
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    mask3 = mask.expand_as(bce)
    return (bce * mask3).sum() / mask3.sum()

@torch.no_grad()
def validate(model, loader, device, threshold=0.1):
    model.eval()
    all_logits_flat, all_true_flat = [], []
    total_loss, total_n = 0.0, 0

    for x, y, lengths, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        logits, _ = model(x)
        loss = masked_bce_with_logits(logits, y, mask)
        total_loss += loss.item() * x.size(0)

        # ---- Extract valid time steps (according to mask) ----
        mask_np = mask.cpu().numpy().astype(bool)
        logits_np = logits.cpu().numpy()[mask_np.squeeze()]  # shape (M, k)
        y_np = y.cpu().numpy()[mask_np.squeeze()]             # shape (M, k)

        all_logits_flat.append(logits_np)
        all_true_flat.append(y_np)
        total_n += x.size(0)

    # Concatenate all valid frames
    all_logits_flat = np.concatenate(all_logits_flat, axis=0)  # (T_total, k)
    all_true_flat = np.concatenate(all_true_flat, axis=0)      # (T_total, k)

    # Add a batch dimension artificially (N=1, L=T_total, k)
    all_logits = all_logits_flat[None, :, :]
    all_true = all_true_flat[None, :, :]

    # Compute probabilities and evaluation metrics
    probs = 1.0 / (1.0 + np.exp(-all_logits))
    metrics = f1_metrics_window(all_true, probs, threshold=threshold, window=4)
    avg_loss = total_loss / total_n

    return avg_loss, metrics, probs, all_true


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, y, lengths, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = masked_bce_with_logits(logits, y, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

# ==================================================
# Main training loop
# ==================================================
data_folder = "/kaggle/input/drum-dataset/RockBinary_Dataset" 
save_path = "/kaggle/working/drum_rnn.pt"
epochs = 5
batch_size = 32
lr = 1e-3
hidden_dim = 128
num_layers = 2
dropout = 0.2
threshold = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
seqs = load_folder_as_sequences(data_folder)
print("Total sequences:", len(seqs))
idx = np.random.permutation(len(seqs))
split = int(len(seqs) * 0.7)
train_seqs = [seqs[i] for i in idx[:split]]
val_seqs = [seqs[i] for i in idx[split:]]
train_ds = ShortDrumDataset(train_seqs)
val_ds = ShortDrumDataset(val_seqs)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_pad)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_pad)

k = seqs[0].shape[1]
model = DrumRNN(input_dim=k, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

best_val = float("inf")
for epoch in range(1, epochs + 1):
    tr_loss = train_epoch(model, train_loader, optimizer, device)
    val_loss, val_metrics, probs, all_true = validate(model, val_loader, device, threshold=threshold)
    print(f"Epoch {epoch}: train_bce={tr_loss:.4f} val_bce={val_loss:.4f} micro_f1={val_metrics['micro_f1']:.4f} macro_f1={val_metrics['macro_f1']:.4f}")
    best_thr, best_f1 = threshold, val_metrics["micro_f1"]
    for thr in np.linspace(0.1, 0.9, 17):
        m = f1_metrics_window(all_true, probs, threshold=thr, window=4)
        if m["micro_f1"] > best_f1:
            best_f1, best_thr = m["micro_f1"], thr
    print(f" Best val thr={best_thr:.2f} best_micro_f1={best_f1:.4f}")
    if val_loss < best_val:
        best_val = val_loss
        save_model(save_path, model, optimizer, epoch=epoch, meta={"val_loss": val_loss, "metrics": val_metrics})
        print("Saved best model to", save_path)
