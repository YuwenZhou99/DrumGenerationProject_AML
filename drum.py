import os
import random
import shutil
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch import nn
from typing import Dict, List

# ==================================================
# =============== Reproducibility ==================
# ==================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ==================================================
# =============== Dataset Functions ================
# ==================================================
def detect_drum_columns_from_df(df: pd.DataFrame) -> List[str]:
    cols = list(df.columns)
    if 'Sub_Beat' in cols:
        idx = cols.index('Sub_Beat')
        return cols[idx + 1:]
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


def load_folder_as_sequences(folder: str, compress_window: int = 2) -> List[np.ndarray]:
    master_cols = build_master_columns(folder)
    seqs = []
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.csv')])
    for fname in files:
        path = os.path.join(folder, fname)
        try:
            arr = load_csv_to_grid_with_master(path, master_cols)
            L = arr.shape[0] // compress_window
            arr = arr[:L * compress_window]
            arr = arr.reshape(L, compress_window, -1).max(axis=1)
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
        L, K = seq.shape

        beat_pos8 = np.arange(L) % 8 / 8.0
        beat_pos32 = np.arange(L) % 32 / 32.0
        beat_pos128 = np.arange(L) % 128 / 128.0

        beat_emb = np.stack([
            np.sin(2 * np.pi * beat_pos8), np.cos(2 * np.pi * beat_pos8),
            np.sin(2 * np.pi * beat_pos32), np.cos(2 * np.pi * beat_pos32),
            np.sin(2 * np.pi * beat_pos128), np.cos(2 * np.pi * beat_pos128)
        ], axis=1)

        seq_with_beat = np.concatenate([seq, beat_emb], axis=1).astype('float32')
        return (
            torch.from_numpy(seq_with_beat[:-1]),
            torch.from_numpy(seq_with_beat[1:]),
            seq.shape[0] - 1
        )


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
# ================== RNN Model =====================
# ==================================================
class DrumRNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.3, bidirectional: bool = False):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout, bidirectional=bidirectional)
        if bidirectional:
            hidden_dim *= 2
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, hidden=None):
        out, hidden = self.lstm(x, hidden)
        logits = self.fc(out)
        return logits, hidden


# ==================================================
# =================== Utils ========================
# ==================================================
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


# ==================================================
# ============== Training Functions ================
# ==================================================
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


# ==================================================
# =============== Generation Function ==============
# ==================================================
@torch.no_grad()
def generate_from_seed_groove(
    model, seed_drums, master_columns,
    steps=256, device="cpu", temperature=0.8, max_context=64
):
    model.eval()

    L0 = seed_drums.shape[0]
    pos8 = (np.arange(L0) % 8) / 8.0
    pos32 = (np.arange(L0) % 32) / 32.0
    pos128 = (np.arange(L0) % 128) / 128.0
    beat_emb = np.stack([
        np.sin(2*np.pi*pos8), np.cos(2*np.pi*pos8),
        np.sin(2*np.pi*pos32), np.cos(2*np.pi*pos32),
        np.sin(2*np.pi*pos128), np.cos(2*np.pi*pos128),
    ], axis=1)
    seed = np.concatenate([seed_drums, beat_emb], axis=1).astype(np.float32)

    context = torch.tensor(seed, dtype=torch.float32, device=device).unsqueeze(0)
    generated = []

    kick_idx = next((i for i, c in enumerate(master_columns) if "Kick" in c or "36" in c), None)
    snare_idx = next((i for i, c in enumerate(master_columns) if "Snare" in c or "38" in c), None)
    crash_idx = next((i for i, c in enumerate(master_columns) if "Crash" in c or "49" in c), None)

    for step in range(steps):
        logits, _ = model(context)
        probs = torch.sigmoid(logits[:, -1, :-6] / temperature)

        # --- musical rules ---
        if kick_idx is not None and snare_idx is not None:
            probs[0, snare_idx] *= (1.0 - 0.3 * probs[0, kick_idx])
        if crash_idx is not None and step % 32 == 0:
            probs[0, crash_idx] += 0.1 * torch.rand((), device=device)
        for i, c in enumerate(master_columns):
            if "Hi-Hat" in c or "46" in c:
                if step % 2 == 0:
                    probs[0, i] = probs[0, i] * 0.9 + 0.1
                else:
                    probs[0, i] = probs[0, i] * 0.9

        probs = probs.clamp(0.0, 1.0)  # <- crucial!
        next_drums = torch.bernoulli(probs)

        # --- add beat embedding for next step ---
        t = context.shape[1]
        p8, p32, p128 = (t % 8) / 8.0, (t % 32) / 32.0, (t % 128) / 128.0
        beat_emb_next = torch.tensor([[
            np.sin(2*np.pi*p8), np.cos(2*np.pi*p8),
            np.sin(2*np.pi*p32), np.cos(2*np.pi*p32),
            np.sin(2*np.pi*p128), np.cos(2*np.pi*p128),
        ]], device=device, dtype=next_drums.dtype)
        next_step_full = torch.cat([next_drums, beat_emb_next], dim=1)
        context = torch.cat([context, next_step_full.unsqueeze(1)], dim=1)
        if context.shape[1] > max_context:
            context = context[:, -max_context:, :]
        generated.append(next_drums.cpu().numpy())

    drum_seq = np.vstack(generated)
    L = drum_seq.shape[0]
    df = pd.DataFrame(drum_seq, columns=master_columns)
    df.insert(0, "Time_Slot", np.arange(L))
    df.insert(1, "Measure", df["Time_Slot"] // 64 + 1)
    df.insert(2, "Beat", (df["Time_Slot"] % 64) // 16 + 1)
    df.insert(3, "Sub_Beat", (df["Time_Slot"] % 16) + 1)
    return df


# ==================================================
# ==================== Main ========================
# ==================================================
def main():
    data_folder = "/kaggle/input/drum-dataset/RockBinary_Dataset"
    model_path = "/kaggle/working/drum_rnn.pt"
    out_folder = "/kaggle/working/generated_samples"

    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)
    os.makedirs(out_folder, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ---------- 1. Train ----------
    seqs = load_folder_as_sequences(data_folder, compress_window=2)
    print("Total sequences:", len(seqs))
    idx = np.random.permutation(len(seqs))
    split = int(len(seqs) * 0.7)
    train_seqs = [seqs[i] for i in idx[:split]]
    val_seqs = [seqs[i] for i in idx[split:]]

    train_ds = ShortDrumDataset(train_seqs)
    val_ds = ShortDrumDataset(val_seqs)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_pad)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_pad)

    k = seqs[0].shape[1] + 6
    model = DrumRNN(input_dim=k, hidden_dim=128, num_layers=2, dropout=0.3, bidirectional=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    best_val = float("inf")
    threshold = 0.1

    for epoch in range(1, 21):
        tr_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_metrics, probs, all_true = validate(model, val_loader, device, threshold)
        print(f"Epoch {epoch}: train_bce={tr_loss:.4f} val_bce={val_loss:.4f} "
              f"micro_f1={val_metrics['micro_f1']:.4f} macro_f1={val_metrics['macro_f1']:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            save_model(model_path, model, optimizer, epoch=epoch,
                       meta={"val_loss": val_loss, "metrics": val_metrics})
            np.save("/kaggle/working/master_columns.npy", np.array(build_master_columns(data_folder)))
            print("master_columns saved.")

    # ---------- 2. Generate ----------
    model = DrumRNN(input_dim=k).to(device)
    load_model(model_path, model, map_location=device)
    model.to(device).eval()
    print("\nModel loaded for generation.")
    master_columns = np.load("/kaggle/working/master_columns.npy", allow_pickle=True).tolist()
    all_csv = [f for f in sorted(os.listdir(data_folder)) if f.endswith(".csv")]
    selected_files = random.sample(all_csv, min(5, len(all_csv)))

    for i, fname in enumerate(selected_files, 1):
        path = os.path.join(data_folder, fname)
        seq_raw = load_csv_to_grid_with_master(path, master_columns)

        # unify compression scale with training
        L = seq_raw.shape[0] // 2
        seq_raw = seq_raw[:L*2].reshape(L, 2, -1).max(axis=1)

        if seq_raw.shape[0] < 64:
            continue
        seed_drums = seq_raw[:64].astype(np.float32)
        df = generate_from_seed_groove(model, seed_drums, master_columns, device=device)
        out_path = os.path.join(out_folder, f"gen_sparse_{i}_{os.path.splitext(fname)[0]}.csv")
        df.to_csv(out_path, index=False)
        print(f"Generated CSV saved: {out_path}")


if __name__ == "__main__":
    main()
