import torch
import pandas as pd
import numpy as np
import os
from typing import List
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



