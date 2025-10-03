import os
from typing import List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

def detect_drum_columns_from_df(df: pd.DataFrame) -> List[str]:
    """
    Select the drum columns
    """
    cols = list(df.columns)
    if 'Sub_Beat' in cols:
        idx = cols.index('Sub_Beat')
        return cols[idx+1:]
    # choose columns that contain digits (like '36_Bass Drum 1')
    cand = [c for c in cols if any(ch.isdigit() for ch in str(c))]
    if cand:
        return cand
    exclude = {'Time_Slot', 'Measure', 'Beat', 'Sub_Beat', 'Cymbal'}
    return [c for c in cols if c not in exclude]

def build_master_columns(folder: str):
    """
    Scan all files and build master_columns.
    """
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
    """
    Read excel and return array aligned to master_columns order.
    Missing columns in the file are filled with zeros.
    """
    df = pd.read_csv(path)
    cols_present = list(df.columns)
    # we try to detect drum columns in this file to map them to master names.
    arr_cols = []
    for c in master_columns:
        if c in cols_present:
            coldata = df[c].fillna(0).astype(int).values
            arr_cols.append((c, coldata))
        else:
            # missing -> zeros
            arr_cols.append((c, np.zeros((len(df),), dtype=int)))
    # stack into (T, k)
    stacked = np.stack([col for (_, col) in arr_cols], axis=1)
    stacked = (stacked != 0).astype(int)
    return stacked

def load_folder_as_sequences(folder: str) -> List[np.ndarray]:
    """
    High-level: scan folder, build master_columns (if not given), then load each file and
    align to master_columns, returning list of arrays (T, k).
    """
    master_cols = build_master_columns(folder)
    seqs = []
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.csv')])
    for fname in files:
        path = os.path.join(folder, fname)
        try:
            arr = load_csv_to_grid_with_master(path, master_cols)
            # If df is shorter than some expected minimal length but still okay, we keep it
            seqs.append(arr)
        except Exception as e:
            print(f"[load_folder] skip {fname} due to error: {e}")
    return seqs

class ShortDrumDataset(Dataset):
    """
    Dataset for variable-length drum sequences.
    Each item: (x_tensor (T_i-1, k), y_tensor (T_i-1, k), length)
    where x = seq[:-1], y = seq[1:].
    """
    def __init__(self, seq_list: List[np.ndarray]):
        assert isinstance(seq_list, list)
        self.seq_list = [s.astype('float32') for s in seq_list]

    def __len__(self):
        return len(self.seq_list)

    def __getitem__(self, idx):
        seq = self.seq_list[idx]
        return torch.from_numpy(seq[:-1]), torch.from_numpy(seq[1:]), seq.shape[0]-1

def collate_pad(batch):
    """
    Collate function to pad variable-length batch.
    batch: list of (x, y, length)
    Returns: x_padded (B, Lmax, k), y_padded (B, Lmax, k), lengths (B,), mask (B, Lmax, 1)
    """
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
