import os
import random
import shutil
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch import nn
from typing import Dict, List
from dataset import load_folder_as_sequences, load_csv_to_grid_with_master, ShortDrumDataset, collate_pad, build_master_columns
from RNNModel import DrumRNN
from train import train_epoch, validate
from utils import save_model, load_model
from generate import generate_from_seed_groove

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)




def main():
    # Parameters for training and generation
    train = True
    generate = True

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_folder = os.path.join(project_root, "RockBinary_Dataset/RockBinary_Dataset")
    model_path = os.path.join(project_root, "models/drum_rnn.pt")
    out_folder = os.path.join(project_root, "generated_samples")

    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)
    os.makedirs(out_folder, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    seqs = load_folder_as_sequences(data_folder, compress_window=2)
    idx = np.random.permutation(len(seqs))
    split = int(len(seqs) * 0.8)
    train_seqs = [seqs[i] for i in idx[:split]]
    val_seqs = [seqs[i] for i in idx[split:]]

    train_ds = ShortDrumDataset(train_seqs)
    val_ds = ShortDrumDataset(val_seqs)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_pad)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_pad)

    k = seqs[0].shape[1] + 6

    # Training parameters
    hidden_dim = 256
    num_layers = 3
    dropout = 0.3
    learning_rate = 1e-3
    threshold = 0.3 
    epochs = 80

    model = DrumRNN(input_dim=k, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, bidirectional=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_val = float("inf")

    if train:
        for epoch in range(1, epochs + 1):
            tr_loss = train_epoch(model, train_loader, optimizer, device)
            val_loss, val_metrics, probs, all_true = validate(model, val_loader, device, threshold)
            print(f"Epoch {epoch}: train_bce={tr_loss:.4f} val_bce={val_loss:.4f} "
                f"micro_f1={val_metrics['micro_f1']:.4f} macro_f1={val_metrics['macro_f1']:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                save_model(model_path, model, optimizer, epoch=epoch,
                        meta={"val_loss": val_loss, "metrics": val_metrics})
                master_columns_path = os.path.join(project_root, "models/master_columns.npy")
                np.save(master_columns_path, np.array(build_master_columns(data_folder)))
                print("master_columns saved.")

    if generate:
        model = DrumRNN(input_dim=k, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, bidirectional=False).to(device)
        load_model(model_path, model, map_location=device)
        model.to(device).eval()
        print("\nModel loaded for generation.")
        
        master_columns_path = os.path.join(project_root, "models/master_columns.npy")
        master_columns = np.load(master_columns_path, allow_pickle=True).tolist()
        all_csv = [f for f in sorted(os.listdir(data_folder)) if f.endswith(".csv")]
        selected_files = random.sample(all_csv, min(10, len(all_csv)))

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
