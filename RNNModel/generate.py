import os
import random
import shutil
import numpy as np
import torch
import pandas as pd
from utils import load_model
from dataset import build_master_columns, load_csv_to_grid_with_master
from RNNModel import DrumRNN


@torch.no_grad()
def generate_from_seed_groove(
    model,
    seed_drums,
    master_columns,
    steps=512,
    device="cpu",
    temperature=0.8,
    base_threshold=0.2,
    max_context=64
):
    
    model.eval()
    L0 = seed_drums.shape[0]
    beat_pos = (np.arange(L0) % 64) / 64.0
    beat_emb = np.stack([np.sin(2*np.pi*beat_pos), np.cos(2*np.pi*beat_pos)], axis=1)
    seed = np.concatenate([seed_drums, beat_emb], axis=1).astype(np.float32)

    context = torch.tensor(seed, dtype=torch.float32, device=device).unsqueeze(0)
    generated = []

    
    kick_idx = next((i for i, c in enumerate(master_columns) if "Kick" in c or "36" in c), None)
    snare_idx = next((i for i, c in enumerate(master_columns) if "Snare" in c or "38" in c), None)
    crash_idx = next((i for i, c in enumerate(master_columns) if "Crash" in c or "49" in c), None)

    for step in range(steps):
        logits, _ = model(context)
        probs = torch.sigmoid(logits[:, -1, :-2] / temperature)

    
        probs = torch.sigmoid(logits[:, -1, :-2] / temperature)
        next_drums = torch.bernoulli(probs)   

    
        if kick_idx is not None and snare_idx is not None:
            if next_drums[0, kick_idx] == 1 and random.random() < 0.2:
                next_drums[0, snare_idx] = 0  

    
        if crash_idx is not None:
            if step % 16 == 0 and random.random() < 0.5:
                next_drums[0, crash_idx] = 1
            else:
                next_drums[0, crash_idx] = 0

   
        for i, c in enumerate(master_columns):
            if "Hi-Hat" in c or "46" in c:
                if step % 2 == 1 and random.random() < 0.25:
                    next_drums[0, i] = 0

   
        phase = (context.shape[1] % 16) / 16.0
        beat_emb_next = torch.tensor(
            [[np.sin(2*np.pi*phase), np.cos(2*np.pi*phase)]],
            device=device, dtype=next_drums.dtype
        )

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

def main():
    data_folder = "RockBinary_Dataset"
    selected_folder = "selected_files"
    model_path = "models/drum_rnn.pt"
    out_folder = "generated_samples"
    os.makedirs(selected_folder, exist_ok=True)
    os.makedirs(out_folder, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    k = seqs[0].shape[1]+2
    model = DrumRNN(input_dim=k, hidden_dim=128, num_layers=2, dropout=0.3, bidirectional=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    best_val = float("inf")
    threshold = 0.1

    for epoch in range(1, 11):
        tr_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_metrics, probs, all_true = validate(model, val_loader, device, threshold)
        print(f"Epoch {epoch}: train_bce={tr_loss:.4f} val_bce={val_loss:.4f} "
              f"micro_f1={val_metrics['micro_f1']:.4f} macro_f1={val_metrics['macro_f1']:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            save_model(model_path, model, optimizer, epoch=epoch,
                       meta={"val_loss": val_loss, "metrics": val_metrics})
            print("Saved best model to", model_path)

    # ---------- 2. Generate ----------
    if os.path.exists(out_folder):
        for f in os.listdir(out_folder):
           os.remove(os.path.join(out_folder, f))
        print(f"Cleared old files in {out_folder}")
    else:
         os.makedirs(out_folder)
    model = DrumRNN(input_dim=k) 
    load_model(model_path, model, map_location=device)
    model.to(device).eval() 
    print("\nModel loaded for generation.") 
    master_columns = np.load("master_columns.npy", allow_pickle=True).tolist() 
    all_csv = [f for f in sorted(os.listdir(data_folder)) if f.endswith(".csv")] 
    selected_files = random.sample(all_csv, min(5, len(all_csv))) 

    for i, fname in enumerate(selected_files, 1):
        path = os.path.join(data_folder, fname) 
        seq = load_csv_to_grid_with_master(path, master_columns) 
        if seq.shape[0] < 64: 
            continue 
        seed_drums = seq[:64].astype(np.float32) 
        df = generate_from_seed_groove(model, seed_drums, master_columns, device=device)
        out_path = os.path.join(out_folder, f"gen_sparse_{i}_{os.path.splitext(fname)[0]}.csv") 
        df.to_csv(out_path, index=False) 
        print(f"Generated sparser CSV saved: {out_path}")


if __name__ == "__main__":
    main()



