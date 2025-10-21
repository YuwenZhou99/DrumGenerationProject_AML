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
def generate_from_seed(model, seed, steps=1024, device="cpu", threshold=0.1, sample=False):
    """
    Each step appends the generated next_step to the context and uses the entire context as the next input
    sample=False: probs>threshold
    sample=True: torch.bernoulli(use Bernoulli sampling)
    shape: (L0 + steps, k)
    """
    model.eval()
    # context shape: (1, L0, k)
    context = torch.tensor(seed, dtype=torch.float32, device=device).unsqueeze(0)
    generated = [seed]

    for step in range(steps):
        logits, _ = model(context)
        probs = torch.sigmoid(logits[:, -1, :])

        if sample:
            next_step = torch.bernoulli(probs)  # random sampling
        else:
            next_step = (probs > threshold).float()

        # Append next_step to the end of context as the next input
        context = torch.cat([context, next_step.unsqueeze(1)], dim=1)  # 变为 (1, L0+1+..., k)
        generated.append(next_step.cpu().numpy())  # append (1,k) numpy

        # test steps
        # if step < 10:
        #     pm = float(probs.mean().cpu().item())
        #     pmin = float(probs.min().cpu().item())
        #     pmax = float(probs.max().cpu().item())
        #     print(f"[gen-grow] step={step} probs mean={pm:.4f} min={pmin:.4f} max={pmax:.4f}")

    return np.vstack([x if isinstance(x, np.ndarray) else x.squeeze(0) for x in generated])

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
    seqs = load_folder_as_sequences(data_folder, compress_window=4)
    print("Total sequences:", len(seqs))
    idx = np.random.permutation(len(seqs))
    split = int(len(seqs) * 0.7)
    train_seqs = [seqs[i] for i in idx[:split]]
    val_seqs = [seqs[i] for i in idx[split:]]

    train_ds = ShortDrumDataset(train_seqs)
    val_ds = ShortDrumDataset(val_seqs)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_pad)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_pad)

    k = seqs[0].shape[1]
    model = DrumRNN(input_dim=k, hidden_dim=128, num_layers=2, dropout=0.3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

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

    master_columns = build_master_columns(data_folder)
    all_csv = [f for f in sorted(os.listdir(data_folder)) if f.endswith(".csv")]
    selected_files = random.sample(all_csv, min(5, len(all_csv)))

    for i, fname in enumerate(selected_files, 1):
        path = os.path.join(data_folder, fname)
        seq = load_csv_to_grid_with_master(path, master_columns)
        if seq.shape[0] < 64:
            continue
        seed = seq[:64].astype(np.float32)
        gen = generate_from_seed(model, seed, steps=1024, device=device, threshold=0.2,sample = True)
        df = pd.DataFrame(gen.astype(int), columns=master_columns)
        out_path = os.path.join(out_folder, f"gen_{i}_{os.path.splitext(fname)[0]}.csv")
        df.to_csv(out_path, index=False)
        print(f"Generated CSV saved: {out_path}")


if __name__ == "__main__":
    main()


