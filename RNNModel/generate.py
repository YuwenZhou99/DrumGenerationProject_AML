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

    # select 10 random files as seeds
    all_csv = [f for f in sorted(os.listdir(data_folder)) if f.endswith(".csv")]
    if len(all_csv) == 0:
        raise RuntimeError("No CSV in data_folder")
    selected_files = random.sample(all_csv, min(10, len(all_csv)))
    print("\nSelected original files:")
    for f in selected_files:
        print(" -", f)
    for f in selected_files:
        shutil.copy2(os.path.join(data_folder, f), os.path.join(selected_folder, f))

    # load checkpoint first to discover expected k
    ck = torch.load(model_path, map_location="cpu")
    if "model_state" in ck and "fc.weight" in ck["model_state"]:
        k_expected = ck["model_state"]["fc.weight"].shape[0]
    else:
        meta = ck.get("meta", {})
        master_cols = meta.get("master_columns", None)
        k_expected = len(master_cols)

    # load the model
    model = DrumRNN(input_dim=k_expected)
    load_model(model_path, model, map_location=device)
    model.to(device)
    model.eval()
    print("Model loaded and ready.")

    # build master columns from the FULL dataset folder (to align columns as in training)
    master_columns = build_master_columns(data_folder)

    # for each selected file, load using master and generate
    for i, fname in enumerate(selected_files, 1):
        path = os.path.join(data_folder, fname)
        seq = load_csv_to_grid_with_master(path, master_columns)
        if seq.shape[0] < 64:
            print(f"Skipping {fname} because too short ({seq.shape[0]} steps)")
            continue

        seed_len = 64
        seed = seq[:seed_len].astype(np.float32)
        print(f"\n[{i}] source={fname}, seed shape={seed.shape}")

        gen = generate_from_seed(model, seed, steps=1024, device=device, threshold=0.5)
        print(f"Generated shape: {gen.shape}")

        # save CSV with original-like columns
        df = pd.DataFrame(gen.astype(int), columns=master_columns)
        base_name = os.path.splitext(fname)[0]
        out_path = os.path.join(out_folder, f"gen_{i}_{base_name}_seed{seed_len}.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved generated CSV: {out_path}")

if __name__ == "__main__":
    main()
