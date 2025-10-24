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



if __name__ == "__main__":
    main()




