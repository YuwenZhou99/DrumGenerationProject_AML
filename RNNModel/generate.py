import torch
import numpy as np
import pandas as pd

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
