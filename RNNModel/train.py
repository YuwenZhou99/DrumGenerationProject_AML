import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import load_folder_as_sequences, ShortDrumDataset, collate_pad
from RNNModel import DrumRNN
from utils import save_model, f1_metrics_window

def masked_bce_with_logits(logits, targets, mask):
    """
    logits, targets: (B, L, k)
    mask: (B, L, 1)
    returns mean BCE over valid positions
    """
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')  # (B,L,k)
    mask3 = mask.expand_as(bce)
    return (bce * mask3).sum() / mask3.sum()

@torch.no_grad()
def validate(model, loader, device, threshold=0.3):
    model.eval()
    all_logits = []
    all_true = []
    total_loss = 0.0
    total_n = 0
    for x, y, lengths, mask in loader:
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)
        logits, _ = model(x)
        loss = masked_bce_with_logits(logits, y, mask)
        batch_n = mask.sum().item()
        total_loss += loss.item() * x.size(0)
        all_logits.append(logits.cpu().numpy())
        all_true.append(y.cpu().numpy())
        total_n += x.size(0)
    all_logits = np.concatenate(all_logits, axis=0)
    all_true = np.concatenate(all_true, axis=0)
    probs = 1.0 / (1.0 + np.exp(-all_logits))
    metrics = f1_metrics_window(all_true, probs,threshold=threshold, window=4)
    avg_loss = total_loss / total_n
    return avg_loss, metrics, probs, all_true

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, y, lengths, mask in loader:
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = masked_bce_with_logits(logits, y, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

def main():
    # adjust the parameters using the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="models/drum_rnn.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    print("Loading sequences from", args.data_folder)
    seqs = load_folder_as_sequences(args.data_folder)
    print("Total sequences:", len(seqs))
    # shuffle and split by sequence (file) level
    idx = np.random.permutation(len(seqs))
    split = int(len(seqs) * 0.7)
    train_seqs = [seqs[i] for i in idx[:split]]
    val_seqs = [seqs[i] for i in idx[split:]]
    print(f"Train seqs: {len(train_seqs)}  Val seqs: {len(val_seqs)}")

    train_ds = ShortDrumDataset(train_seqs)
    val_ds = ShortDrumDataset(val_seqs)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_pad)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pad)

    # create the model
    k = seqs[0].shape[1]
    model = DrumRNN(input_dim=k, hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # train and validate
    best_val = float('inf')
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_metrics, probs, all_true = validate(model, val_loader, device, threshold=args.threshold)
        print(f"Epoch {epoch}: train_bce={tr_loss:.4f} val_bce={val_loss:.4f} micro_f1={val_metrics['micro_f1']:.4f} macro_f1={val_metrics['macro_f1']:.4f}")

        best_thr = args.threshold
        best_f1 = val_metrics['micro_f1']
        # try grid thresholds
        for thr in np.linspace(0.1, 0.9, 17):
            m = f1_metrics_window(all_true, probs,threshold=thr, window=4)
            if m['micro_f1'] > best_f1:
                best_f1 = m['micro_f1']
                best_thr = thr
        print(f" Best val thr={best_thr:.2f} best_micro_f1={best_f1:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            save_model(args.save_path, model, optimizer=optimizer, epoch=epoch, meta={"val_loss": val_loss, "metrics": val_metrics})
            print("Saved best model to", args.save_path)

if __name__ == "__main__":
    main()
