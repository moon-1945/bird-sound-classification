import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from classifiers.custom.models.bird_res_cnn import BirdResCNN
from data_utils import build_label_mapping
from waveform_dataset import WaveformDataset, load_splits


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    with torch.set_grad_enabled(train):
        for waveforms, labels in tqdm(loader, leave=False):
            waveforms, labels = waveforms.to(device), labels.to(device)
            logits = model(waveforms)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * waveforms.size(0)
            all_preds.extend(logits.argmax(-1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None,
                         help="Path to filtered_inat_sounds/. Defaults to <project_root>/filtered_inat_sounds "
                              "regardless of which directory you run this script from.")
    parser.add_argument("--sr", type=int, default=32000)
    parser.add_argument("--duration_s", type=float, default=5.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=150,
                         help="Upper bound -- early stopping will likely end the run sooner.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--patience", type=int, default=15,
                         help="Stop after this many epochs with no val_f1 improvement.")
    parser.add_argument("--num_workers", type=int, default=0,
                         help="0 on Windows -- see data_utils.decode_and_resample.")
    parser.add_argument("--out_dir", default="checkpoints_cnn")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_dict = load_splits(args.data_dir) if args.data_dir else load_splits()
    label2id, id2label = build_label_mapping(dataset_dict, "order")
    num_classes = len(label2id)
    print(f"Classes ({num_classes}): {label2id}")

    train_ds = WaveformDataset(dataset_dict["train"], label2id, target_sr=args.sr,
                                duration_s=args.duration_s, train=True)
    val_ds = WaveformDataset(dataset_dict["validation"], label2id, target_sr=args.sr,
                              duration_s=args.duration_s, train=False)
    test_ds = WaveformDataset(dataset_dict["test"], label2id, target_sr=args.sr,
                               duration_s=args.duration_s, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    model = BirdResCNN(num_classes=num_classes, sr=args.sr, base_ch=args.base_ch,
                     dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_f1 = 0.0
    best_epoch = 0
    epochs_since_improvement = 0
    best_ckpt_path = os.path.join(args.out_dir, "best_cnn.pt")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1 = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc, val_f1 = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()

        print(f"Epoch {epoch:02d} | train loss {train_loss:.4f} acc {train_acc:.4f} f1 {train_f1:.4f} "
              f"| val loss {val_loss:.4f} acc {val_acc:.4f} f1 {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_since_improvement = 0
            torch.save({"model": model.state_dict(), "label2id": label2id}, best_ckpt_path)
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (no val_f1 improvement for "
                      f"{args.patience} epochs, best was {best_val_f1:.4f} at epoch {best_epoch})")
                break

    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_loss, test_acc, test_f1 = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
    print(f"\nBest checkpoint: epoch {best_epoch}, val_f1 {best_val_f1:.4f}")
    print(f"TEST | loss {test_loss:.4f} acc {test_acc:.4f} macro-f1 {test_f1:.4f}")


if __name__ == "__main__":
    main()