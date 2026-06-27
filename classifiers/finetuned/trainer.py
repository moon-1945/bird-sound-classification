import os

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


def run_epoch(model, loader, criterion, optimizer=None, train=True):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    with torch.set_grad_enabled(train):
        for inputs, labels in tqdm(loader, leave=False):
            logits = model(**inputs)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            all_preds.extend(logits.argmax(-1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    n = len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return total_loss / n, acc, f1


def make_optimizer(model, lr_head, lr_encoder, weight_decay, frozen=True):
    """
    Frozen phase:   single param group, head only, lr_head.
    Unfrozen phase: two param groups with separate LRs for encoder vs head.
    """
    if frozen:
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr_head, weight_decay=weight_decay,
        )
    return torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": lr_encoder},
        {"params": model.head.parameters(),    "lr": lr_head},
    ], weight_decay=weight_decay)


def train(
    model, train_loader, val_loader,
    freeze_epochs, total_epochs, patience,
    lr_head, lr_encoder, weight_decay, label_smoothing,
    out_dir, model_name, label2id
):
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    best_ckpt = os.path.join(out_dir, f"best_{model_name}.pt")

    # Phase 1: frozen encoder
    optimizer = make_optimizer(model, lr_head, lr_encoder, weight_decay, frozen=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=freeze_epochs)

    best_val_f1, best_epoch, epochs_no_improve = 0.0, 0, 0

    for epoch in range(1, total_epochs + 1):

        # Transition to phase 2
        if epoch == freeze_epochs + 1:
            print(f"\n--- Unfreezing encoder at epoch {epoch} ---")
            model.encoder.unfreeze_encoder()
            optimizer = make_optimizer(model, lr_head, lr_encoder, weight_decay, frozen=False)
            remaining = total_epochs - freeze_epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining)

        phase = "frozen" if epoch <= freeze_epochs else "full"
        train_loss, train_acc, train_f1 = run_epoch(model, train_loader, criterion, optimizer, train=True)
        val_loss, val_acc, val_f1 = run_epoch(model, val_loader, criterion, train=False)
        scheduler.step()

        print(
            f"[{phase}] Epoch {epoch:03d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} f1 {train_f1:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} f1 {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1, best_epoch, epochs_no_improve = val_f1, epoch, 0
            torch.save({"model": model.state_dict(), "label2id": label2id}, best_ckpt)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f"\nEarly stopping at epoch {epoch} "
                    f"(best val_f1 {best_val_f1:.4f} at epoch {best_epoch})"
                )
                break

    return best_ckpt, best_epoch, best_val_f1