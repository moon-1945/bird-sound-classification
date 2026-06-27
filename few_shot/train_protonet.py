"""
few_shot/train_protonet.py

Prototypical Network training on bird episodes.
Two modes controlled by --mode flag:

    head_only:  freeze CNN encoder, train a small projection head only
                [D → 256] on top of frozen get_features() output.
                Fast, safe, good first experiment.

    full:       train entire CNN end-to-end with episodic loss.
                Slower, risks forgetting, but can reshape embedding space.

Base classes:  10 bird orders (all except Strigiformes + Caprimulgiformes)
Val classes:   Strigiformes + Caprimulgiformes (held out, never in training episodes)

Usage:
    python few_shot/train_protonet.py --mode head_only --cnn_ckpt checkpoints_cnn/best_cnn.pt
    python few_shot/train_protonet.py --mode full      --cnn_ckpt checkpoints_cnn/best_cnn.pt
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from data_utils import build_label_mapping
from waveform_dataset import WaveformDataset, load_splits
from few_shot.episode_sampler import EpisodeSampler


VAL_ORDERS = {"Strigiformes", "Caprimulgiformes", "Psittaciformes", "Gruiformes"}


class ProjectionHead(nn.Module):
    """
    Small MLP projection from encoder dim to a lower-dimensional
    metric space. Only this module is trained in head_only mode.

    Input:  [B, in_dim]  — L2-normalised encoder output
    Output: [B, out_dim] — L2-normalised projected embedding
    """

    def __init__(self, in_dim: int, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim // 2, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=-1)


class ProtoNetModel(nn.Module):
    """
    Wraps a CNN encoder and optional projection head into a single
    embed() call used by the training loop.

    head_only mode:  encoder frozen, only head parameters trained
    full mode:       entire model trained, no head (embed directly from encoder)
    """

    def __init__(self, cnn: nn.Module, mode: str, encoder_dim: int,
                 proj_dim: int = 256, train: bool = True):
        super().__init__()
        self.cnn = cnn
        self.mode = mode

        if mode == "head_only":
            # freeze encoder completely
            for p in self.cnn.parameters():
                p.requires_grad_(False)
            self.head = ProjectionHead(encoder_dim, proj_dim)
        else:
            # full fine-tune — no projection head, embed directly
            for p in self.cnn.parameters():
                p.requires_grad_(train)
            self.head = None

    def embed(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform: [B, 1, T] → [B, D] L2-normalised embedding"""
        if self.mode == "head_only":
            with torch.no_grad():
                feats = self.cnn.get_features(waveform)   # frozen
            return self.head(feats)
        else:
            return self.cnn.get_features(waveform)        # trainable


def prototypical_loss(
    support: torch.Tensor,
    query: torch.Tensor,
    model: ProtoNetModel,
) -> tuple[torch.Tensor, float]:
    """
    support: [N, K, 1, T]
    query:   [N, Q, 1, T]

    Returns (loss, episode_accuracy).

    The loss is cross-entropy over negative squared Euclidean distances
    to prototypes — exactly the Prototypical Networks paper formulation.
    """
    N, K, C, T = support.shape
    Q = query.shape[1]

    # embed support → prototypes
    flat_support = support.reshape(N * K, C, T)
    support_emb = model.embed(flat_support)               # [N*K, D]
    support_emb = support_emb.reshape(N, K, -1)           # [N, K, D]
    prototypes = support_emb.mean(dim=1)                  # [N, D]
    prototypes = F.normalize(prototypes, p=2, dim=-1)

    # embed query
    flat_query = query.reshape(N * Q, C, T)
    query_emb = model.embed(flat_query)                   # [N*Q, D]

    # negative squared euclidean distance → logits
    # [N*Q, D] vs [N, D] → [N*Q, N]
    dists = torch.cdist(query_emb, prototypes).pow(2)
    logits = -dists                                       # higher = closer = more likely

    # ground truth: class 0 for first Q queries, class 1 for next Q, etc.
    gt = torch.arange(N, device=support.device).repeat_interleave(Q)

    loss = F.cross_entropy(logits, gt)
    acc = (logits.argmax(dim=-1) == gt).float().mean().item()
    return loss, acc


def run_episodes(
    model: ProtoNetModel,
    sampler: EpisodeSampler,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool,
) -> tuple[float, float]:
    model.train() if train else model.eval()
    total_loss, total_acc = 0.0, 0.0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for support, query, _ in tqdm(sampler, leave=False):
            support = support.to(device)
            query   = query.to(device)

            loss, acc = prototypical_loss(support, query, model)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            total_acc  += acc
            del support, query

    n = len(sampler)
    return total_loss / n, total_acc / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["head_only", "full"],
                        default="head_only")

    parser.add_argument("--cnn_ckpt",
                        default=r"..\classifiers\custom\checkpoints_cnn\best_cnn_base_ch=32_res_no_se_block.pt",
                        help="Path to pretrained CNN checkpoint.")
    parser.add_argument("--cnn_type", choices=["conv", "res"], default="res",
                        help="conv=BirdCNN  res=BirdResCNN")
    parser.add_argument("--with_se", default=False)

    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--proj_dim", type=int, default=512,
                        help="Projection head output dim (head_only mode only).")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--sr", type=int, default=32000)
    parser.add_argument("--duration_s", type=float, default=5.0)
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--k_shot", type=int, default=4)
    parser.add_argument("--n_query", type=int, default=8)
    parser.add_argument("--n_train_episodes", type=int, default=10)
    parser.add_argument("--n_val_episodes", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="checkpoints_protonet")
    args = parser.parse_args()

    # reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Mode: {args.mode}")

    # ── load CNN ──────────────────────────────────────────────────────────
    if args.cnn_type == "res":
        from classifiers.custom.models.bird_res_cnn import BirdResCNN
        cnn = BirdResCNN(
            num_classes=12, in_ch=1, base_ch=args.base_ch,
            sr=args.sr, with_se=args.with_se,
        )
    else:
        from classifiers.custom.models.bird_cnn import BirdCNN
        cnn = BirdCNN(num_classes=12, in_ch=1, base_ch=args.base_ch, sr=args.sr)

    ckpt = torch.load(args.cnn_ckpt, map_location=device)
    cnn.load_state_dict(ckpt["model"])
    cnn.to(device)

    # resolve encoder dim from a dummy forward
    with torch.no_grad():
        dummy = torch.zeros(1, 1, int(args.sr * args.duration_s), device=device)
        encoder_dim = cnn.get_features(dummy).shape[-1]
    print(f"Encoder dim: {encoder_dim}")

    # ── build ProtoNet model ──────────────────────────────────────────────
    model = ProtoNetModel(
        cnn=cnn,
        mode=args.mode,
        encoder_dim=encoder_dim,
        proj_dim=args.proj_dim,
    ).to(device)

    # only train parameters that require grad
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")

    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # ── build datasets ────────────────────────────────────────────────────
    dataset_dict = load_splits(args.data_dir) if args.data_dir else load_splits()
    label2id, id2label = build_label_mapping(dataset_dict, "order")

    base_label2id = {k: v for k, v in label2id.items() if k not in VAL_ORDERS}
    val_label2id = {k: v for k, v in label2id.items() if k in VAL_ORDERS}

    print(f"Base orders ({len(base_label2id)}): {list(base_label2id.keys())}")
    print(f"Val  orders ({len(val_label2id)}):  {list(val_label2id.keys())}")

    def make_split_dataset(split_name: str, lbl2id: dict, is_train: bool) -> WaveformDataset:
        allowed_orders = set(lbl2id.keys())
        hf_split = dataset_dict[split_name].filter(
            lambda row: row["order"] in allowed_orders
        )
        return WaveformDataset(
            hf_split, lbl2id,
            target_sr=args.sr, duration_s=args.duration_s, train=is_train,
        )

    train_ds = make_split_dataset("train",      base_label2id, is_train=True)
    val_ds   = make_split_dataset("validation", val_label2id,  is_train=False)

    train_sampler = EpisodeSampler(
        dataset=train_ds,
        n_way=min(args.n_way, len(base_label2id)),
        k_shot=args.k_shot,
        n_query=args.n_query,
        n_episodes=args.n_train_episodes,
        seed=None,
    )
    val_sampler = EpisodeSampler(
        dataset=val_ds,
        n_way=len(val_label2id),
        k_shot=args.k_shot,
        n_query=args.n_query,
        n_episodes=args.n_val_episodes,
        seed=args.seed,
    )

    # ── training loop ─────────────────────────────────────────────────────
    se_tag = ("se" if args.with_se else "nose") if args.cnn_type == "res" else ""
    ckpt_name = (
        f"protonet_{args.cnn_type}_ch{args.base_ch}_{se_tag}_{args.mode}"
        f"_N{args.n_way}K{args.k_shot}Q{args.n_query}.pt"
    )

    best_ckpt_path = os.path.join(args.out_dir, ckpt_name)
    best_val_acc, epochs_no_improve = 0.0, 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_episodes(
            model, train_sampler, optimizer, device, train=True
        )
        val_loss, val_acc = run_episodes(
            model, val_sampler, optimizer, device, train=False
        )
        scheduler.step()

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save({
                "model": model.state_dict(),
                "mode": args.mode,
                "encoder_dim": encoder_dim,
                "proj_dim": args.proj_dim,
                "cnn_type": args.cnn_type,
                "base_ch": args.base_ch,
                "label2id": label2id,
            }, best_ckpt_path)
            print(f"  ✓ saved best checkpoint (val_acc {best_val_acc:.3f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(
                    f"\nEarly stopping at epoch {epoch} "
                    f"(best val_acc {best_val_acc:.3f})"
                )
                break

    print(f"\nDone. Best val acc: {best_val_acc:.3f}")
    print(f"Checkpoint: {best_ckpt_path}")


if __name__ == "__main__":
    main()