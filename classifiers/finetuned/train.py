"""
Fine-tune DistilHuBERT or BirdAVES on the filtered iNat bird-order dataset.

Usage:
    python train_hubert.py --model distilhubert
    python train_hubert.py --model birdaves       # requires: pip install avex
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader
from classifiers.finetuned.audio_classifier import AudioClassifier
from classifiers.finetuned.encoders import BirdAVESEncoder
from data_utils import build_label_mapping
from waveform_dataset import WaveformDataset, load_splits
from encoders import build_encoder
from dataset import collate_fn
from trainer import train, run_epoch

SR = 16000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["distilhubert", "birdaves"],
                        default="birdaves")
    parser.add_argument("--birdaves_model", default=BirdAVESEncoder.DEFAULT_MODEL,
                        help="avex model name, only used when --model birdaves")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--duration_s",    type=float, default=5.0)
    parser.add_argument("--batch_size",    type=int,   default=8)
    parser.add_argument("--freeze_epochs", type=int,   default=5)
    parser.add_argument("--epochs",        type=int,   default=40)
    parser.add_argument("--lr_head",       type=float, default=1e-3)
    parser.add_argument("--lr_encoder",    type=float, default=1e-5)
    parser.add_argument("--weight_decay",  type=float, default=1e-2)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--dropout",       type=float, default=0.3)
    parser.add_argument("--patience",      type=int,   default=10)
    parser.add_argument("--num_workers",   type=int,   default=0)
    parser.add_argument("--out_dir",       default="checkpoints")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Model: {args.model}")

    dataset_dict = load_splits(args.data_dir) if args.data_dir else load_splits()
    label2id, id2label = build_label_mapping(dataset_dict, "order")
    num_classes = len(label2id)
    print(f"Classes ({num_classes}): {list(label2id.keys())}")

    train_ds = WaveformDataset(dataset_dict["train"],      label2id, target_sr=SR,
                               duration_s=args.duration_s, train=True)
    val_ds   = WaveformDataset(dataset_dict["validation"], label2id, target_sr=SR,
                               duration_s=args.duration_s, train=False)
    test_ds  = WaveformDataset(dataset_dict["test"],       label2id, target_sr=SR,
                               duration_s=args.duration_s, train=False)

    encoder = build_encoder(args.model, model_name=args.birdaves_model).to(device)
    encoder.freeze_encoder()                    # start frozen

    model = AudioClassifier(encoder, num_classes, dropout=args.dropout).to(device)

    cfn = collate_fn(encoder, device)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=cfn)
    val_loader = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=cfn)
    test_loader = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=cfn)

    if args.model == "birdaves":
        model_name = f"birdaves_{args.birdaves_model or BirdAVESEncoder.DEFAULT_MODEL}"
    else:
        model_name = args.model

    best_ckpt, best_epoch, best_val_f1 = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        freeze_epochs=args.freeze_epochs,
        total_epochs=args.epochs,
        patience=args.patience,
        lr_head=args.lr_head,
        lr_encoder=args.lr_encoder,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        out_dir=args.out_dir,
        model_name=model_name,
        label2id=label2id
    )

    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_loss, test_acc, test_f1 = run_epoch(model, test_loader,
                                              criterion=torch.nn.CrossEntropyLoss(),
                                              train=False)
    print(f"\nBest checkpoint: epoch {best_epoch}, val_f1 {best_val_f1:.4f}")
    print(f"TEST | loss {test_loss:.4f} acc {test_acc:.4f} macro-f1 {test_f1:.4f}")


if __name__ == "__main__":
    main()