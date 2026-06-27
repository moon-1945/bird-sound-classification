import csv
import json
import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
from classifiers.custom.models.bird_cnn import BirdCNN
from classifiers.custom.models.bird_res_cnn import BirdResCNN
from data_utils import build_label_mapping
from waveform_dataset import WaveformDataset, load_splits
from classifiers.finetuned.audio_classifier import AudioClassifier
from classifiers.finetuned.encoders import build_encoder
from classifiers.finetuned.dataset import collate_fn

NUM_CLASSES = 12

# Each entry holds an already-constructed model, its checkpoint path, and the
# sample rate it was trained with. Custom CNNs consume raw waveforms directly;
# finetuned models wrap an encoder and are detected via `model.encoder` to pick
# the right collate/forward path automatically.
MODEL_CONFIGS = [
    {
        "path": "custom/checkpoints_cnn/best_cnn_base_ch=32_conv.pt",
        "model": BirdCNN(num_classes=NUM_CLASSES, in_ch=1, base_ch=32, sr=32000),
        "sr": 32000,
    },
    {
        "path": "custom/checkpoints_cnn/best_cnn_base_ch=32_res.pt",
        "model": BirdResCNN(num_classes=NUM_CLASSES, in_ch=1, base_ch=32, sr=32000),
        "sr": 32000,
    },
    {
        "path": "custom/checkpoints_cnn/best_cnn_base_ch=32_res_no_se_block.pt",
        "model": BirdResCNN(num_classes=NUM_CLASSES, in_ch=1, base_ch=32, sr=32000, with_se=False),
        "sr": 32000,
    },
    {
        "path": "finetuned/checkpoints/best_distilhubert.pt",
        "model": AudioClassifier(build_encoder("distilhubert"), num_classes=NUM_CLASSES),
        "sr": 16000,
    },
    {
        "path": "finetuned/checkpoints/best_birdaves_esp_aves2_naturelm_audio_v1_beats.pt",
        "model": AudioClassifier(
            build_encoder("birdaves", model_name="esp_aves2_naturelm_audio_v1_beats"),
            num_classes=NUM_CLASSES,
        ),
        "sr": 16000,
    },
    {
        "path": "finetuned/checkpoints/best_birdaves_esp_aves2_sl_beats_bio.pt",
        "model": AudioClassifier(
            build_encoder("birdaves", model_name="esp_aves2_sl_beats_bio"),
            num_classes=NUM_CLASSES,
        ),
        "sr": 16000,
    },
]

DATA_DIR = None
DURATION_S = 5.0
BATCH_SIZE = 16
NUM_WORKERS = 0
SAVE_REPORT_DIR = "test_eval_reports"


def arch_name(model):
    """Human-readable architecture name for the summary table."""
    encoder = getattr(model, "encoder", None)
    if encoder is not None:
        return encoder.__class__.__name__
    return model.__class__.__name__


@torch.no_grad()
def evaluate(model, loader, device, id2label):
    all_preds, all_labels = [], []
    for inputs, labels in tqdm(loader, desc="evaluating", leave=False):
        if isinstance(inputs, dict):
            # finetuned models: collate_fn already moved tensors to device
            logits = model(**inputs)
        else:
            # custom CNNs: raw waveform tensor, move to device here
            logits = model(inputs.to(device))
        all_preds.extend(logits.argmax(-1).cpu().tolist())
        all_labels.extend(labels.tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    target_names = [id2label[i] for i in range(len(id2label))]
    report = classification_report(
        all_labels, all_preds, target_names=target_names, zero_division=0
    )
    return acc, f1, report, all_preds, all_labels


def format_summary_table(summary):
    lines = [f"{'checkpoint':75s} {'arch':45s} {'acc':>8s} {'macro-f1':>10s}"]
    for row in summary:
        lines.append(f"{row['checkpoint']:75s} {row['arch']:45s} {row['acc']:8.4f} {row['f1']:10.4f}")
    return "\n".join(lines)


def write_summary_report(summary, save_dir):
    txt_path = os.path.join(save_dir, "summary.txt")
    with open(txt_path, "w") as f:
        f.write(format_summary_table(summary) + "\n")

    csv_path = os.path.join(save_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["checkpoint", "arch", "acc", "f1"])
        writer.writeheader()
        writer.writerows(summary)

    return txt_path, csv_path


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_dict = load_splits(DATA_DIR) if DATA_DIR else load_splits()
    label2id, id2label = build_label_mapping(dataset_dict, "order")
    num_classes = len(label2id)
    print(f"Classes ({num_classes}): {label2id}")

    if SAVE_REPORT_DIR:
        os.makedirs(SAVE_REPORT_DIR, exist_ok=True)

    summary = []
    for cfg in MODEL_CONFIGS:
        ckpt_path = cfg["path"]
        model = cfg["model"]
        sr = cfg["sr"]
        arch = arch_name(model)

        print(f"\n=== {ckpt_path} ({arch}) ===")
        if not os.path.exists(ckpt_path):
            print(f"  SKIPPED — checkpoint not found: {ckpt_path}")
            continue

        ckpt = torch.load(ckpt_path, map_location=device)

        model.load_state_dict(ckpt["model"])
        model.to(device).eval()

        test_ds = WaveformDataset(
            dataset_dict["test"], label2id,
            target_sr=sr, duration_s=DURATION_S, train=False,
        )

        encoder = getattr(model, "encoder", None)
        if encoder is not None:
            test_loader = DataLoader(
                test_ds, batch_size=BATCH_SIZE, shuffle=False,
                num_workers=NUM_WORKERS,
                collate_fn=collate_fn(encoder, device),
            )
        else:
            test_loader = DataLoader(
                test_ds, batch_size=BATCH_SIZE, shuffle=False,
                num_workers=NUM_WORKERS, pin_memory=True,
            )

        acc, f1, report, preds, labels = evaluate(model, test_loader, device, id2label)
        print(f"TEST acc {acc:.4f} | macro-f1 {f1:.4f}")
        print(report)

        summary.append({
            "checkpoint": ckpt_path,
            "arch": arch,
            "acc": acc,
            "f1": f1,
        })

        if SAVE_REPORT_DIR:
            name = os.path.splitext(os.path.basename(ckpt_path))[0]
            with open(os.path.join(SAVE_REPORT_DIR, f"{name}_report.txt"), "w") as f:
                f.write(f"Checkpoint: {ckpt_path}\nArchitecture: {arch}\n")
                f.write(f"Test acc: {acc:.4f}\nTest macro-f1: {f1:.4f}\n\n")
                f.write(report)
            with open(os.path.join(SAVE_REPORT_DIR, f"{name}_predictions.json"), "w") as f:
                json.dump({"preds": preds, "labels": labels, "id2label": id2label}, f)

    if summary:
        table = format_summary_table(summary)
        if len(summary) > 1:
            print("\n=== Summary ===")
            print(table)

        if SAVE_REPORT_DIR:
            txt_path, csv_path = write_summary_report(summary, SAVE_REPORT_DIR)
            print(f"\nSummary written to {txt_path} and {csv_path}")


if __name__ == "__main__":
    main()