import numpy as np
from datasets import DatasetDict
import io
import soundfile as sf
import torch
import torchaudio.functional as AF


def decode_and_resample(audio_dict: dict, target_sr: int) -> np.ndarray:
    waveform, native_sr = sf.read(
        io.BytesIO(audio_dict["bytes"]), dtype="float32", always_2d=False
    )

    if native_sr == target_sr:
        return waveform

    wf_t = torch.from_numpy(waveform)
    if wf_t.ndim == 2:
        wf_t = wf_t.T  # soundfile: (frames, channels) -> torchaudio wants (channels, frames)
    wf_t = AF.resample(wf_t, native_sr, target_sr)
    waveform = wf_t.numpy()
    if waveform.ndim == 2:
        waveform = waveform.T  # back to (frames, channels)
    return waveform


def build_label_mapping(dataset_dict: DatasetDict, label_col: str = "order"):
    """Build a consistent label2id / id2label mapping across all splits."""
    labels = set()
    for split in dataset_dict.values():
        labels.update(split.unique(label_col))
    labels = sorted(labels)
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label


def _to_mono(waveform: np.ndarray) -> np.ndarray:
    if waveform.ndim == 1:
        return waveform.astype(np.float32)
    # Channel count (1-2) is always far smaller than sample count, so the
    # smaller axis is the channel axis regardless of layout convention.
    channel_axis = 0 if waveform.shape[0] < waveform.shape[1] else 1
    return waveform.mean(axis=channel_axis).astype(np.float32)


def _fix_length(waveform: np.ndarray, target_len: int) -> np.ndarray:
    if len(waveform) >= target_len:
        start = (len(waveform) - target_len) // 2
        return waveform[start:start + target_len]
    pad = target_len - len(waveform)
    left = pad // 2
    right = pad - left
    return np.pad(waveform, (left, right), mode="constant")


