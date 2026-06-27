"""
few_shot/mammal_dataset.py

Loads the mammal sound dataset from a flat folder structure:
    mammals_root/
        lion/        (50 .wav files)
        wolf/
        elephant/
        ...

Returns (waveform_tensor [1, T], label_id) matching WaveformDataset's
interface exactly, so the same episode sampler and eval code works for both.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from data_utils import decode_and_resample, _to_mono, _fix_length


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def find_audio_files(folder: str) -> list[str]:
    """Return sorted list of audio file paths inside folder."""
    files = [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS
    ]
    return files


def build_mammal_label_mapping(root: str) -> tuple[dict, dict]:
    """
    Scan root for subfolders — each subfolder name is a class label.
    Returns label2id and id2label sorted alphabetically for reproducibility.
    """
    classes = sorted([
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ])
    if not classes:
        raise ValueError(f"No subfolders found in {root}. "
                         f"Expected one subfolder per mammal class.")
    label2id = {cls: i for i, cls in enumerate(classes)}
    id2label = {i: cls for cls, i in label2id.items()}
    return label2id, id2label


class MammalDataset(Dataset):
    """
    Loads mammal sounds from a folder-per-class structure.

    Each item is (waveform_tensor [1, T], label_id) — identical output
    shape to WaveformDataset so the same episode sampler and collate_fn
    work for both birds and mammals without any changes.

    Audio is read from disk on __getitem__ (lazy) via soundfile + torchaudio
    resample, using the same decode_and_resample() as the bird pipeline so
    any fixes there apply here automatically.

    Args:
        root:        path to mammals_root/ containing one subfolder per class
        target_sr:   resample all clips to this rate (32000 for CNN,
                     16000 for transformer encoders)
        duration_s:  fixed clip length in seconds — clips longer than this
                     are cropped, shorter ones are zero-padded
        train:       if True, random crop for clips longer than duration_s;
                     if False, deterministic center crop (use False for eval)
        label2id:    optional pre-built mapping — pass the bird label2id to
                     share a unified label space, or None to build from
                     the mammal folder names only (default)
    """

    def __init__(
        self,
        root: str,
        target_sr: int,
        duration_s: float = 5.0,
        train: bool = False,
        label2id: dict = None,
    ):
        self.root = root
        self.target_sr = target_sr
        self.target_len = int(target_sr * duration_s)
        self.train = train

        if label2id is not None:
            self.label2id = label2id
            self.id2label = {v: k for k, v in label2id.items()}
        else:
            self.label2id, self.id2label = build_mammal_label_mapping(root)

        # Build flat index: list of (file_path, label_id)
        self.samples: list[tuple[str, int]] = []
        missing_classes = []

        for cls, label_id in sorted(self.label2id.items(), key=lambda x: x[1]):
            cls_dir = os.path.join(root, cls)
            if not os.path.isdir(cls_dir):
                missing_classes.append(cls)
                continue
            files = find_audio_files(cls_dir)
            if not files:
                missing_classes.append(cls)
                continue
            for f in files:
                self.samples.append((f, label_id))

        if missing_classes:
            print(f"[MammalDataset] WARNING: no audio found for classes: {missing_classes}")

        # Index for fast per-class lookup (used by episode sampler)
        self.class_to_indices: dict[int, list[int]] = {}
        for idx, (_, label_id) in enumerate(self.samples):
            self.class_to_indices.setdefault(label_id, []).append(idx)

        print(f"[MammalDataset] Loaded {len(self.samples)} clips "
              f"across {len(self.class_to_indices)} classes from {root}")
        for label_id, indices in sorted(self.class_to_indices.items()):
            print(f"  {self.id2label[label_id]:20s}: {len(indices)} clips")

        print("[MammalDataset] pre-caching audio...")
        self._cache: list[np.ndarray] = []
        for path, label_id in self.samples:
            with open(path, "rb") as f:
                audio_bytes = f.read()
            waveform = decode_and_resample(
                {"bytes": audio_bytes, "path": path}, self.target_sr
            )
            if waveform.ndim > 1:
                waveform = _to_mono(waveform)
            self._cache.append(waveform)  # store full-length numpy array

        total_mb = sum(w.nbytes for w in self._cache) / 1e6
        print(f"[MammalDataset] cached {len(self._cache)} clips, {total_mb:.1f} MB")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        _, label_id = self.samples[idx]
        waveform = self._cache[idx]  # np.ndarray, full length

        if self.train and len(waveform) > self.target_len:
            start = np.random.randint(0, len(waveform) - self.target_len + 1)
            waveform = waveform[start: start + self.target_len]
        else:
            waveform = _fix_length(waveform, self.target_len)

        return torch.from_numpy(waveform).unsqueeze(0), label_id

    @property
    def classes(self) -> list[str]:
        return [self.id2label[i] for i in sorted(self.id2label)]

    @property
    def num_classes(self) -> int:
        return len(self.label2id)