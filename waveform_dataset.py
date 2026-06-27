import os

import numpy as np
import torch
from datasets import DatasetDict, load_from_disk, Audio
from torch.utils.data import Dataset

from data_utils import decode_and_resample, _to_mono, _fix_length

# This file lives in <project_root>/classifiers/data_utils.py, and
# filtered_inat_sounds/ lives in <project_root>/. Resolving the default path
# from __file__ (rather than a bare relative string) means it's found
# correctly whether you run a script from the project root, from inside
# classifiers/, or from anywhere else -- it no longer depends on cwd.
_DEFAULT_DATA_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "", "filtered_inat_sounds")
)


def load_splits(path: str = _DEFAULT_DATA_DIR) -> DatasetDict:
    dataset_dict = load_from_disk(path)
    dataset_dict = dataset_dict.cast_column("audio", Audio(decode=False))
    return dataset_dict


class WaveformDataset(Dataset):
    def __init__(self, hf_split, label2id, target_sr, label_col="order",
                 duration_s=5.0, train=False):
        self.ds = hf_split
        self.label2id = label2id
        self.label_col = label_col
        self.train = train
        self.target_sr = target_sr
        self.target_len = int(target_sr * duration_s)

        # build class_to_indices by scanning labels once at init
        # cheap since it only reads the label column, not audio bytes
        self.class_to_indices: dict[int, list[int]] = {}
        for idx in range(len(self.ds)):
            label_id = self.label2id[self.ds[idx][self.label_col]]
            self.class_to_indices.setdefault(label_id, []).append(idx)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        waveform = decode_and_resample(sample["audio"], self.target_sr)

        if waveform.ndim > 1:
            waveform = _to_mono(waveform)

        if self.train and len(waveform) > self.target_len:
            start = np.random.randint(0, len(waveform) - self.target_len + 1)
            waveform = waveform[start:start + self.target_len]
        else:
            waveform = _fix_length(waveform, self.target_len)

        label_id = self.label2id[sample[self.label_col]]
        return torch.from_numpy(waveform).unsqueeze(0), label_id



