"""
Collation logic that bridges data_utils.WaveformDataset
(returns torch.Tensor [1, T]) with the encoder preprocessors
(expect np.ndarray [B, T]).
"""

import numpy as np
import torch


def collate_fn(encoder, device):
    """
    Returns a collate function closed over encoder and device.
    WaveformDataset returns (Tensor[1, T], int) -- we squeeze the channel
    dim and convert to numpy so each encoder's preprocess() gets [B, T].
    """
    def _collate(batch):
        waveforms, labels = zip(*batch)
        waveforms_np = np.stack([w.squeeze(0).numpy() for w in waveforms])
        inputs = encoder.preprocess(waveforms_np)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels_t = torch.tensor(labels, dtype=torch.long, device=device)
        return inputs, labels_t
    return _collate