import numpy as np
import torch
import torch.nn as nn

SR = 16000
HF_NAME = "ntu-spml/distilhubert"


class DistilHuBERTEncoder(nn.Module):
    """
    Speech-pretrained HuBERT (ntu-spml/distilhubert).
    Input: raw waveform at 16 kHz.
    Output: mean-pooled last hidden state [B, 768].

    Domain note: pretrained on LibriSpeech (speech), not bird audio.
    High-frequency content above 8 kHz is lost when resampling to 16 kHz,
    which may discard discriminative call features. Compare against
    BirdAVESEncoder to measure how much domain mismatch costs.
    """
    HIDDEN = 768

    def __init__(self):
        super().__init__()
        from transformers import AutoModel, AutoFeatureExtractor
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(HF_NAME)
        self.encoder = AutoModel.from_pretrained(HF_NAME)

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad_(True)

    def preprocess(self, waveforms_np: np.ndarray) -> dict:
        """np.ndarray [B, T] -> dict of tensors for forward()."""
        inputs = self.feature_extractor(
            list(waveforms_np),
            sampling_rate=SR,
            return_tensors="pt",
            padding=True,
        )
        return inputs

    def forward(self, input_values, attention_mask=None):
        out = self.encoder(input_values=input_values, attention_mask=attention_mask)
        hidden = out.last_hidden_state          # [B, T_frames, 768]
        if attention_mask is not None:
            # feature extractor subsamples 320x; approximate the frame mask
            frame_mask = attention_mask[:, ::320][:, :hidden.size(1)]
            frame_mask = frame_mask.unsqueeze(-1).float()
            pooled = (hidden * frame_mask).sum(1) / frame_mask.sum(1).clamp(min=1)
        else:
            pooled = hidden.mean(1)
        return pooled                           # [B, 768]