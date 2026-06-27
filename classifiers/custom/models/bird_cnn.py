import torch
import torch.nn as nn
from classifiers.custom.models.log_mel_spectrogram import LogMelSpectrogram


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class BirdCNN(nn.Module):
    def __init__(self, num_classes, in_ch=1, base_ch=32, sr=32000, dropout=0.3):
        super().__init__()
        self.frontend = LogMelSpectrogram(sample_rate=sr)
        self.features = nn.Sequential(
            ConvBlock(in_ch, base_ch),
            ConvBlock(base_ch, base_ch * 2),
            ConvBlock(base_ch * 2, base_ch * 4),
            ConvBlock(base_ch * 4, base_ch * 8),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_ch * 8, base_ch * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(base_ch * 4, num_classes),
        )

    def get_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Returns L2-normalised embedding [B, base_ch*8] stopping before
        the classifier head. Used for few-shot nearest-prototype evaluation.
        """
        spec = self.frontend(waveform)
        feats = self.features(spec)
        pooled = self.pool(feats).flatten(1)   # [B, base_ch*8]
        return nn.functional.normalize(pooled, p=2, dim=-1)


    def forward(self, waveform):
        spec = self.frontend(waveform)
        feats = self.features(spec)
        pooled = self.pool(feats)
        return self.classifier(pooled)