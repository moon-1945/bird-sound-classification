import torch
import torch.nn as nn
from classifiers.custom.models.log_mel_spectrogram import LogMelSpectrogram


class SEBlock(nn.Module):
    """Squeeze-and-excitation: lets the network reweight feature channels
    by how useful they are for the current clip, rather than treating
    every channel equally."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class ResConvBlock(nn.Module):
    """Two conv layers with a residual skip connection and channel
    attention, then downsample."""

    def __init__(self, in_ch, out_ch, with_se=True):
        super().__init__()
        self.with_se = with_se
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if with_se:
            self.se = SEBlock(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

        self.skip = (
            nn.Identity() if in_ch == out_ch
            else nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        )

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.with_se:
            out = self.se(out)
        out = self.relu(out + identity)
        return self.pool(out)


class BirdResCNN(nn.Module):
    """CNN over log-mel spectrograms: residual conv stages with channel
    attention, avg+max pooled head."""

    def __init__(self, num_classes, in_ch=1, base_ch=32, sr=32000, dropout=0.4, with_se=True):
        super().__init__()
        self.frontend = LogMelSpectrogram(sample_rate=sr)
        self.features = nn.Sequential(
            ResConvBlock(in_ch, base_ch, with_se=with_se),
            ResConvBlock(base_ch, base_ch * 2, with_se=with_se),
            ResConvBlock(base_ch * 2, base_ch * 4, with_se=with_se),
            ResConvBlock(base_ch * 4, base_ch * 8, with_se=with_se),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_ch * 8 * 2, base_ch * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(base_ch * 4, num_classes),
        )

    def get_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Returns L2-normalised embedding [B, base_ch*8*2] stopping before
        the classifier head. avg+max pool concat matches the forward() pooling
        exactly so the embedding space is consistent with what the classifier
        was trained on.
        """
        spec = self.frontend(waveform)
        feats = self.features(spec)
        avg = self.avg_pool(feats).flatten(1)
        mx = self.max_pool(feats).flatten(1)
        pooled = torch.cat([avg, mx], dim=1)  # [B, base_ch*8*2]
        return nn.functional.normalize(pooled, p=2, dim=-1)

    def forward(self, waveform):
        spec = self.frontend(waveform)
        feats = self.features(spec)
        pooled = torch.cat([self.avg_pool(feats), self.max_pool(feats)], dim=1)
        return self.classifier(pooled)