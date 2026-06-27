import torch.nn as nn
import torchaudio


class LogMelSpectrogram(nn.Module):
    def __init__(self, sample_rate=32000, n_fft=1024, hop_length=320, n_mels=128):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, power=2.0,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)

    def forward(self, waveform):
        spec = self.mel(waveform)
        spec = self.to_db(spec)
        mean = spec.mean(dim=(-2, -1), keepdim=True)
        std = spec.std(dim=(-2, -1), keepdim=True) + 1e-6
        return (spec - mean) / std