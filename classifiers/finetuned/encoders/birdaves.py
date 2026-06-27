import numpy as np
import torch
import torch.nn as nn


class BirdAVESEncoder(nn.Module):
    HIDDEN = 768
    # DEFAULT_MODEL = "esp_aves2_naturelm_audio_v1_beats"
    DEFAULT_MODEL = "esp_aves2_sl_beats_bio"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        super().__init__()
        try:
            from avex import load_model, list_models
        except ImportError:
            raise ImportError(
                "avex is not installed.\n"
                "Run: pip install avex\n"
                "See: https://github.com/earthspecies/avex"
            )

        available = list(list_models().keys())
        if model_name not in available:
            raise ValueError(
                f"Model '{model_name}' not found in avex registry.\n"
                f"Available models: {available}\n"
            )

        self.model_name = model_name
        self.encoder = load_model(model_name, return_features_only=True)

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad_(True)

    def preprocess(self, waveforms_np: np.ndarray) -> dict:
        return {"input_values": torch.from_numpy(waveforms_np)}

    def forward(self, input_values, attention_mask=None):
        hidden = self.encoder(input_values)   # [B, T_frames, 768]
        return hidden.mean(1)                 # [B, 768]