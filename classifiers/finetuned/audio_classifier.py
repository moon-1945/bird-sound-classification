import torch.nn as nn


class AudioClassifier(nn.Module):
    """Encoder + classification head. Encoder is swappable via build_encoder()."""

    def __init__(self, encoder, num_classes, dropout=0.3):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder.HIDDEN, encoder.HIDDEN // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(encoder.HIDDEN // 2, num_classes),
        )

    def forward(self, input_values, attention_mask=None):
        pooled = self.encoder(input_values, attention_mask)
        return self.head(pooled)