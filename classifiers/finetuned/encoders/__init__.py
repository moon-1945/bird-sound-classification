from .distilhubert import DistilHuBERTEncoder
from .birdaves import BirdAVESEncoder

ENCODER_REGISTRY = {
    "distilhubert": DistilHuBERTEncoder,
    "birdaves": BirdAVESEncoder,
}


def build_encoder(name: str, **kwargs):
    if name == "distilhubert":
        return DistilHuBERTEncoder()
    elif name == "birdaves":
        from .birdaves import BirdAVESEncoder
        return BirdAVESEncoder(**kwargs)  # passes model_name if provided
    else:
        raise ValueError(f"Unknown encoder '{name}'. Choose from: ['distilhubert', 'birdaves']")