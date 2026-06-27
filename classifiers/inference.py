"""
Run inference on a single .wav file with one chosen trained checkpoint --
works for both custom CNN models and finetuned encoder-based models.

Usage:
    python predict_wav.py

Edit the CONFIG block below: point WAV_PATH at your clip and pick the
matching checkpoint + model object + sample rate (same pattern as
MODEL_CONFIGS in evaluate_checkpoints.py). The rest of the script auto-detects
which kind of model you picked and handles preprocessing/calling accordingly
-- nothing else needs to change between the two.

NOTE on preprocessing:
- Custom CNNs take a raw waveform tensor directly. This script loads/resamples
  /crops the waveform itself with torchaudio+soundfile rather than going
  through WaveformDataset, since the exact preprocessing in data_utils.py
  wasn't available here. The logic below (mono, resample to SR, center-crop
  /pad to DURATION_S) mirrors the obvious default -- if WaveformDataset does
  something different for eval, swap load_waveform() below for whatever
  internal helper it actually uses so train/eval preprocessing matches
  exactly. train.py's docstring mentions data_utils.decode_and_resample as a
  likely candidate to reuse.
- Finetuned models don't take a waveform tensor directly -- they need
  whatever encoder-specific `inputs` dict collate_fn(encoder, device) builds
  (e.g. feature-extractor output). Rather than approximate that, this script
  runs the *real* collate_fn on a single-item batch, so preprocessing is
  guaranteed to match training/eval exactly.
"""
import soundfile as sf
import torch
import torchaudio
from classifiers.custom.models.bird_cnn import BirdCNN
from classifiers.custom.models.bird_res_cnn import BirdResCNN
from classifiers.finetuned.audio_classifier import AudioClassifier
from classifiers.finetuned.encoders import build_encoder
from classifiers.finetuned.dataset import collate_fn


# --- pick ONE model/checkpoint/sr combo, matching MODEL_CONFIGS entries ---

WAV_PATH = "../example/goose.wav"

# Custom CNN example:
# CHECKPOINT_PATH = "custom/checkpoints_cnn/best_cnn_base_ch=32_res.pt"
# MODEL = BirdResCNN(num_classes=12, in_ch=1, base_ch=32, sr=32000)
# SR = 32000

# Finetuned example -- uncomment to use instead:
CHECKPOINT_PATH = "finetuned/checkpoints/best_distilhubert.pt"
MODEL = AudioClassifier(build_encoder("distilhubert"), num_classes=12)
SR = 16000

DURATION_S = 5.0
TOP_K = 3


def load_waveform(path, target_sr, duration_s):
    # soundfile reads wav/flac natively -- no FFmpeg/TorchCodec dependency,
    # which is what torchaudio.load() now requires by default and what was
    # failing to load on Windows.
    data, sr = sf.read(path, dtype="float32", always_2d=True)  # [samples, channels]
    waveform = torch.from_numpy(data.T)                         # [channels, samples]

    if waveform.shape[0] > 1:                      # downmix to mono
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    target_len = int(target_sr * duration_s)
    cur_len = waveform.shape[-1]
    if cur_len < target_len:
        waveform = torch.nn.functional.pad(waveform, (0, target_len - cur_len))
    elif cur_len > target_len:
        start = (cur_len - target_len) // 2         # center crop
        waveform = waveform[:, start:start + target_len]

    return waveform                                 # [1, target_len]


@torch.no_grad()
def predict(model, waveform, device, id2label, top_k=3):
    model.eval()
    encoder = getattr(model, "encoder", None)

    if encoder is not None:
        # finetuned model: run the real collate_fn on a single-item batch so
        # the encoder gets inputs in exactly the format it expects.
        collate = collate_fn(encoder, device)
        inputs, _ = collate([(waveform, 0)])  # dummy label -- unused at inference
        logits = model(**inputs)
    else:
        # custom CNN: raw waveform tensor, add batch dim and move to device.
        logits = model(waveform.unsqueeze(0).to(device))

    probs = torch.softmax(logits, dim=-1).squeeze(0)
    top_probs, top_idxs = probs.topk(min(top_k, probs.shape[-1]))
    return [(id2label[idx.item()], prob.item()) for idx, prob in zip(top_idxs, top_probs)]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    label2id = ckpt["label2id"]
    id2label = {v: k for k, v in label2id.items()}

    model = MODEL
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    waveform = load_waveform(WAV_PATH, SR, DURATION_S)
    results = predict(model, waveform, device, id2label, top_k=TOP_K)

    print(f"\n{WAV_PATH}")
    print(f"Predicted: {results[0][0]} ({results[0][1]:.4f})")
    if TOP_K > 1:
        print("Top predictions:")
        for label, prob in results:
            print(f"  {label:20s} {prob:.4f}")


if __name__ == "__main__":
    main()