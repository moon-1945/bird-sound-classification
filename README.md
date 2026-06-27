# Bird & Animal Sound Classification Platform

A comprehensive deep learning framework for audio-based classification of bird and mammal vocalizations. This repository implements data processing pipelines, custom convolutional architectures, pretrained audio encoders, and episodic prototypical networks for few-shot bioacoustics classification.

---

## 📂 Project Architecture

```directory
bird-classification/
├── classifiers/                 # Supervised classification models
│   ├── custom/                  # Convolutional models built from scratch
│   │   ├── models/              # BirdCNN, BirdResCNN (Residual + Squeeze-and-Excitation), LogMelSpectrogram
│   │   └── train_cnn.py         # Script to train custom CNNs
│   ├── finetuned/               # Fine-tuning pipelines for pre-trained backbones
│   │   ├── encoders/            # DistilHuBERT, BirdAVES (AVEX) wrappers
│   │   ├── audio_classifier.py  # Classification head wrapper
│   │   ├── dataset.py           # Collation logic mapping waveforms to encoders
│   │   ├── train.py             # Fine-tuning orchestrator
│   │   └── trainer.py           # Custom two-stage trainer (frozen/unfrozen phase)
│   ├── inference.py             # Interactive single-file inference script
│   └── test_inference.py        # Automated test evaluation and report generation
├── few_shot/                    # Few-shot episodic learning
│   ├── checkpoints_protonet/    # Prototypical network checkpoints
│   ├── episode_sampler.py       # Episodic sampler (N-way, K-shot, Q-query)
│   ├── train_protonet.py        # Prototypical network trainer
│   ├── eval.py                  # Evaluation suite for prototypical networks
│   └── mammal_dataset.py        # Dataset wrapper for mammal classification
├── mammals-dataset/             # Mammal audio dataset
│   └── Animal_Sound.csv         # Mammal dataset metadata
├── example/                     # Example files for quick testing (e.g., goose.wav)
├── data_utils.py                # Audio decoders, resamplers, and padding utilities
├── waveform_dataset.py          # PyTorch dataset for processing raw audio waveforms
└── create_dataset.py            # Dataset filtering, caching, and serialization script
```

---

## ⚙️ Installation & Setup

1. **Clone the repository** and navigate to the project directory.
2. **Install the dependencies** listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
   > [!NOTE]
   > The fine-tuning scripts support `BirdAVES` using the Earth Species Project `avex` library. This is installed automatically by the `requirements.txt` file.

---

## 📊 1. Dataset Creation & Management

To create the curated training split:
```bash
python create_dataset.py
```

### Disk Space Warning
The script downloads and streams the Hugging Face `DBD-research-group/inat_sounds` dataset, selecting up to 1,200 clips per taxonomic order for 12 allowed orders (14,400 clips total).
Because audio bytes are written/cached during processing, the selection pass copies audio data across three stages:
1. **Pass 1 (Streaming selection)**: Audio files are written one-by-one to `audio_cache/`.
2. **Pass 2a (Generator cache)**: Arrow shards are built in the Hugging Face cache directory (`./hf_gen_cache`).
3. **Pass 2b (Arrow serialization)**: The dataset is serialized to the `filtered_inat_sounds/` folder.

Once the dataset is successfully generated, you can free up ~14 GB of disk space by deleting intermediate directories:
```python
import shutil
shutil.rmtree("audio_cache", ignore_errors=True)
shutil.rmtree("hf_gen_cache", ignore_errors=True)
```

---

## 🔊 2. Supervised Classifiers

### Training Custom CNNs (`BirdResCNN` / `BirdCNN`)
To train a convolutional model from scratch using log-mel spectrograms:
```bash
python classifiers/custom/train_cnn.py --sr 32000 --epochs 150 --base_ch 32 --dropout 0.4
```
*   `BirdResCNN` includes residual stages, MaxPool downsampling, and optional **Squeeze-and-Excitation (SE)** attention blocks.
*   The training pipeline uses Cosine Annealing learning rate schedules and early stopping based on validation macro-F1.

### Fine-Tuning Audio Encoders
You can fine-tune pretrained models on the processed dataset. Supported models are:
*   `distilhubert` (pretrained speech representation)
*   `birdaves` (pretrained animal vocalization representation via AVEX)

To fine-tune, run:
```bash
python classifiers/finetuned/train.py --model birdaves --birdaves_model esp_aves2_sl_beats_bio --epochs 40
```
The trainer uses a **two-phase unfreezing strategy**:
1. **Phase 1 (Frozen Encoder)**: Trains only the classification head for a set number of epochs.
2. **Phase 2 (Full Fine-Tuning)**: Unfreezes the encoder backbone and trains the entire network with separate learning rates for the backbone and classification head.

---

## 🧪 3. Evaluation & Inference

### Model Evaluation
To evaluate your trained checkpoints on the test split, configure the model configurations array in `classifiers/test_inference.py` and run:
```bash
python classifiers/test_inference.py
```
This generates:
*   A classification report for each checkpoint in `test_eval_reports/`.
*   A summary table (`summary.txt` and `summary.csv`) containing accuracy and macro-F1 comparisons.

### Single-File Inference
To run classification on a single audio clip (e.g., `example/goose.wav`):
```bash
python classifiers/inference.py
```
Configure the path, checkpoint location, and model configuration block inside `classifiers/inference.py` to match the target checkpoint.

---

## 🧠 4. Few-Shot Prototypical Networks

Prototypical Networks (`ProtoNet`) allow classification of completely unseen categories with very few examples.

### episodic Training
To train the prototypical network (episodic training with base taxonomic classes):
```bash
python few_shot/train_protonet.py --mode head_only --cnn_ckpt classifiers/custom/checkpoints_cnn/best_cnn.pt
```
*   `--mode head_only`: Keeps the CNN encoder frozen and trains a projection MLP head into a metric space.
*   `--mode full`: End-to-end episodic training of the entire feature extractor.

### episodic Evaluation
To evaluate few-shot accuracy on unseen classes:
```bash
python few_shot/eval.py --n_way 5 --k_shot 5
```
This calculates classification accuracy on held-out orders (`Strigiformes`, `Caprimulgiformes`, `Psittaciformes`, `Gruiformes`) or mammal classes.

---

### 🐾 Case Study: Transferring Bird Embeddings to Few-Shot Mammal Classification

An important experiment was conducted to evaluate how well embeddings trained on bird classification generalize to an entirely different domain: **mammal vocalizations** (`mammals-dataset`).

#### Experimental Results (13-Way Episodic Evaluation on Mammals)

Using `few_shot/eval.py` to evaluate different embedding backbones across 1, 5, 10, and 20 shots yields the following results:

| Backbone Encoder / Model | 1-Shot Acc | 5-Shot Acc | 10-Shot Acc | 20-Shot Acc |
| :--- | :---: | :---: | :---: | :---: |
| **`res_ch32_nose_frozen`** (Supervised CNN, no SE) | **28.6% ± 0.9%** | **49.6% ± 0.7%** | **57.9% ± 0.6%** | **62.2% ± 0.6%** |
| **`res_ch32_se_frozen`** (Supervised CNN, with SE) | 27.0% ± 0.8% | 43.7% ± 0.6% | 50.9% ± 0.5% | 54.4% ± 0.5% |
| **`conv_ch32_plain_frozen`** (Supervised Plain CNN) | 29.2% ± 0.9% | 45.1% ± 0.7% | 51.0% ± 0.6% | 53.7% ± 0.5% |
| **`protonet_res_ch32_nose_head_only_N5K4Q8`** | 15.4% ± 0.6% | 22.0% ± 0.6% | 24.9% ± 0.5% | 26.8% ± 0.5% |
| **`protonet_res_ch32_se_head_only_N5K4Q8`** | 17.3% ± 0.7% | 23.4% ± 0.5% | 25.7% ± 0.5% | 28.1% ± 0.5% |
| **`protonet_conv_ch32__head_only_N5K5Q10`** | 20.3% ± 0.8% | 27.7% ± 0.6% | 30.9% ± 0.5% | 33.1% ± 0.6% |

#### 💡 Key Insights & Findings

1. **Supervised Features Generalize Best:**
   The raw, frozen feature space from the supervised classifier trained via Cross-Entropy (specifically `res_ch32_nose_frozen`) performs the best, reaching **62.2%** accuracy at 20 shots. Standard supervised pre-training preserves broad, high-dimensional acoustic representations that adapt well to out-of-domain transfer (birds ➡️ mammals).

2. **Prototypical Bottleneck & Feature Suppression:**
   Models trained using ProtoNet episodic training with a custom projection head (`protonet_*`) perform much worse (only **26.8% - 33.1%** at 20-shot). Training a projection MLP head specifically to optimize metric distances on bird orders creates a feature bottleneck—it discards acoustic information that is not discriminative for birds but crucial for mammal sound recognition.

3. **Avoiding Over-Specialization:**
   For cross-domain transfer (e.g., classifying mammals using bird models), **avoid using the ProtoNet projection heads trained on bird subclasses**. Instead, extract raw embeddings directly from the CNN backbone, or perform episodic fine-tuning end-to-end (`--mode full`) using a very low learning rate to preserve the generalization of the feature extractor.

