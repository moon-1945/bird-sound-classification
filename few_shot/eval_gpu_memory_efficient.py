import argparse
import os
import json
import numpy as np
import torch
import scipy.stats
from tqdm import tqdm
from classifiers.custom.models.bird_cnn import BirdCNN
from classifiers.custom.models.bird_res_cnn import BirdResCNN
from few_shot.episode_sampler import EpisodeSampler
from few_shot.mammal_dataset import MammalDataset
from few_shot.train_protonet import ProtoNetModel

NUM_CLASSES = 12
DURATION_S = 5.0


MODEL_CONFIGS = [
    {
        "name": "conv_ch32_plain_frozen",
        "cnn_ckpt": "../classifiers/custom/checkpoints_cnn/best_cnn_base_ch=32_conv.pt",
        "build_model": lambda: BirdCNN(num_classes=NUM_CLASSES, in_ch=1, base_ch=32, sr=32000),
        "protonet_ckpt": None,
        "sr": 32000,
    },
    {
        "name": "res_ch32_se_frozen",
        "cnn_ckpt": "../classifiers/custom/checkpoints_cnn/best_cnn_base_ch=32_res.pt",
        "build_model": lambda: BirdResCNN(num_classes=NUM_CLASSES, in_ch=1, base_ch=32, sr=32000),
        "protonet_ckpt": None,
        "sr": 32000,
    },
    {
        "name": "res_ch32_nose_frozen",
        "cnn_ckpt": "../classifiers/custom/checkpoints_cnn/best_cnn_base_ch=32_res_no_se_block.pt",
        "build_model": lambda: BirdResCNN(num_classes=NUM_CLASSES, in_ch=1, base_ch=32, sr=32000, with_se=False),
        "protonet_ckpt": None,
        "sr": 32000,
    },
    {
        "name": "protonet_conv_ch32__head_only_N5K5Q10",
        "build_model": lambda: BirdCNN(num_classes=NUM_CLASSES, in_ch=1, base_ch=32, sr=32000),
        "protonet_ckpt": "checkpoints_protonet/protonet_conv_ch32__head_only_N5K5Q10.pt",
        "mode": "head_only",
        "sr": 32000,
    },
    {
        "name": "protonet_res_ch32_se_head_only_N5K4Q8",
        "build_model": lambda: BirdResCNN(num_classes=NUM_CLASSES, in_ch=1, base_ch=32, sr=32000, with_se=True),
        "protonet_ckpt": "checkpoints_protonet/protonet_res_ch32_se_head_only_N5K4Q8.pt",
        "mode": "head_only",
        "sr": 32000,
    },
    {
        "name": "protonet_res_ch32_nose_head_only_N5K4Q8",
        "build_model": lambda: BirdResCNN(num_classes=NUM_CLASSES, in_ch=1, base_ch=32, sr=32000, with_se=False),
        "protonet_ckpt": "checkpoints_protonet/protonet_res_ch32_nose_head_only_N5K4Q8.pt",
        "mode": "head_only",
        "sr": 32000,
    },
]


def build_encoder(cfg: dict, device: torch.device) -> ProtoNetModel:
    cnn = cfg["build_model"]()

    if cfg["protonet_ckpt"] is None:
        ckpt = torch.load(cfg["cnn_ckpt"], map_location=device)
        cnn.load_state_dict(ckpt["model"])
        model = ProtoNetModel(
            cnn=cnn, mode="full",
            encoder_dim=None, proj_dim=None, train=False,
        )
    else:
        proto_ckpt = torch.load(cfg["protonet_ckpt"], map_location=device)
        mode        = proto_ckpt["mode"]
        proj_dim    = proto_ckpt["proj_dim"]
        encoder_dim = proto_ckpt["encoder_dim"]

        model = ProtoNetModel(
            cnn=cnn, mode=mode,
            encoder_dim=encoder_dim, proj_dim=proj_dim, train=False,
        )
        model.load_state_dict(proto_ckpt["model"])

    # single device transfer for the fully-assembled wrapper — avoids moving
    # the bare cnn to device and then moving the whole ProtoNetModel again
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def embed_in_chunks(
    flat: torch.Tensor,
    encoder,
    batch_size: int = None,
) -> torch.Tensor:
    """
    flat: [M, 1, T]
    returns: [M, D] embeddings.

    Encodes `flat` batch_size rows at a time instead of in one forward pass,
    so a large episode (big n_way * k_shot or n_way * n_query) doesn't blow
    past available GPU memory. If batch_size is None or >= M, this is just
    a single forward pass (same behaviour as before).
    """
    if batch_size is None or batch_size >= flat.shape[0]:
        return encoder.embed(flat)

    chunks = []
    for start in range(0, flat.shape[0], batch_size):
        end = start + batch_size
        chunks.append(encoder.embed(flat[start:end]))
    return torch.cat(chunks, dim=0)


def compute_prototypes(
    support: torch.Tensor,
    encoder,
    batch_size: int = None,
) -> torch.Tensor:
    """
    support: [N, K, 1, T]
    returns: [N, D] — one prototype per class, mean of K support embeddings
    """
    N, K, C, T = support.shape
    # flatten N and K into one logical batch, then encode it in
    # capped-size chunks to bound peak GPU memory usage
    flat = support.reshape(N * K, C, T)                      # [N*K, 1, T]
    embeddings = embed_in_chunks(flat, encoder, batch_size)   # [N*K, D]
    embeddings = embeddings.reshape(N, K, -1)                 # [N, K, D]
    prototypes = embeddings.mean(dim=1)                       # [N, D]
    # re-normalise after averaging — mean of unit vectors isn't unit length
    return torch.nn.functional.normalize(prototypes, p=2, dim=-1)


def classify_queries(
    query: torch.Tensor,
    prototypes: torch.Tensor,
    encoder,
    batch_size: int = None,
) -> torch.Tensor:
    """
    query:      [N, Q, 1, T]
    prototypes: [N, D]
    returns:    [N*Q] predicted episode-local class ids
    """
    N, Q, C, T = query.shape
    flat = query.reshape(N * Q, C, T)                         # [N*Q, 1, T]
    embeddings = embed_in_chunks(flat, encoder, batch_size)    # [N*Q, D]

    # squared euclidean distance to each prototype
    # embeddings: [N*Q, D]  prototypes: [N, D]
    dists = torch.cdist(embeddings, prototypes)               # [N*Q, N]
    return dists.argmin(dim=-1)                               # [N*Q] predicted class


def episode_accuracy(
    support: torch.Tensor,
    query: torch.Tensor,
    labels: torch.Tensor,
    encoder,
    batch_size: int = None,
) -> float:
    """
    Run one episode and return accuracy.
    labels: [N] episode-local 0..N-1, one per class.
    Ground truth for query: repeat each label Q times.
    """
    N, Q = query.shape[0], query.shape[1]
    prototypes = compute_prototypes(support, encoder, batch_size)
    preds = classify_queries(query, prototypes, encoder, batch_size)

    # ground truth: class 0 appears Q times, class 1 Q times, etc.
    gt = labels.repeat_interleave(Q)                          # [N*Q]
    correct = (preds.cpu() == gt).float().sum().item()
    return correct / (N * Q)


def evaluate_encoder(
    encoder,
    sampler: EpisodeSampler,
    device: torch.device,
    batch_size: int = None,
) -> dict:
    """
    Run all episodes in sampler, return mean acc and 95% CI.
    """
    accs = []
    for support, query, labels in tqdm(sampler):
        support = support.to(device)
        query   = query.to(device)
        acc = episode_accuracy(support, query, labels, encoder, batch_size)
        accs.append(acc)

    accs = np.array(accs)
    mean = accs.mean()
    # 95% confidence interval via t-distribution
    ci   = scipy.stats.sem(accs) * scipy.stats.t.ppf(0.975, df=len(accs) - 1)
    return {"mean_acc": float(mean), "ci_95": float(ci), "n_episodes": len(accs)}


def print_results_table(results: list[dict]):
    header = f"{'encoder':50s} {'k_shot':>6s} {'acc':>8s} {'95% CI':>10s}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['encoder']:50s} {r['k_shot']:>6d} "
            f"{r['mean_acc']:>7.1%} ± {r['ci_95']:>6.1%}"
        )
    print("=" * len(header))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mammal_dir", type=str, default="../mammals-dataset/Animal-Soundprepros")
    parser.add_argument("--k_shot", type=int, nargs="+", default=[1, 5, 10, 20],
                        help="K values to evaluate. Default: 1 5")
    parser.add_argument("--n_way", type=int, default=13)
    parser.add_argument("--n_query", type=int, default=20)
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42,
                        help="Fixed seed for reproducible episode sampling.")
    parser.add_argument("--embed_batch_size", type=int, default=32,
                        help="Max number of clips encoded in a single forward "
                             "pass. Support/query tensors for an episode are "
                             "split into chunks of this size before being fed "
                             "to the encoder, to avoid GPU OOM on large "
                             "n_way/k_shot/n_query combinations. Set to 0 or "
                             "a negative number to disable chunking.")
    parser.add_argument("--out", default="few_shot_results/frozen_baseline.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    embed_batch_size = args.embed_batch_size if args.embed_batch_size > 0 else None

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    all_results = []

    mammal_ds = {32000: MammalDataset(
        args.mammal_dir, target_sr=32000, duration_s=5.0, train=False
    )}

    with torch.no_grad():
        for cfg in MODEL_CONFIGS:
            name = cfg["name"]
            sr = cfg["sr"]

            print(f"\n{'='*60}")
            print(f"Loading {name}")
            print(f"{'='*60}")

            encoder = build_encoder(cfg, device)

            for k_shot in args.k_shot:
                sampler = EpisodeSampler(
                    dataset=mammal_ds[sr],
                    n_way=args.n_way,
                    k_shot=k_shot,
                    n_query=args.n_query,
                    n_episodes=args.n_episodes,
                    seed=args.seed,
                )

                result = evaluate_encoder(encoder, sampler, device, embed_batch_size)

                result["encoder"] = name
                result["k_shot"] = k_shot
                all_results.append(result)

                print(
                    f"  k={k_shot}  {name}: {result['mean_acc']:.1%} "
                    f"± {result['ci_95']:.1%}  ({result['n_episodes']} episodes)"
                )

            del encoder
            torch.cuda.empty_cache()

    print_results_table(all_results)

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.out}")


if __name__ == '__main__':
    main()