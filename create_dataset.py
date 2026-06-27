"""
Memory-efficient rewrite of the inat_sounds filtering/splitting script.

1. Audio bytes are written to disk immediately when a sample is selected,
   instead of being kept in a big Python list. Only lightweight metadata
   (including a file path) is kept in memory during the streaming pass.
2. Splits are built one at a time, and large intermediate structures are
   deleted as soon as they're no longer needed.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DISK SPACE WARNING — read before running
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This script writes audio data to disk THREE TIMES, so you
need roughly 3x the size of the final filtered dataset free:

  PASS 1 — streaming selection  ->  audio_cache/
    Selected audio files are written one-by-one to AUDIO_DIR
    as they are pulled from the upstream stream. At the end
    of the selection pass, audio_cache/ holds the full
    filtered dataset as individual audio files.

  PASS 2a — generator cache  ->  HF_DATASETS_CACHE/generator/...
    Dataset.from_generator() doesn't stream straight into
    filtered_inat_sounds/. It first builds Arrow shards via
    download_and_prepare() into the HuggingFace datasets
    cache (default ~/.cache/huggingface/datasets/generator/,
    overridden below via cache_dir="hf_gen_cache"). This is
    a second full physical copy of the audio bytes, written
    once per split (train/validation/test).

  PASS 2b — Arrow serialization  ->  filtered_inat_sounds/
    save_to_disk() copies the Arrow data out of the
    generator cache into filtered_inat_sounds/, giving a
    third copy. Peak RAM throughout PASS 2 is only
    writer_batch_size x avg_clip, but all three copies sit
    on disk simultaneously until you manually delete
    audio_cache/ and hf_gen_cache/ afterward.

  Example:
    filtered dataset (14 400 clips, ~500 KB avg) ~ 7 GB
    audio_cache/                                  ~ 7 GB
    hf_gen_cache/ (generator cache)                ~ 7 GB
    total peak disk usage                         ~ 21 GB

  Once filtered_inat_sounds/ is saved and verified
  (e.g. via datasets.load_from_disk), you can safely delete
  both intermediate caches to reclaim the extra space:

    import shutil
    shutil.rmtree("audio_cache", ignore_errors=True)
    shutil.rmtree("hf_gen_cache", ignore_errors=True)

RAM usage stays low throughout (only one batch of
writer_batch_size rows in memory at any moment).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import gc
import random
from collections import defaultdict
from datasets import load_dataset, Dataset, DatasetDict, Audio
from tqdm import tqdm

ALLOWED_ORDERS = {
    "Passeriformes", "Piciformes", "Strigiformes", "Charadriiformes",
    "Caprimulgiformes", "Accipitriformes", "Cuculiformes", "Gruiformes",
    "Columbiformes", "Anseriformes", "Psittaciformes", "Galliformes",
}
MAX_PER_ORDER = 1200
AUDIO_DIR = "audio_cache"

os.makedirs(AUDIO_DIR, exist_ok=True)

order_counts = defaultdict(int)
records = []  # metadata only -- no raw audio bytes kept here

ds = load_dataset("DBD-research-group/inat_sounds", split="train", streaming=True)
ds = ds.cast_column("audio", Audio(decode=False))

for i, sample in enumerate(tqdm(ds, desc="Processing taxonomy")):

    order = sample.get("order")
    if order not in ALLOWED_ORDERS or order_counts[order] >= MAX_PER_ORDER:
        continue

    audio_info = sample["audio"]
    audio_bytes = audio_info.get("bytes")
    if not audio_bytes:
        continue

    # Preserve original extension where possible, default to .wav
    src_path = audio_info.get("path") or ""
    ext = os.path.splitext(src_path)[1] or ".wav"
    out_path = os.path.join(AUDIO_DIR, f"{order}_{i}{ext}")
    with open(out_path, "wb") as f:
        f.write(audio_bytes)

    record = {k: v for k, v in sample.items() if k != "audio"}
    record["audio_path"] = out_path
    records.append(record)

    order_counts[order] += 1
    if all(order_counts[o] >= MAX_PER_ORDER for o in ALLOWED_ORDERS):
        break

print("Selected per order:", {o: order_counts[o] for o in ALLOWED_ORDERS})

# Group lightweight records by order, shuffle, split 80/10/10
order_to_records = defaultdict(list)
for r in records:
    order_to_records[r["order"]].append(r)
del records  # free the flat list; data still referenced via order_to_records
gc.collect()

train_records, val_records, test_records = [], [], []
for order, recs in order_to_records.items():
    random.shuffle(recs)
    n = len(recs)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train_records.extend(recs[:n_train])
    val_records.extend(recs[n_train:n_train + n_val])
    test_records.extend(recs[n_train + n_val:])
    print(f"{order}: {n} samples -> train {n_train}, val {n_val}, test {n - n_train - n_val}")

del order_to_records
gc.collect()

random.shuffle(train_records)
random.shuffle(val_records)
random.shuffle(test_records)


def build_split(records_list, name, writer_batch_size=100):

    def row_generator():
        for r in records_list:
            path = r.pop("audio_path")
            with open(path, "rb") as f:
                audio_bytes = f.read()
            r["audio"] = {"bytes": audio_bytes, "path": os.path.basename(path)}
            yield r
            # `r` and `audio_bytes` go out of scope after this iteration's
            # row is written, so the writer's gc can reclaim them before
            # the next row is even read from disk.

    d = Dataset.from_generator(
        row_generator,
        writer_batch_size=writer_batch_size,
        cache_dir="./hf_gen_cache",
    )
    d = d.cast_column("audio", Audio())  # metadata-only change, not a copy
    print(f"Built {name}: {len(d)} rows")
    return d


dataset_dict = DatasetDict({
    "train": build_split(train_records, "train"),
    "validation": build_split(val_records, "validation"),
    "test": build_split(test_records, "test"),
})

dataset_dict.save_to_disk("filtered_inat_sounds")

print("\nFinal split sizes:")
for split, d in dataset_dict.items():
    print(f"{split}: {len(d)}")