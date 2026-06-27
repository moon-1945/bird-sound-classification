"""
Taxonomy distribution analysis for filtered_inat_sounds.

Covers:
- Per-level class counts, imbalance ratios, top-20 breakdowns
- Per-split distribution (to catch train/val/test skew)
- Species richness per order (how many unique species live inside each order)
- Long-tail analysis (classes with very few samples — the hard ones for a model)
- Co-occurrence check: species that appear in more than one order (data sanity)
- Least-represented classes (the bottom-20 most likely to hurt recall)
"""

from collections import Counter, defaultdict
from datasets import load_from_disk, Audio

# ── helpers ───────────────────────────────────────────────────────────────────

SEP  = "=" * 60
SEP2 = "-" * 60

def pct(n, total):
    return f"{100 * n / total:.1f}%" if total else "n/a"

def print_counter_summary(lvl, c, top_n=20):
    if not c:
        return
    total  = sum(c.values())
    unique = len(c)
    counts = sorted(c.values(), reverse=True)
    print(SEP)
    print(f"LEVEL : {lvl.upper()}")
    print(f"  unique classes : {unique}")
    print(f"  total samples  : {total}")
    print(f"  mean / class   : {total / unique:.1f}")
    print(f"  median count   : {counts[unique // 2]}")
    print(f"  max count      : {counts[0]}  ({c.most_common(1)[0][0]})")
    print(f"  min count      : {counts[-1]}  ({c.most_common()[-1][0]})")
    print(f"  imbalance ratio: {counts[0] / counts[-1]:.1f}x")

    # share of data held by the top-10 classes
    top10_sum = sum(n for _, n in c.most_common(10))
    print(f"  top-10 classes hold {pct(top10_sum, total)} of data")

    print(f"\n  TOP {top_n}:")
    for rank, (name, count) in enumerate(c.most_common(top_n), 1):
        bar = "█" * (count * 30 // counts[0])
        print(f"  {rank:>3}. {name:<35} {count:>5}  {pct(count, total):>6}  {bar}")

# ── load ──────────────────────────────────────────────────────────────────────

dataset_dict = load_from_disk("filtered_inat_sounds")

LEVELS = ["kingdom", "phylum", "class", "order",
          "family", "genus", "specific_epithet", "supercategory"]

# ── 1. global counters (all splits combined) ──────────────────────────────────

global_counters = {lvl: Counter() for lvl in LEVELS}

# also build per-split counters for the split-skew check
split_counters = {
    split: {lvl: Counter() for lvl in LEVELS}
    for split in dataset_dict
}

# order → set of species (for richness analysis)
order_species: dict[str, set] = defaultdict(set)

# species → set of orders (for sanity check)
species_orders: dict[str, set] = defaultdict(set)

for split_name, split_ds in dataset_dict.items():
    split_ds = split_ds.cast_column("audio", Audio(decode=False))
    for row in split_ds:
        for lvl in LEVELS:
            val = row.get(lvl)
            if val is not None:
                global_counters[lvl][val] += 1
                split_counters[split_name][lvl][val] += 1

        order   = row.get("order")
        species = row.get("specific_epithet")
        if order and species:
            order_species[order].add(species)
            species_orders[species].add(order)

# ── 2. per-level global summary ───────────────────────────────────────────────

print("\n\n📊  GLOBAL TAXONOMY DISTRIBUTION  (all splits combined)\n")
for lvl in LEVELS:
    print_counter_summary(lvl, global_counters[lvl])
print()

# ── 3. per-split distribution (catch skew) ───────────────────────────────────

print("\n\n🔀  PER-SPLIT DISTRIBUTION CHECK  (order level)\n")
print(f"  {'order':<25}", end="")
for split in dataset_dict:
    print(f"  {split:>12}", end="")
print(f"  {'total':>8}  {'split balance'}")
print(SEP2 * 2)

order_counter = global_counters["order"]
for order, _ in order_counter.most_common():
    row_parts = [f"  {order:<25}"]
    counts = []
    for split in dataset_dict:
        n = split_counters[split]["order"].get(order, 0)
        counts.append(n)
        row_parts.append(f"  {n:>12}")
    total = sum(counts)
    # expected rough split is 80/10/10; flag if any split deviates > 5 pp
    expected = [0.80, 0.10, 0.10]
    deviations = [abs(c / total - e) for c, e in zip(counts, expected)]
    flag = "  ⚠️  SKEWED" if any(d > 0.05 for d in deviations) else ""
    row_parts.append(f"  {total:>8}{flag}")
    print("".join(row_parts))

# ── 4. species richness per order ─────────────────────────────────────────────

print("\n\n🌿  SPECIES RICHNESS PER ORDER  (unique specific_epithet per order)\n")
print(f"  {'order':<25} {'species':>8}  {'samples':>8}  {'samples/species':>16}")
print(SEP2)

order_sample_counts = global_counters["order"]
richness_rows = sorted(
    order_species.items(), key=lambda kv: len(kv[1]), reverse=True
)
for order, species_set in richness_rows:
    n_species = len(species_set)
    n_samples = order_sample_counts.get(order, 0)
    ratio     = n_samples / n_species if n_species else 0
    print(f"  {order:<25} {n_species:>8}  {n_samples:>8}  {ratio:>16.1f}")

# ── 5. long-tail analysis (family / genus / species) ─────────────────────────

print("\n\n🐦  LONG-TAIL ANALYSIS\n")

thresholds = [5, 10, 20, 50]
for lvl in ("family", "genus", "specific_epithet"):
    c = global_counters[lvl]
    total_classes = len(c)
    if not total_classes:
        continue
    print(f"  {lvl.upper()}")
    for t in thresholds:
        rare = sum(1 for n in c.values() if n < t)
        print(f"    classes with < {t:>3} samples: {rare:>5} / {total_classes}"
              f"  ({pct(rare, total_classes)} of classes,"
              f"  {pct(sum(n for n in c.values() if n < t), sum(c.values()))} of data)")
    print()

# ── 6. bottom-20 least-represented species ────────────────────────────────────

print("\n\n⚠️   BOTTOM 20 SPECIES  (most likely to hurt recall)\n")
species_counter = global_counters["specific_epithet"]
bottom20 = species_counter.most_common()[:-21:-1]
for rank, (name, count) in enumerate(reversed(bottom20), 1):
    orders = ", ".join(sorted(species_orders.get(name, [])))
    print(f"  {rank:>3}. {name:<35} {count:>4} samples  ({orders})")

# ── 7. data-sanity: species appearing in multiple orders ──────────────────────

print("\n\n🔍  SANITY CHECK: species appearing in > 1 order\n")
multi_order = {sp: ords for sp, ords in species_orders.items() if len(ords) > 1}
if not multi_order:
    print("  ✅  All species appear in exactly one order.")
else:
    print(f"  ⚠️  {len(multi_order)} species found in multiple orders:")
    for sp, ords in sorted(multi_order.items()):
        print(f"    {sp:<35} → {', '.join(sorted(ords))}")

# ── 8. quick summary table ────────────────────────────────────────────────────

print("\n\n📋  QUICK SUMMARY TABLE\n")
print(f"  {'level':<20} {'classes':>8}  {'avg/class':>10}  {'imbalance':>12}")
print(SEP2)
for lvl in LEVELS:
    c = global_counters[lvl]
    if not c:
        continue
    vals = list(c.values())
    imbalance = max(vals) / min(vals) if min(vals) else float("inf")
    print(f"  {lvl:<20} {len(c):>8}  {sum(vals)/len(vals):>10.1f}  {imbalance:>11.1f}x")