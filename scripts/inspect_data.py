"""Inspect raw and normalized training data samples.

Usage:
  python -m scripts.inspect_data
  # or in container:
  docker compose run --rm --entrypoint python train /app/scripts/inspect_data.py
"""

from datasets import load_dataset
from safety_finetune.data_utils import (
    _load_clearharm,
    _load_llm_lat_biochem,
    _load_beavertails_biochem,
    _load_slimorca_safe,
    _load_wmdp_eval,
    _matches_biochem,
)

N_SAMPLES = 3


def header(title: str):
    print(f"\n{'=' * 80}")
    print(title)
    print("=" * 80)


def show_raw_clearharm():
    header("CLEARHARM - Raw (first 3 rows)")
    ds = load_dataset("AlignmentResearch/ClearHarm", "default", split="train")
    for i, row in enumerate(ds):
        if i >= N_SAMPLES:
            break
        print(f"\n--- Row {i} ---")
        print(f"content:            {row['content']}")
        print(f"gen_target:         {row['gen_target'][:200]}")
        print(f"proxy_gen_target:   {row['proxy_gen_target'][:200]}")
        print(f"clf_label:          {row['clf_label']}")


def show_raw_llm_lat():
    header("LLM-LAT/harmful-dataset - Raw (first 3 bio/chem matches)")
    ds = load_dataset("LLM-LAT/harmful-dataset", split="train")
    shown = 0
    for row in ds:
        if not _matches_biochem(row["prompt"]):
            continue
        print(f"\n--- Match {shown} ---")
        print(f"prompt:   {row['prompt']}")
        print(f"chosen:   {row['chosen'][:300]}")
        print(f"rejected: {row['rejected'][:300]}")
        shown += 1
        if shown >= N_SAMPLES:
            break
    print(f"\n(showed {shown} bio/chem matches)")


def show_raw_beavertails():
    header("BEAVERTAILS - Raw (first 3 bio/chem keyword matches)")
    ds = load_dataset("PKU-Alignment/BeaverTails", split="330k_train")
    shown = 0
    for row in ds:
        if row["is_safe"]:
            continue
        if not _matches_biochem(row["prompt"]):
            continue
        print(f"\n--- Match {shown} ---")
        print(f"prompt:     {row['prompt'][:400]}")
        print(f"response:   {row['response'][:400]}")
        print(f"is_safe:    {row['is_safe']}")
        cats = [k for k, v in row["category"].items() if v]
        print(f"categories: {cats}")
        shown += 1
        if shown >= N_SAMPLES:
            break
    print(f"\n(showed {shown} bio/chem matches)")


def show_raw_slimorca():
    header("SLIMORCA - Raw (first 3 rows)")
    ds = load_dataset("Open-Orca/SlimOrca", split="train")
    for i, row in enumerate(ds):
        if i >= N_SAMPLES:
            break
        convs = row["conversations"]
        print(f"\n--- Row {i} ---")
        for c in convs:
            print(f"  [{c['from']}]: {c['value'][:200]}")


def show_raw_wmdp():
    header("WMDP - Raw (first 3 bio + first 3 chem)")
    for subset in ["wmdp-bio", "wmdp-chem"]:
        ds = load_dataset("cais/wmdp", subset, split="test")
        print(f"\n  --- {subset} ---")
        for i, row in enumerate(ds):
            if i >= N_SAMPLES:
                break
            choices = " | ".join(f"{chr(65+j)}. {c}" for j, c in enumerate(row["choices"]))
            print(f"\n  [{i}] {row['question']}")
            print(f"      {choices}")
            print(f"      answer: {chr(65 + row['answer'])}")


def show_normalized():
    header("NORMALIZED - ClearHarm (after data_utils processing)")
    examples = _load_clearharm()
    for i, ex in enumerate(examples[:N_SAMPLES * 2]):  # show pairs
        print(f"\n--- Example {i} ---")
        print(f"prompt:     {ex['prompt'][:300]}")
        print(f"completion: {ex['completion'][:300]}")
        print(f"label:      {ex['label']}")

    header("NORMALIZED - LLM-LAT bio/chem (after data_utils processing)")
    examples = _load_llm_lat_biochem()
    for i, ex in enumerate(examples[:N_SAMPLES * 2]):
        print(f"\n--- Example {i} ---")
        print(f"prompt:     {ex['prompt'][:300]}")
        print(f"completion: {ex['completion'][:300]}")
        print(f"label:      {ex['label']}")

    header("NORMALIZED - BeaverTails bio/chem (after data_utils processing)")
    examples = _load_beavertails_biochem()
    for i, ex in enumerate(examples[:N_SAMPLES * 2]):
        print(f"\n--- Example {i} ---")
        print(f"prompt:     {ex['prompt'][:300]}")
        print(f"completion: {ex['completion'][:300]}")
        print(f"label:      {ex['label']}")

    header("NORMALIZED - SlimOrca safe (after data_utils processing)")
    examples = _load_slimorca_safe(n=N_SAMPLES)
    for i, ex in enumerate(examples):
        print(f"\n--- Example {i} ---")
        print(f"prompt:     {ex['prompt'][:300]}")
        print(f"completion: {ex['completion'][:300]}")
        print(f"label:      {ex['label']}")

    header("NORMALIZED - WMDP eval (after data_utils processing)")
    examples = _load_wmdp_eval()
    for i, ex in enumerate(examples[:N_SAMPLES]):
        print(f"\n--- Example {i} ---")
        print(f"prompt:     {ex['prompt'][:400]}")
        print(f"source:     {ex['source']}")


def show_counts():
    header("DATASET COUNTS (normalized)")
    clearharm = _load_clearharm()
    llm_lat = _load_llm_lat_biochem()
    beavertails = _load_beavertails_biochem()
    safe = _load_slimorca_safe(n=2000)
    wmdp = _load_wmdp_eval()

    print(f"ClearHarm:     {len(clearharm):>5} examples ({sum(1 for e in clearharm if e['label']):>4} desirable, {sum(1 for e in clearharm if not e['label']):>4} undesirable)")
    print(f"LLM-LAT:       {len(llm_lat):>5} examples ({sum(1 for e in llm_lat if e['label']):>4} desirable, {sum(1 for e in llm_lat if not e['label']):>4} undesirable)")
    print(f"BeaverTails:   {len(beavertails):>5} examples ({sum(1 for e in beavertails if e['label']):>4} desirable, {sum(1 for e in beavertails if not e['label']):>4} undesirable)")
    print(f"SlimOrca safe: {len(safe):>5} examples (all desirable)")
    print(f"WMDP eval:     {len(wmdp):>5} prompts (held-out, no labels)")
    total_train = len(clearharm) + len(llm_lat) + len(beavertails) + len(safe)
    print(f"\nTotal training: {total_train}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect training and eval data")
    parser.add_argument("--raw", action="store_true", help="Show raw dataset samples")
    parser.add_argument("--normalized", action="store_true", help="Show normalized samples")
    parser.add_argument("--counts", action="store_true", help="Show dataset counts")
    parser.add_argument("--all", action="store_true", help="Show everything")
    args = parser.parse_args()

    if not any([args.raw, args.normalized, args.counts, args.all]):
        args.all = True

    if args.raw or args.all:
        show_raw_clearharm()
        show_raw_llm_lat()
        show_raw_beavertails()
        show_raw_slimorca()
        show_raw_wmdp()

    if args.normalized or args.all:
        show_normalized()

    if args.counts or args.all:
        show_counts()
