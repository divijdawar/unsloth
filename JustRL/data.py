"""Dataset preparation for JustRL GRPO on DAPO-Math-17k."""
from __future__ import annotations
from typing import Optional
from datasets import Dataset, load_dataset
from .reward import dapo_binary_reward

DAPO_DATASET_ID = "BytedTsinghua-SIA/DAPO-Math-17k"
BOXED_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."

def _flatten_example(example: dict) -> dict:
    return {
        "prompt": example["prompt"],
        "answer": str(example["reward_model"]["ground_truth"]),
    }

def _append_boxed_suffix(example: dict) -> dict:
    content = example["prompt"][0]["content"] + BOXED_SUFFIX
    return {"prompt": [{"role": "user", "content": content}]}

def _dedup_by_prompt(ds: Dataset) -> Dataset:
    seen: set[str] = set()
    keep: list[int] = []
    for i, msgs in enumerate(ds["prompt"]):
        key = msgs[0]["content"]
        if key not in seen:
            seen.add(key)
            keep.append(i)
    return ds.select(keep)

def investigate_duplicates(ds: Dataset, sample_step: int = 100) -> None:
    import hashlib

    hashes: set[str] = set()
    n = 0
    for i in range(0, len(ds), sample_step):
        hashes.add(hashlib.md5(ds[i]["prompt"][0]["content"].encode()).hexdigest())
        n += 1
    print(f"Sampled {n} rows (every {sample_step}th), unique prompts: {len(hashes)}")

def load_dapo_math_17k(
    split: str = "train",
    append_boxed_suffix: bool = True,
    dedup: bool = False,
    num_proc: Optional[int] = None,
) -> Dataset:
    ds = load_dataset(DAPO_DATASET_ID, split=split)
    ds = ds.map(_flatten_example, remove_columns=ds.column_names, num_proc=num_proc)
    if append_boxed_suffix:
        ds = ds.map(_append_boxed_suffix, num_proc=num_proc)
    if dedup:
        ds = _dedup_by_prompt(ds)
    return ds

def sanity_check(ds: Dataset, n: int = 5) -> None:
    for i in range(min(n, len(ds))):
        gt = ds[i]["answer"]
        dummy = f"Therefore the answer is \\boxed{{{gt}}}."
        r = dapo_binary_reward(
            prompts=[ds[i]["prompt"]],
            completions=[[{"content": dummy}]],
            answer=[gt],
        )[0]
        print(f"row {i}: gt={gt!r:15} dummy_reward={r} (expect 1.0)")

if __name__ == "__main__":
    ds = load_dapo_math_17k(append_boxed_suffix=True, dedup=False)
    print(f"Loaded {len(ds)} rows. Columns: {ds.column_names}")
    investigate_duplicates(ds)
    print("--- Sample row ---")
    print("prompt:", ds[0]["prompt"])
    print("answer:", ds[0]["answer"])
    print("--- Sanity check ---")
    sanity_check(ds)
