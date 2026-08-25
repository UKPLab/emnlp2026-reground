#!/usr/bin/env python3
"""Create deterministic train/validation/test JSONL splits for content labels."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LABELS = ("figure", "table", "line", "section")


def load_examples(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Expected dataset JSON to be an object: comment -> labels")

    examples = []
    for idx, (comment, labels) in enumerate(data.items()):
        if not isinstance(comment, str) or not isinstance(labels, list):
            raise ValueError(f"Bad example at index {idx}: expected string -> list")
        normalized = [label for label in LABELS if label in {str(x).lower() for x in labels}]
        if not normalized:
            raise ValueError(f"Example {idx} has no supported labels: {labels!r}")
        examples.append({"id": idx, "comment": comment, "labels": normalized})
    return examples


def target_counts(size: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    train = round(size * train_ratio)
    val = round(size * val_ratio)
    return train, val, size - train - val


def split_examples(
    examples: list[dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """Stratify by observed label set, a stable fit for this four-label dataset."""
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        groups[tuple(example["labels"])].append(example)

    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for group in groups.values():
        rng.shuffle(group)
        train_n, val_n, _ = target_counts(len(group), train_ratio, val_ratio)
        splits["train"].extend(group[:train_n])
        splits["val"].extend(group[train_n : train_n + val_n])
        splits["test"].extend(group[train_n + val_n :])

    for split in splits.values():
        rng.shuffle(split)
    return splits


def write_jsonl(path: Path, examples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for example in examples:
            out.write(json.dumps(example, ensure_ascii=False) + "\n")


def describe(examples: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter()
    combinations = Counter()
    for example in examples:
        labels.update(example["labels"])
        combinations["+".join(example["labels"])] += 1
    return {
        "size": len(examples),
        "label_counts": {label: labels[label] for label in LABELS},
        "label_combinations": dict(combinations.most_common()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("dataset.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("splits/content_types"))
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.val_ratio <= 0:
        raise ValueError("Train and validation ratios must be positive")
    if args.train_ratio + args.val_ratio >= 1:
        raise ValueError("Train ratio + validation ratio must leave a test split")

    examples = load_examples(args.dataset)
    splits = split_examples(examples, args.train_ratio, args.val_ratio, args.seed)
    manifest = {
        "source": str(args.dataset),
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": 1 - args.train_ratio - args.val_ratio,
        "stratification": "observed label combination",
        "splits": {name: describe(split) for name, split in splits.items()},
    }

    for name, split in splits.items():
        write_jsonl(args.output_dir / f"{name}.jsonl", split)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
