#!/usr/bin/env python3
"""Prepare content-type prediction data from retrieval result keys.

The retrieval experiments only need content-type predictions for queries that
already have retrieval rankings. This script extracts those exact query sets and
writes JSONL files compatible with run_content_type_inference.py and
train_embedding_content_type_classifier.py.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ALLOWED_LABELS = ("figure", "table", "line", "section")
TEXT3_LABELS = ("figure", "table", "text")
DEFAULT_MODELS = {
    "gpt": "GPT-5.1",
    "qwen": "qwen3-30b-instruct",
    "gemma": "gemma300",
}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for record in records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def record_id(dataset_name: str, paper_id: str, query_idx: str) -> str:
    return f"{dataset_name}::{paper_id}::{query_idx}"


def labels_for_mode(label_mode: str) -> tuple[str, ...]:
    return TEXT3_LABELS if label_mode == "text3" else ALLOWED_LABELS


def normalize_gold_types(gold_refs: list[dict[str, Any]], label_mode: str) -> list[str]:
    labels = set()
    for ref in gold_refs or []:
        label = str(ref.get("type", "")).strip().lower()
        if label_mode == "text3" and label in {"line", "section"}:
            labels.add("text")
        elif label in ALLOWED_LABELS:
            labels.add(label)
    return [label for label in labels_for_mode(label_mode) if label in labels]


def load_retrieval_records(results_dir: Path, model: str, label_mode: str) -> list[dict[str, Any]]:
    records = []
    for dataset_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        path = dataset_dir / f"{model}.json"
        if not path.exists():
            continue
        data = read_json(path)
        for paper_id, queries in data.items():
            for query_idx, item in queries.items():
                labels = normalize_gold_types(item.get("gold", []), label_mode)
                if not labels:
                    continue
                records.append(
                    {
                        "id": record_id(dataset_dir.name, paper_id, str(query_idx)),
                        "dataset_name": dataset_dir.name,
                        "og_dataset_name": "EMNLP24" if dataset_dir.name == "EMNLP24_R" else dataset_dir.name,
                        "paper_id": paper_id,
                        "query_idx": str(query_idx),
                        "retrieval_model": model,
                        "label_mode": label_mode,
                        "comment": item.get("query", ""),
                        "gold": labels,
                        "gold_raw": labels,
                    }
                )
    records.sort(key=lambda r: (r["dataset_name"], r["paper_id"], int(r["query_idx"]) if r["query_idx"].isdigit() else r["query_idx"]))
    return records


def stratified_split(
    records: list[dict[str, Any]],
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[tuple(record["gold"])].append(record)

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for group in groups.values():
        rng.shuffle(group)
        n = len(group)
        n_test = max(1, round(n * test_fraction)) if test_fraction > 0 and n >= 3 else 0
        n_val = max(1, round(n * val_fraction)) if val_fraction > 0 and n - n_test >= 3 else 0
        test.extend(group[:n_test])
        val.extend(group[n_test : n_test + n_val])
        train.extend(group[n_test + n_val :])

    for split in (train, val, test):
        split.sort(key=lambda r: r["id"])
    return train, val, test


def print_summary(name: str, records: list[dict[str, Any]]) -> None:
    labels = Counter(label for record in records for label in record["gold"])
    print(f"{name}: n={len(records)} labels={dict(labels)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("splits/retrieval_content_types"))
    parser.add_argument("--gpt-model", default=DEFAULT_MODELS["gpt"])
    parser.add_argument("--qwen-model", default=DEFAULT_MODELS["qwen"])
    parser.add_argument("--gemma-model", default=DEFAULT_MODELS["gemma"])
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--label-mode", choices=("four", "text3"), default="four")
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument(
        "--gemma-test-mode",
        choices=("gpt-shared", "random"),
        default="gpt-shared",
        help="Use GPT-5.1 retrieval keys as Gemma test set, or make a random Gemma split.",
    )
    args = parser.parse_args()

    gpt_records = load_retrieval_records(args.results_dir, args.gpt_model, args.label_mode)
    qwen_records = load_retrieval_records(args.results_dir, args.qwen_model, args.label_mode)
    gemma_records = load_retrieval_records(args.results_dir, args.gemma_model, args.label_mode)

    write_jsonl(args.output_dir / "gpt5_1_predict.jsonl", gpt_records)
    write_jsonl(args.output_dir / "qwen3_30b_instruct_predict.jsonl", qwen_records)
    write_json(args.output_dir / "gpt_shared_ids.json", [record["id"] for record in gpt_records])

    if args.gemma_test_mode == "gpt-shared":
        gpt_ids = {record["id"] for record in gpt_records}
        gemma_test = [record for record in gemma_records if record["id"] in gpt_ids]
        gemma_remaining = [record for record in gemma_records if record["id"] not in gpt_ids]
        gemma_train, gemma_val, _unused = stratified_split(
            gemma_remaining,
            val_fraction=args.val_fraction,
            test_fraction=0.0,
            seed=args.seed,
        )
    else:
        gemma_train, gemma_val, gemma_test = stratified_split(
            gemma_records,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )

    write_jsonl(args.output_dir / "gemma300_train.jsonl", gemma_train)
    write_jsonl(args.output_dir / "gemma300_val.jsonl", gemma_val)
    write_jsonl(args.output_dir / "gemma300_test.jsonl", gemma_test)

    manifest = {
        "results_dir": str(args.results_dir),
        "models": {
            "gpt": args.gpt_model,
            "qwen": args.qwen_model,
            "gemma": args.gemma_model,
        },
        "gemma_test_mode": args.gemma_test_mode,
        "label_mode": args.label_mode,
        "seed": args.seed,
        "files": {
            "gpt_predict": "gpt5_1_predict.jsonl",
            "qwen_predict": "qwen3_30b_instruct_predict.jsonl",
            "gemma_train": "gemma300_train.jsonl",
            "gemma_val": "gemma300_val.jsonl",
            "gemma_test": "gemma300_test.jsonl",
            "shared_ids": "gpt_shared_ids.json",
        },
    }
    write_json(args.output_dir / "manifest.json", manifest)

    print_summary("gpt_predict", gpt_records)
    print_summary("qwen_predict", qwen_records)
    print_summary("gemma_train", gemma_train)
    print_summary("gemma_val", gemma_val)
    print_summary("gemma_test", gemma_test)
    print(f"wrote: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
