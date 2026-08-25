#!/usr/bin/env python3
"""Evaluate JSONL predictions from run_content_type_inference.py."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


FOUR_LABELS = ("figure", "table", "line", "section")
TEXT3_LABELS = ("figure", "table", "text")


def f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def normalize_labels(labels: list[str], label_mode: str) -> set[str]:
    out = set()
    for label in labels:
        label = label.strip().lower()
        if label_mode == "text3" and label in {"line", "section"}:
            out.add("text")
        else:
            out.add(label)
    allowed = set(TEXT3_LABELS if label_mode == "text3" else FOUR_LABELS)
    return out & allowed


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
    return records


def evaluate(records: list[dict[str, Any]], label_mode: str) -> dict[str, Any]:
    labels = TEXT3_LABELS if label_mode == "text3" else FOUR_LABELS
    counts = {label: Counter() for label in labels}
    exact = 0
    empty_predictions = 0
    invalid_predictions = 0
    total_predicted_labels = 0
    total_gold_labels = 0

    for record in records:
        gold = normalize_labels(record.get("gold", record.get("gold_raw", [])), label_mode)
        pred_raw = record.get("prediction", [])
        pred = normalize_labels(pred_raw if isinstance(pred_raw, list) else [], label_mode)

        if not pred:
            empty_predictions += 1
        if isinstance(pred_raw, list) and len(pred) != len({str(x).strip().lower() for x in pred_raw}):
            invalid_predictions += 1

        exact += int(pred == gold)
        total_predicted_labels += len(pred)
        total_gold_labels += len(gold)

        for label in labels:
            in_gold = label in gold
            in_pred = label in pred
            if in_gold and in_pred:
                counts[label]["tp"] += 1
            elif not in_gold and in_pred:
                counts[label]["fp"] += 1
            elif in_gold and not in_pred:
                counts[label]["fn"] += 1
            else:
                counts[label]["tn"] += 1

    per_label = {}
    micro = Counter()
    for label in labels:
        tp = counts[label]["tp"]
        fp = counts[label]["fp"]
        fn = counts[label]["fn"]
        support = tp + fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        per_label[label] = {
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1(precision, recall),
        }
        micro.update({"tp": tp, "fp": fp, "fn": fn})

    micro_precision = micro["tp"] / (micro["tp"] + micro["fp"]) if micro["tp"] + micro["fp"] else 0.0
    micro_recall = micro["tp"] / (micro["tp"] + micro["fn"]) if micro["tp"] + micro["fn"] else 0.0
    macro_f1 = sum(per_label[label]["f1"] for label in labels) / len(labels)

    return {
        "label_mode": label_mode,
        "n": len(records),
        "set_accuracy": exact / len(records) if records else 0.0,
        "avg_gold_labels": total_gold_labels / len(records) if records else 0.0,
        "avg_predicted_labels": total_predicted_labels / len(records) if records else 0.0,
        "empty_predictions": empty_predictions,
        "invalid_predictions": invalid_predictions,
        "micro": {
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": f1(micro_precision, micro_recall),
            "tp": micro["tp"],
            "fp": micro["fp"],
            "fn": micro["fn"],
        },
        "macro_f1": macro_f1,
        "per_label": per_label,
    }


def print_report(metrics: dict[str, Any]) -> None:
    print(f"label_mode: {metrics['label_mode']}")
    print(f"n: {metrics['n']}")
    print(f"set_accuracy: {metrics['set_accuracy']:.4f}")
    print(f"micro_precision: {metrics['micro']['precision']:.4f}")
    print(f"micro_recall: {metrics['micro']['recall']:.4f}")
    print(f"micro_f1: {metrics['micro']['f1']:.4f}")
    print(f"macro_f1: {metrics['macro_f1']:.4f}")
    print(f"avg_gold_labels: {metrics['avg_gold_labels']:.3f}")
    print(f"avg_predicted_labels: {metrics['avg_predicted_labels']:.3f}")
    print(f"empty_predictions: {metrics['empty_predictions']}")
    print(f"invalid_predictions: {metrics['invalid_predictions']}")
    print()
    print("per_label:")
    for label, item in metrics["per_label"].items():
        print(
            f"  {label:8s} support={item['support']:4d} "
            f"p={item['precision']:.4f} r={item['recall']:.4f} f1={item['f1']:.4f} "
            f"tp={item['tp']} fp={item['fp']} fn={item['fn']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--label-mode", choices=("four", "text3"), default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    records = read_jsonl(args.predictions)
    label_mode = args.label_mode
    if label_mode is None:
        modes = {record.get("label_mode") for record in records if record.get("label_mode")}
        label_mode = modes.pop() if len(modes) == 1 else "four"

    metrics = evaluate(records, label_mode)
    print_report(metrics)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
