#!/usr/bin/env python3
"""Evaluate retrieval after filtering candidates by predicted content type."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


TEXT_TYPES = {"Table", "Figure", "Section", "Line"}
DEFAULT_CUTOFFS = (1, 2, 10)
SDR_PATH = "data/{dataset_name}/{paper_id}/v1/paper.sdr.json"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
    return records


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def record_id(dataset_name: str, paper_id: str, query_idx: str) -> str:
    return f"{dataset_name}::{paper_id}::{query_idx}"


def parse_prediction_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected MODEL=PATH")
    model, path = value.split("=", 1)
    if not model or not path:
        raise argparse.ArgumentTypeError("Expected MODEL=PATH")
    return model, Path(path)


def normalize_prediction(labels: Any, label_mode: str) -> set[str]:
    if not isinstance(labels, list):
        return set()
    allowed = {"figure", "table", "text"} if label_mode == "text3" else {"figure", "table", "line", "section"}
    normalized = set()
    for label in labels:
        label = str(label).strip().lower()
        if label_mode == "text3" and label in {"line", "section"}:
            normalized.add("text")
        elif label in allowed:
            normalized.add(label)
    return normalized


def load_predictions(path: Path, label_mode: str) -> dict[str, set[str]]:
    predictions = {}
    for record in read_jsonl(path):
        predictions[str(record["id"])] = normalize_prediction(record.get("prediction", []), label_mode)
    return predictions


def hit_rate(retrieved: list[str], gold: list[str], cutoff: int) -> float:
    return 1.0 if set(retrieved[:cutoff]) & set(gold) else 0.0


def recall_score(retrieved: list[str], gold: list[str], cutoff: int) -> float:
    if not gold:
        return 0.0
    return len(set(retrieved[:cutoff]) & set(gold)) / len(set(gold))


def mrr_score(retrieved: list[str], gold: list[str]) -> float:
    gold_set = set(gold)
    for idx, doc in enumerate(retrieved):
        if doc in gold_set:
            return 1.0 / (idx + 1)
    return 0.0


def load_sdr(paper_id: str, dataset_name: str) -> dict[str, Any] | None:
    path = PROJECT_ROOT / SDR_PATH.format(dataset_name=dataset_name, paper_id=paper_id)
    try:
        return read_json(path)
    except Exception:
        return None


def expand_line_gold(sdr: dict[str, Any], gold_label: str) -> list[str]:
    label = gold_label.replace("Line", "").strip()
    if "-" in label:
        start, end = map(int, label.split("-"))
        gold_lines = list(range(start, end + 1))
    else:
        gold_lines = [int(label)]
    return [f"Paragraph {pid}" for pid in get_paragraph_ids_by_line(sdr, gold_lines)]


def get_paragraph_ids_by_line(sdr: dict[str, Any], gold_lines: list[int]) -> list[str]:
    target_lines = set(gold_lines)
    matched = set()
    for para in sdr.get("paragraphs", []):
        pid = para.get("paragraph_id")
        if pid is None:
            continue
        try:
            start_line = int(para["start_line"])
            end_line = int(para["end_line"])
        except (KeyError, TypeError, ValueError):
            continue
        if start_line == -1 or end_line == -1:
            continue
        if any(start_line <= line <= end_line for line in target_lines):
            matched.add(str(pid))
    return sorted(matched, key=lambda value: int(value) if value.isdigit() else value)


def expand_gold_label(sdr: dict[str, Any], gold_label: str, ref_type: str) -> list[str]:
    if ref_type.lower() == "line":
        return expand_line_gold(sdr, gold_label)
    return [gold_label]


def candidate_type(candidate: str) -> str | None:
    if candidate.startswith("Figure "):
        return "figure"
    if candidate.startswith("Table "):
        return "table"
    if candidate.startswith("Section "):
        return "section"
    if candidate.startswith("Paragraph "):
        return "line"
    if candidate.startswith("Line "):
        return "raw_line"
    return None


def alternative_candidates(scores: dict[str, Any]) -> list[str]:
    return [candidate for candidate in scores.keys() if not candidate.startswith("Line ")]


def filter_by_predicted_types(scores: dict[str, Any], predicted_types: set[str]) -> list[str]:
    retrieved = []
    for candidate in alternative_candidates(scores):
        ctype = candidate_type(candidate)
        if ctype in predicted_types or ("text" in predicted_types and ctype in {"line", "section"}):
            retrieved.append(candidate)
    return retrieved


def expanded_gold_references(dataset_name: str, paper_id: str, item: dict[str, Any]) -> list[list[str]] | None:
    sdr = load_sdr(paper_id, dataset_name)
    if sdr is None:
        return None

    expanded_refs = []
    for ref in item.get("gold", []) or []:
        ref_type = str(ref.get("type", "")).strip().capitalize()
        if ref_type not in TEXT_TYPES:
            continue
        label = str(ref.get("label", "")).strip()
        if not label:
            continue
        gold_label = f"{ref_type} {label}"
        try:
            expanded = expand_gold_label(sdr, gold_label, ref_type)
        except Exception:
            continue
        if expanded:
            expanded_refs.append(expanded)
    return expanded_refs or None


def expanded_gold_labels(dataset_name: str, paper_id: str, item: dict[str, Any]) -> list[str] | None:
    expanded_refs = expanded_gold_references(dataset_name, paper_id, item)
    if not expanded_refs:
        return None
    expanded = []
    for ref_labels in expanded_refs:
        expanded.extend(ref_labels)
    return expanded or None


def new_accumulator(cutoffs: list[int]) -> dict[str, Any]:
    return {
        "n": 0,
        "hits": {str(cutoff): [] for cutoff in cutoffs},
        "recalls": {str(cutoff): [] for cutoff in cutoffs},
        "mrrs": [],
        "empty_type_predictions": 0,
        "avg_predicted_types": [],
        "avg_candidates": [],
    }


def add_example(acc: dict[str, Any], retrieved: list[str], gold: list[str], cutoffs: list[int], predicted_types: set[str] | None = None) -> None:
    acc["n"] += 1
    for cutoff in cutoffs:
        acc["hits"][str(cutoff)].append(hit_rate(retrieved, gold, cutoff))
        acc["recalls"][str(cutoff)].append(recall_score(retrieved, gold, cutoff))
    acc["mrrs"].append(mrr_score(retrieved, gold))
    acc["avg_candidates"].append(len(retrieved))
    if predicted_types is not None:
        acc["avg_predicted_types"].append(len(predicted_types))
        if not predicted_types:
            acc["empty_type_predictions"] += 1


def finalize(acc: dict[str, Any]) -> dict[str, Any]:
    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        "n": acc["n"],
        "hit_rate": {cutoff: mean(values) for cutoff, values in acc["hits"].items()},
        "recall": {cutoff: mean(values) for cutoff, values in acc["recalls"].items()},
        "mrr": mean(acc["mrrs"]),
        "empty_type_predictions": acc["empty_type_predictions"],
        "avg_predicted_types": mean(acc["avg_predicted_types"]),
        "avg_candidates": mean(acc["avg_candidates"]),
    }


def evaluate_model(
    *,
    results_dir: Path,
    model: str,
    predictions: dict[str, set[str]],
    cutoffs: list[int],
    shared_ids: set[str] | None,
    reference_mode: str,
) -> dict[str, Any]:
    baseline = new_accumulator(cutoffs)
    filtered = new_accumulator(cutoffs)
    by_dataset = defaultdict(lambda: {"baseline": new_accumulator(cutoffs), "filtered": new_accumulator(cutoffs)})
    skipped = defaultdict(int)

    for dataset_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        path = dataset_dir / f"{model}.json"
        if not path.exists():
            continue
        data = read_json(path)
        for paper_id, queries in data.items():
            for query_idx, item in queries.items():
                rid = record_id(dataset_dir.name, paper_id, str(query_idx))
                if shared_ids is not None and rid not in shared_ids:
                    skipped["outside_shared_subset"] += 1
                    continue
                if rid not in predictions:
                    skipped["missing_prediction"] += 1
                    continue
                scores = item.get("scores") or {}
                if not scores:
                    skipped["empty_scores"] += 1
                    continue
                gold_refs = expanded_gold_references(dataset_dir.name, paper_id, item)
                if not gold_refs:
                    skipped["missing_or_unexpandable_gold"] += 1
                    continue

                predicted_types = predictions[rid]
                baseline_retrieved = alternative_candidates(scores)
                filtered_retrieved = filter_by_predicted_types(scores, predicted_types)

                if reference_mode == "query":
                    gold_units = [[label for ref_labels in gold_refs for label in ref_labels]]
                else:
                    gold_units = gold_refs

                for gold in gold_units:
                    add_example(baseline, baseline_retrieved, gold, cutoffs)
                    add_example(filtered, filtered_retrieved, gold, cutoffs, predicted_types)
                    add_example(by_dataset[dataset_dir.name]["baseline"], baseline_retrieved, gold, cutoffs)
                    add_example(by_dataset[dataset_dir.name]["filtered"], filtered_retrieved, gold, cutoffs, predicted_types)

    return {
        "model": model,
        "baseline": finalize(baseline),
        "type_filtered": finalize(filtered),
        "by_dataset": {
            dataset: {
                "baseline": finalize(parts["baseline"]),
                "type_filtered": finalize(parts["filtered"]),
            }
            for dataset, parts in sorted(by_dataset.items())
        },
        "skipped": dict(skipped),
    }


def print_report(title: str, result: dict[str, Any], cutoffs: list[int]) -> None:
    print(f"\n{title}: {result['model']}")
    for name in ("baseline", "type_filtered"):
        metrics = result[name]
        hits = " ".join(f"hit@{k}={metrics['hit_rate'][str(k)]:.4f}" for k in cutoffs)
        recalls = " ".join(f"recall@{k}={metrics['recall'][str(k)]:.4f}" for k in cutoffs)
        print(
            f"  {name:13s} n={metrics['n']} {hits} {recalls} mrr={metrics['mrr']:.4f} "
            f"avg_candidates={metrics['avg_candidates']:.1f}"
        )
    if result["type_filtered"]["n"]:
        tf = result["type_filtered"]
        print(
            f"  type preds    avg_types={tf['avg_predicted_types']:.2f} "
            f"empty={tf['empty_type_predictions']}"
        )
    if result["skipped"]:
        print(f"  skipped       {result['skipped']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--prediction",
        type=parse_prediction_arg,
        action="append",
        required=True,
        help="MODEL=prediction_jsonl. Repeat once per retrieval model.",
    )
    parser.add_argument("--cutoffs", type=int, nargs="+", default=list(DEFAULT_CUTOFFS))
    parser.add_argument("--label-mode", choices=("four", "text3"), default="four")
    parser.add_argument("--shared-ids", type=Path, default=None)
    parser.add_argument(
        "--reference-mode",
        choices=("query", "reference"),
        default="reference",
        help="reference evaluates each gold reference separately; query matches the original pooled-gold metric.",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    shared_ids = set(read_json(args.shared_ids)) if args.shared_ids else None
    all_results = {"native": {}, "shared_subset": {}}

    for model, path in args.prediction:
        predictions = load_predictions(path, args.label_mode)
        native = evaluate_model(
            results_dir=args.results_dir,
            model=model,
            predictions=predictions,
            cutoffs=args.cutoffs,
            shared_ids=None,
            reference_mode=args.reference_mode,
        )
        all_results["native"][model] = native
        print_report("native", native, args.cutoffs)

        if shared_ids is not None:
            shared = evaluate_model(
                results_dir=args.results_dir,
                model=model,
                predictions=predictions,
                cutoffs=args.cutoffs,
                shared_ids=shared_ids,
                reference_mode=args.reference_mode,
            )
            all_results["shared_subset"][model] = shared
            print_report("shared_subset", shared, args.cutoffs)

    if args.json_out:
        write_json(args.json_out, all_results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
