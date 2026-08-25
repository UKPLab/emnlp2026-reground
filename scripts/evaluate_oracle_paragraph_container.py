#!/usr/bin/env python3
"""Compare exact Paragraph retrieval with descendant-based relaxation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from evaluate_ancestor_relaxed import MODELS, line_numbers, paragraphs_for_lines


CUTOFFS = (1, 2, 10)


def exact_paragraph_sets(item: dict[str, Any], sdr: dict[str, Any]) -> list[set[str]]:
    accepted = []
    for reference in item.get("gold", []) or []:
        if str(reference.get("type", "")).strip().lower() != "line":
            continue
        try:
            paragraphs = paragraphs_for_lines(
                sdr, line_numbers(str(reference.get("label", "")))
            )
        except ValueError:
            continue
        accepted.extend({paragraph} for paragraph in paragraphs)
    return deduplicate(accepted)


def paragraphs_in_section_tree(sdr: dict[str, Any], gold_number: str) -> set[str]:
    """Return paragraphs in the gold section or any of its subsections."""
    intervals = []
    for section in sdr.get("sections", []):
        number = str(section.get("number", ""))
        if number != gold_number and not number.startswith(f"{gold_number}."):
            continue
        try:
            start, end = int(section["start_line"]), int(section["end_line"])
        except (KeyError, TypeError, ValueError):
            continue
        if start >= 0 and end >= 0:
            intervals.append((start, end))

    paragraphs = set()
    for paragraph in sdr.get("paragraphs", []):
        try:
            start, end = int(paragraph["start_line"]), int(paragraph["end_line"])
        except (KeyError, TypeError, ValueError):
            continue
        if any(section_start <= start and end <= section_end for section_start, section_end in intervals):
            paragraphs.add(f"Paragraph {paragraph['paragraph_id']}")
    return paragraphs


def relaxed_paragraph_sets(item: dict[str, Any], sdr: dict[str, Any]) -> list[set[str]]:
    accepted = exact_paragraph_sets(item, sdr)
    for reference in item.get("gold", []) or []:
        if str(reference.get("type", "")).strip().lower() != "section":
            continue
        label = str(reference.get("label", "")).strip()
        if not label:
            continue
        descendants = paragraphs_in_section_tree(sdr, label)
        if descendants:
            accepted.append(descendants)
    return deduplicate(accepted)


def paired_paragraph_sets(
    item: dict[str, Any], sdr: dict[str, Any]
) -> tuple[list[set[str]], list[set[str]]]:
    """Build exact/relaxed alternatives over the same gold references.

    A section reference has no exact Paragraph match (empty set), but under the
    relaxation any paragraph in that section tree is acceptable. Keeping that
    empty exact slot preserves the same denominator in both conditions.
    """
    exact_sets: list[set[str]] = []
    relaxed_sets: list[set[str]] = []
    seen_references = set()
    for reference in item.get("gold", []) or []:
        ref_type = str(reference.get("type", "")).strip().lower()
        label = str(reference.get("label", "")).strip()
        key = (ref_type, label)
        if not label or key in seen_references or ref_type not in {"line", "section"}:
            continue
        seen_references.add(key)
        if ref_type == "line":
            try:
                paragraphs = paragraphs_for_lines(sdr, line_numbers(label))
            except ValueError:
                continue
            for paragraph in sorted(paragraphs):
                exact_sets.append({paragraph})
                relaxed_sets.append({paragraph})
        else:
            descendants = paragraphs_in_section_tree(sdr, label)
            if descendants:
                exact_sets.append(set())
                relaxed_sets.append(descendants)
    return exact_sets, relaxed_sets


def deduplicate(sets: list[set[str]]) -> list[set[str]]:
    unique, seen = [], set()
    for alternatives in sets:
        key = tuple(sorted(alternatives))
        if key not in seen:
            unique.append(alternatives)
            seen.add(key)
    return unique


def add_scores(accumulator: dict[str, Any], retrieved: list[str], gold_sets: list[set[str]]) -> None:
    for cutoff in CUTOFFS:
        top_k = set(retrieved[:cutoff])
        value = sum(bool(top_k & gold) for gold in gold_sets) / len(gold_sets)
        accumulator["recall"][cutoff].append(value)
    rr = 0.0
    for rank, prediction in enumerate(retrieved, start=1):
        if any(prediction in gold for gold in gold_sets):
            rr = 1.0 / rank
            break
    accumulator["mrr"].append(rr)


def evaluate(results_root: Path, data_root: Path) -> dict[str, Any]:
    accumulators = {
        model: {
            mode: {"recall": {cutoff: [] for cutoff in CUTOFFS}, "mrr": []}
            for mode in ("exact", "descendant_relaxed")
        }
        for model in MODELS
    }
    for venue_dir in sorted(path for path in results_root.iterdir() if path.is_dir()):
        venue = venue_dir.name
        if venue.startswith("retrieval_"):
            continue
        for model in MODELS:
            path = venue_dir / f"{model}.json"
            if not path.exists():
                continue
            results = json.loads(path.read_text())
            for paper_id, queries in results.items():
                sdr_path = data_root / venue / paper_id / "v1" / "paper.sdr.json"
                if not sdr_path.exists():
                    continue
                sdr = json.loads(sdr_path.read_text())
                for item in queries.values():
                    if not item.get("scores"):
                        continue
                    retrieved = [
                        candidate.strip()
                        for candidate in item["scores"]
                        if candidate.startswith("Paragraph ")
                    ]
                    exact, relaxed = paired_paragraph_sets(item, sdr)
                    if relaxed:
                        add_scores(accumulators[model]["exact"], retrieved, exact)
                        add_scores(
                            accumulators[model]["descendant_relaxed"], retrieved, relaxed
                        )

    output = {"models": {}}
    for model, modes in accumulators.items():
        output["models"][model] = {}
        for mode, values in modes.items():
            output["models"][model][mode] = {
                "n": len(values["mrr"]),
                "mrr": mean(values["mrr"]) if values["mrr"] else 0.0,
                "recall": {
                    str(cutoff): mean(scores) if scores else 0.0
                    for cutoff, scores in values["recall"].items()
                },
            }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument(
        "--output", type=Path, default=Path("oracle_paragraph_container_metrics.json")
    )
    args = parser.parse_args()
    output = evaluate(args.results, args.data)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print("model\tmode\tn\tMRR\tR@1\tR@2\tR@10")
    for model, modes in output["models"].items():
        for mode, values in modes.items():
            print(
                f"{model}\t{mode}\t{values['n']}\t{100 * values['mrr']:.2f}\t"
                f"{100 * values['recall']['1']:.2f}\t{100 * values['recall']['2']:.2f}\t"
                f"{100 * values['recall']['10']:.2f}"
            )


if __name__ == "__main__":
    main()
