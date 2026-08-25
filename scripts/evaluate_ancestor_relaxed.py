#!/usr/bin/env python3
"""Evaluate exact and ancestor-only relaxed retrieval on stored model rankings.

The relaxed metric counts a gold unit as recovered when the top-k contains either
the exact unit or one of its ancestors. Descendants and siblings receive no credit.
Line references are resolved to their containing paragraphs, as in the paper.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


MODELS = ("gemma300", "qwen3-30b-instruct", "GPT-5.1")
CUTOFFS = (1, 2, 10)
SECTION_RE = re.compile(r"^Section\s+([^\s/]+)")
PARAGRAPH_RE = re.compile(r"^Paragraph\s+(.+?)\s*$")


def canonical(candidate: str) -> str:
    match = SECTION_RE.match(candidate)
    if match:
        return f"Section {match.group(1)}"
    return candidate.strip()


def line_numbers(label: str) -> list[int]:
    normalized = label.strip().replace("–", "-").replace("—", "-")
    if "-" not in normalized:
        return [int(normalized)]
    start, end = (int(part.strip()) for part in normalized.split("-", 1))
    return list(range(start, end + 1))


def paragraphs_for_lines(sdr: dict[str, Any], lines: list[int]) -> set[str]:
    targets = set(lines)
    matches = set()
    for paragraph in sdr.get("paragraphs", []):
        try:
            start = int(paragraph["start_line"])
            end = int(paragraph["end_line"])
        except (KeyError, TypeError, ValueError):
            continue
        if start < 0 or end < 0 or not any(start <= line <= end for line in targets):
            continue
        matches.add(f"Paragraph {paragraph['paragraph_id']}")
    return matches


def section_ancestors(number: str, valid_sections: set[str]) -> set[str]:
    """Return strict section ancestors based on hierarchical section numbers."""
    parts = number.split(".")
    ancestors = set()
    for length in range(1, len(parts)):
        parent = ".".join(parts[:length])
        if parent in valid_sections:
            ancestors.add(f"Section {parent}")
    return ancestors


def paragraph_ancestors(sdr: dict[str, Any], paragraph_id: str) -> set[str]:
    paragraph = next(
        (p for p in sdr.get("paragraphs", []) if str(p.get("paragraph_id")) == paragraph_id),
        None,
    )
    if paragraph is None:
        return set()

    try:
        start, end = int(paragraph["start_line"]), int(paragraph["end_line"])
    except (KeyError, TypeError, ValueError):
        return set()

    ancestors = set()
    for section in sdr.get("sections", []):
        try:
            section_start = int(section["start_line"])
            section_end = int(section["end_line"])
        except (KeyError, TypeError, ValueError):
            continue
        # Some extracted sections have an unknown end (-1); do not infer containment.
        if section_start >= 0 and section_end >= 0 and section_start <= start and end <= section_end:
            ancestors.add(f"Section {section['number']}")
    return ancestors


def gold_units(item: dict[str, Any], sdr: dict[str, Any]) -> list[set[str]]:
    """Return one singleton set per resolved gold evidence unit."""
    units: list[set[str]] = []
    for reference in item.get("gold", []) or []:
        ref_type = str(reference.get("type", "")).strip().capitalize()
        label = str(reference.get("label", "")).strip()
        if not label or ref_type not in {"Line", "Section", "Figure", "Table"}:
            continue
        if ref_type == "Line":
            try:
                resolved = paragraphs_for_lines(sdr, line_numbers(label))
            except ValueError:
                continue
        else:
            resolved = {f"{ref_type} {label}"}
        units.extend({canonical(unit)} for unit in resolved)

    # Recall uses a set of gold evidence units, so repeated references do not
    # increase the denominator.
    unique = []
    seen = set()
    for unit in units:
        key = next(iter(unit))
        if key not in seen:
            unique.append(unit)
            seen.add(key)
    return unique


def accepted_predictions(
    alternatives: set[str], sdr: dict[str, Any], relaxed: bool
) -> set[str]:
    accepted = set(alternatives)
    if not relaxed:
        return accepted

    valid_sections = {str(section.get("number")) for section in sdr.get("sections", [])}
    for unit in alternatives:
        section_match = SECTION_RE.match(unit)
        paragraph_match = PARAGRAPH_RE.match(unit)
        if section_match:
            accepted.update(section_ancestors(section_match.group(1), valid_sections))
        elif paragraph_match:
            accepted.update(paragraph_ancestors(sdr, paragraph_match.group(1)))
    return accepted


def score_query(
    retrieved: list[str], gold: list[set[str]], sdr: dict[str, Any], relaxed: bool
) -> tuple[dict[int, float], float, int]:
    accepted = [accepted_predictions(unit, sdr, relaxed) for unit in gold]
    recalls = {}
    for cutoff in CUTOFFS:
        top_k = set(retrieved[:cutoff])
        recalls[cutoff] = sum(bool(top_k & unit) for unit in accepted) / len(accepted)

    reciprocal_rank = 0.0
    for rank, prediction in enumerate(retrieved, start=1):
        if any(prediction in unit for unit in accepted):
            reciprocal_rank = 1.0 / rank
            break
    relaxed_only = sum(bool(unit - exact) for unit, exact in zip(accepted, gold))
    return recalls, reciprocal_rank, relaxed_only


def evaluate(results_root: Path, data_root: Path) -> dict[str, Any]:
    raw: dict[str, dict[str, dict[str, list[float] | int]]] = defaultdict(dict)

    for venue_dir in sorted(path for path in results_root.iterdir() if path.is_dir()):
        venue = venue_dir.name
        if venue.startswith("retrieval_"):
            continue
        for model in MODELS:
            result_path = venue_dir / f"{model}.json"
            if not result_path.exists():
                continue
            results = json.loads(result_path.read_text())
            accumulator: dict[str, Any] = {
                mode: {"n": 0, "recall": {k: [] for k in CUTOFFS}, "mrr": [], "eligible_gold": 0}
                for mode in ("legacy_exact", "corrected_exact", "ancestor_relaxed")
            }
            for paper_id, queries in results.items():
                sdr_path = data_root / venue / paper_id / "v1" / "paper.sdr.json"
                if not sdr_path.exists():
                    continue
                sdr = json.loads(sdr_path.read_text())
                for item in queries.values():
                    if not item.get("scores"):
                        continue
                    gold = gold_units(item, sdr)
                    if not gold:
                        continue
                    retrieved_raw = [
                        candidate.strip()
                        for candidate in item["scores"]
                        if not candidate.startswith("Line ")
                    ]
                    retrieved_canonical = [canonical(candidate) for candidate in retrieved_raw]
                    for mode, retrieved, relaxed in (
                        ("legacy_exact", retrieved_raw, False),
                        ("corrected_exact", retrieved_canonical, False),
                        ("ancestor_relaxed", retrieved_canonical, True),
                    ):
                        recalls, mrr, eligible = score_query(retrieved, gold, sdr, relaxed)
                        accumulator[mode]["n"] += 1
                        accumulator[mode]["mrr"].append(mrr)
                        accumulator[mode]["eligible_gold"] += eligible
                        for cutoff, value in recalls.items():
                            accumulator[mode]["recall"][cutoff].append(value)
            raw[venue][model] = accumulator

    output: dict[str, Any] = {"definition": "exact match or retrieved strict ancestor of gold; descendants and siblings receive no credit", "cutoffs": list(CUTOFFS), "by_venue": {}, "overall": {}}
    overall_values: dict[str, dict[str, dict[str, list[float] | int]]] = defaultdict(
        lambda: {
            mode: {"n": 0, "recall": {str(k): [] for k in CUTOFFS}, "mrr": [], "eligible_gold": 0}
            for mode in ("legacy_exact", "corrected_exact", "ancestor_relaxed")
        }
    )
    for venue, models in raw.items():
        output["by_venue"][venue] = {}
        for model, modes in models.items():
            output["by_venue"][venue][model] = {}
            for mode, values in modes.items():
                summary = {
                    "n": values["n"],
                    "recall": {str(k): mean(v) if v else 0.0 for k, v in values["recall"].items()},
                    "mrr": mean(values["mrr"]) if values["mrr"] else 0.0,
                    "ancestor_eligible_gold_references": values["eligible_gold"],
                }
                output["by_venue"][venue][model][mode] = summary
                target = overall_values[model][mode]
                target["n"] += values["n"]
                target["mrr"].extend(values["mrr"])
                target["eligible_gold"] += values["eligible_gold"]
                for k, scores in values["recall"].items():
                    target["recall"][str(k)].extend(scores)

    for model, modes in overall_values.items():
        output["overall"][model] = {}
        for mode, values in modes.items():
            output["overall"][model][mode] = {
                "n": values["n"],
                "recall": {k: mean(v) if v else 0.0 for k, v in values["recall"].items()},
                "mrr": mean(values["mrr"]) if values["mrr"] else 0.0,
                "ancestor_eligible_gold_references": values["eligible_gold"],
            }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("ancestor_relaxed_metrics.json"))
    args = parser.parse_args()
    output = evaluate(args.results, args.data)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    columns = ["model", "n"]
    for cutoff in CUTOFFS:
        columns.extend(
            f"R@{cutoff}_{mode}"
            for mode in ("legacy", "corrected", "ancestor")
        )
    columns.extend(("MRR_legacy", "MRR_corrected", "MRR_ancestor"))
    print("\t".join(columns))
    for model, modes in output["overall"].items():
        legacy, exact, relaxed = modes["legacy_exact"], modes["corrected_exact"], modes["ancestor_relaxed"]
        values = [model, str(exact["n"])]
        for cutoff in CUTOFFS:
            key = str(cutoff)
            values.extend([
                f"{100 * legacy['recall'][key]:.2f}",
                f"{100 * exact['recall'][key]:.2f}",
                f"{100 * relaxed['recall'][key]:.2f}",
            ])
        values.extend([
            f"{100 * legacy['mrr']:.2f}",
            f"{100 * exact['mrr']:.2f}",
            f"{100 * relaxed['mrr']:.2f}",
        ])
        print("\t".join(values))


if __name__ == "__main__":
    main()
