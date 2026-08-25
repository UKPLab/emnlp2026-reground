#!/usr/bin/env python3
"""Oracle Section retrieval with sections treated as containers of gold text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from evaluate_ancestor_relaxed import (
    MODELS,
    canonical,
    line_numbers,
    paragraph_ancestors,
    paragraphs_for_lines,
    section_ancestors,
)


CUTOFFS = (1, 2)


def acceptable_section_sets(item: dict[str, Any], sdr: dict[str, Any]) -> list[set[str]]:
    """Map each gold section/paragraph unit to acceptable containing sections."""
    valid_sections = {str(section.get("number")) for section in sdr.get("sections", [])}
    accepted: list[set[str]] = []
    for reference in item.get("gold", []) or []:
        ref_type = str(reference.get("type", "")).strip().lower()
        label = str(reference.get("label", "")).strip()
        if not label:
            continue
        if ref_type == "section":
            exact = f"Section {label}"
            alternatives = {exact} | section_ancestors(label, valid_sections)
            accepted.append(alternatives)
        elif ref_type == "line":
            try:
                paragraphs = paragraphs_for_lines(sdr, line_numbers(label))
            except ValueError:
                continue
            for paragraph in paragraphs:
                paragraph_id = paragraph.removeprefix("Paragraph ")
                alternatives = paragraph_ancestors(sdr, paragraph_id)
                if alternatives:
                    accepted.append(alternatives)

    # Repeated references to the same unit should not inflate the denominator.
    unique = []
    seen = set()
    for alternatives in accepted:
        key = tuple(sorted(alternatives))
        if key not in seen:
            unique.append(alternatives)
            seen.add(key)
    return unique


def paired_section_sets(
    item: dict[str, Any], sdr: dict[str, Any]
) -> tuple[list[set[str]], list[set[str]]]:
    """Build exact/relaxed alternatives over an identical gold denominator."""
    valid_sections = {str(section.get("number")) for section in sdr.get("sections", [])}
    exact_sets: list[set[str]] = []
    relaxed_sets: list[set[str]] = []
    seen_references = set()
    for reference in item.get("gold", []) or []:
        ref_type = str(reference.get("type", "")).strip().lower()
        label = str(reference.get("label", "")).strip()
        key = (ref_type, label)
        if not label or key in seen_references or ref_type not in {"section", "line"}:
            continue
        seen_references.add(key)
        if ref_type == "section":
            exact = {f"Section {label}"}
            exact_sets.append(exact)
            relaxed_sets.append(exact | section_ancestors(label, valid_sections))
        else:
            try:
                paragraphs = paragraphs_for_lines(sdr, line_numbers(label))
            except ValueError:
                continue
            for paragraph in sorted(paragraphs):
                paragraph_id = paragraph.removeprefix("Paragraph ")
                ancestors = paragraph_ancestors(sdr, paragraph_id)
                if ancestors:
                    exact_sets.append(set())
                    relaxed_sets.append(ancestors)
    return exact_sets, relaxed_sets


def evaluate(results_root: Path, data_root: Path) -> dict[str, Any]:
    accumulators = {
        model: {
            mode: {"recall": {cutoff: [] for cutoff in CUTOFFS}, "mrr": []}
            for mode in ("exact", "container_relaxed")
        }
        for model in MODELS
    }
    for venue_dir in sorted(path for path in results_root.iterdir() if path.is_dir()):
        venue = venue_dir.name
        if venue.startswith("retrieval_"):
            continue
        for model in MODELS:
            result_path = venue_dir / f"{model}.json"
            if not result_path.exists():
                continue
            results = json.loads(result_path.read_text())
            for paper_id, queries in results.items():
                sdr_path = data_root / venue / paper_id / "v1" / "paper.sdr.json"
                if not sdr_path.exists():
                    continue
                sdr = json.loads(sdr_path.read_text())
                for item in queries.values():
                    if not item.get("scores"):
                        continue
                    exact_sets, relaxed_sets = paired_section_sets(item, sdr)
                    if not relaxed_sets:
                        continue
                    retrieved = [
                        canonical(candidate)
                        for candidate in item["scores"]
                        if candidate.startswith("Section ")
                    ]
                    for mode, gold_sets in (
                        ("exact", exact_sets),
                        ("container_relaxed", relaxed_sets),
                    ):
                        for cutoff in CUTOFFS:
                            top_k = set(retrieved[:cutoff])
                            recall = sum(bool(top_k & gold) for gold in gold_sets) / len(gold_sets)
                            accumulators[model][mode]["recall"][cutoff].append(recall)
                        rr = 0.0
                        for rank, prediction in enumerate(retrieved, start=1):
                            if any(prediction in gold for gold in gold_sets):
                                rr = 1.0 / rank
                                break
                        accumulators[model][mode]["mrr"].append(rr)

    output = {
        "definition": (
            "oracle Section pool; a gold section, a strict ancestor of a gold "
            "subsection, or a section containing a gold paragraph is relevant"
        ),
        "models": {},
    }
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
        "--output", type=Path, default=Path("oracle_section_container_metrics.json")
    )
    args = parser.parse_args()
    output = evaluate(args.results, args.data)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print("model\tmode\tn\tMRR\tR@1\tR@2")
    for model, modes in output["models"].items():
        for mode, values in modes.items():
            print(
                f"{model}\t{mode}\t{values['n']}\t{100 * values['mrr']:.2f}\t"
                f"{100 * values['recall']['1']:.2f}\t{100 * values['recall']['2']:.2f}"
            )


if __name__ == "__main__":
    main()
