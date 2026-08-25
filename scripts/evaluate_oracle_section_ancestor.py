#!/usr/bin/env python3
"""Evaluate exact and ancestor-relaxed retrieval in the oracle Section pool."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from evaluate_ancestor_relaxed import MODELS, canonical, section_ancestors


CUTOFFS = (1, 2)


def section_golds(item: dict[str, Any]) -> list[str]:
    golds = []
    for reference in item.get("gold", []) or []:
        if str(reference.get("type", "")).strip().lower() != "section":
            continue
        label = str(reference.get("label", "")).strip()
        if label:
            golds.append(f"Section {label}")
    return list(dict.fromkeys(golds))


def evaluate(results_root: Path, data_root: Path) -> dict[str, Any]:
    accumulators = {
        model: {
            mode: {"recall": {cutoff: [] for cutoff in CUTOFFS}, "mrr": []}
            for mode in ("exact", "ancestor_relaxed")
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
                valid_sections = {
                    str(section.get("number")) for section in sdr.get("sections", [])
                }
                for item in queries.values():
                    golds = section_golds(item)
                    if not golds or not item.get("scores"):
                        continue
                    retrieved = [
                        canonical(candidate)
                        for candidate in item["scores"]
                        if candidate.startswith("Section ")
                    ]
                    accepted = {
                        "exact": [{gold} for gold in golds],
                        "ancestor_relaxed": [
                            {gold}
                            | section_ancestors(gold.removeprefix("Section "), valid_sections)
                            for gold in golds
                        ],
                    }
                    for mode, gold_sets in accepted.items():
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
        "definition": "oracle Section candidate pool; exact gold or strict parent section",
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
        "--output", type=Path, default=Path("oracle_section_ancestor_metrics.json")
    )
    args = parser.parse_args()
    output = evaluate(args.results, args.data)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print("model\tn\tMRR_exact\tMRR_ancestor\tR@1_exact\tR@1_ancestor\tR@2_exact\tR@2_ancestor")
    for model, modes in output["models"].items():
        exact, relaxed = modes["exact"], modes["ancestor_relaxed"]
        print(
            "\t".join(
                (
                    model,
                    str(exact["n"]),
                    f"{100 * exact['mrr']:.2f}",
                    f"{100 * relaxed['mrr']:.2f}",
                    f"{100 * exact['recall']['1']:.2f}",
                    f"{100 * relaxed['recall']['1']:.2f}",
                    f"{100 * exact['recall']['2']:.2f}",
                    f"{100 * relaxed['recall']['2']:.2f}",
                )
            )
        )


if __name__ == "__main__":
    main()
