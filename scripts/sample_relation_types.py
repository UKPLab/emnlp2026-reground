#!/usr/bin/env python3
"""Create a deterministic sample of reference-bearing review/rebuttal pairs."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import find_reference_mentions, get_reference_patterns  # noqa: E402


def collect(data: dict[str, Any]) -> list[dict[str, str]]:
    patterns = get_reference_patterns()
    records: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str, str]] = set()

    for conference, papers in data.items():
        for paper_id, rebuttals in papers.items():
            if not isinstance(rebuttals, dict):
                continue
            for review_id, rebuttal in rebuttals.items():
                if not isinstance(rebuttal, dict):
                    continue
                for alignment in rebuttal.get("aligned", []):
                    comment = alignment.get("review_spans", "").strip()
                    response = alignment.get("rebuttal_sentence", "").strip()
                    if not comment or not response or len(response.split()) <= 6:
                        continue
                    mentions = find_reference_mentions(response, patterns)
                    if not mentions:
                        continue
                    key = (conference, paper_id, review_id, comment, response)
                    if key in seen:
                        continue
                    seen.add(key)
                    records.append(
                        {
                            "conference": conference,
                            "paper_id": paper_id,
                            "review_id": review_id,
                            "review_comment": comment,
                            "rebuttal_sentence": response,
                            "paper_references": " | ".join(
                                f"{mention.kind}: {mention.text}" for mention in mentions
                            ),
                            "manual_relation": "",
                            "manual_notes": "",
                        }
                    )
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "reground.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "relation_type_samples" / "relation_types_200.csv",
    )
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    records = collect(json.loads(args.input.read_text(encoding="utf-8")))
    if len(records) < args.sample_size:
        raise SystemExit(f"Only {len(records)} eligible records; requested {args.sample_size}")
    sample = random.Random(args.seed).sample(records, args.sample_size)
    for index, record in enumerate(sample, start=1):
        record["sample_id"] = f"REL-{index:03d}"

    fields = [
        "sample_id",
        "conference",
        "paper_id",
        "review_id",
        "review_comment",
        "rebuttal_sentence",
        "paper_references",
        "manual_relation",
        "manual_notes",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sample)
    print(f"Sampled {len(sample)} of {len(records)} eligible pairs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
