#!/usr/bin/env python3
"""Compute evidence-unit and query-level evidence-combination statistics.

The source dataset stores one or more aligned rebuttal sentences for each
review span.  This script detects the explicit paper references in those
sentences, aggregates them by review query, and reports both link and unique
unit counts.  Unit identity is paper-local, so unique units are keyed by
conference, paper ID, reference type, and normalized reference text.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import find_reference_mentions, get_reference_patterns  # noqa: E402


TYPE_NAMES = {
    "line": "paragraph",
    "section": "section",
    "figure": "figure",
    "table": "table",
    "appendix": "appendix",
    "equation": "equation",
    "page": "page",
    "footnote": "footnote",
}
MAIN_TYPES = {"paragraph", "section", "figure", "table"}


def normalize_reference(kind: str, text: str) -> str:
    """Normalize superficial variants while retaining the referenced ID."""
    normalized = text.casefold().strip()
    normalized = normalized.replace("–", "-").replace("—", "-")
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(
        r"^(?:lines?|l\.?|sections?|secs?\.?|§|tables?|tabs?\.?|"
        r"figures?|figs?\.?|appendix|equations?|eqs?\.?|pages?|pp?\.?|"
        r"footnote)\s*",
        "",
        normalized,
    )
    normalized = normalized.strip(" ()[]{}.,:;")
    if kind == "section" and normalized.endswith(" section"):
        normalized = normalized[: -len(" section")].strip()
    return normalized


def load_queries(data: dict[str, Any]) -> dict[tuple[str, str, str], list[str]]:
    """Aggregate aligned rebuttal sentences by (conference, paper, review)."""
    queries: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for conference, papers in data.items():
        for paper_id, rebuttals in papers.items():
            for rebuttal in rebuttals.values():
                for alignment in rebuttal.get("aligned", []):
                    review = alignment.get("review_spans", "").strip()
                    sentence = alignment.get("rebuttal_sentence", "").strip()
                    if review and sentence:
                        queries[(conference, paper_id, review)].append(sentence)
    return queries


def describe(data: dict[str, Any]) -> dict[str, Any]:
    patterns = get_reference_patterns()
    queries = load_queries(data)
    query_units: dict[tuple[str, str, str], set[tuple[str, str]]] = {}

    for query_key, sentences in queries.items():
        units: set[tuple[str, str]] = set()
        for sentence in sentences:
            if len(sentence.split()) <= 6:
                continue
            for mention in find_reference_mentions(sentence, patterns):
                evidence_type = TYPE_NAMES[mention.kind]
                unit_id = normalize_reference(mention.kind, mention.text)
                units.add((evidence_type, unit_id))
        if units:
            query_units[query_key] = units

    unique_units = {
        (conference, paper_id, evidence_type, unit_id)
        for (conference, paper_id, _), units in query_units.items()
        for evidence_type, unit_id in units
    }
    links = sum(len(units) for units in query_units.values())

    combinations: Counter[str] = Counter()
    main_combinations: Counter[str] = Counter()
    for units in query_units.values():
        type_counts = Counter(evidence_type for evidence_type, _ in units)
        combination = " + ".join(
            f"{count} {evidence_type}" if count > 1 else evidence_type
            for evidence_type, count in sorted(type_counts.items())
        )
        combinations[combination] += 1

        main_counts = {key: value for key, value in type_counts.items() if key in MAIN_TYPES}
        if main_counts:
            main_combination = " + ".join(
                f"{count} {evidence_type}" if count > 1 else evidence_type
                for evidence_type, count in sorted(main_counts.items())
            )
            main_combinations[main_combination] += 1

    return {
        "queries": len(query_units),
        "comment_evidence_links": links,
        "unique_evidence_units": len(unique_units),
        "queries_with_multiple_evidence": sum(len(units) >= 2 for units in query_units.values()),
        "evidence_type_combinations": dict(combinations.most_common()),
        "main_evidence_type_combinations": dict(main_combinations.most_common()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "reground.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    data = json.loads(args.dataset.read_text(encoding="utf-8"))
    result = describe(data)
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
