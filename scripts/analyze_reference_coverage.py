#!/usr/bin/env python3
"""Measure coverage of reviewer spans by retained reference-bearing rebuttals."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import safe_sentence_split  # noqa: E402


THREAD_FILES = {
    "ACL25": ("acl25.json",),
    "COLING25": ("coling25.json",),
    "EMNLP24": ("emnlp24_1.json", "emnlp24_2.json"),
    "EMNLP25": ("emnlp25.json",),
    "NAACL25": ("naacl25.json",),
}


def normalize(text: str) -> str:
    return " ".join((text or "").split()).casefold()


def review_fields(comment: Any) -> list[tuple[str, str]]:
    if isinstance(comment, str):
        return [("unstructured", comment)]
    if isinstance(comment, dict):
        return [(key, value) for key, value in comment.items() if isinstance(value, str)]
    return []


FEEDBACK_FIELDS = {"summary_of_weaknesses", "comments_suggestions_and_typos"}


def load_threads(directory: Path) -> dict[str, dict[str, dict[str, Any]]]:
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for conference, filenames in THREAD_FILES.items():
        papers: dict[str, dict[str, Any]] = {}
        for filename in filenames:
            shard = json.loads((directory / filename).read_text(encoding="utf-8"))
            overlap = set(papers).intersection(shard)
            if overlap:
                raise ValueError(f"Duplicate paper IDs across {conference} shards: {len(overlap)}")
            papers.update(shard)
        output[conference] = papers
    return output


def retained_queries(og: dict[str, Any]) -> dict[tuple[str, str], list[str]]:
    retained: dict[tuple[str, str], list[str]] = defaultdict(list)
    for conference, papers in og.items():
        for paper_id, rebuttals in papers.items():
            for rebuttal in rebuttals.values():
                for alignment in rebuttal.get("aligned", []):
                    spans = alignment.get("review_spans", "")
                    if isinstance(spans, list):
                        retained[(conference, paper_id)].extend(str(span) for span in spans)
                    elif spans:
                        retained[(conference, paper_id)].append(str(spans))
    return retained


def has_author_response(note_id: str, notes: dict[str, dict[str, Any]]) -> bool:
    return any(
        note.get("from") == "Authors" and note.get("replyto") == note_id
        for note in notes.values()
    )


def analyze(og: dict[str, Any], threads: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    retained = retained_queries(og)
    all_spans: list[dict[str, Any]] = []

    for conference, papers in threads.items():
        for paper_id, notes in papers.items():
            retained_texts = [normalize(text) for text in retained.get((conference, paper_id), [])]
            for note_id, note in notes.items():
                if note.get("depth") != 0 or note.get("from") == "Authors":
                    continue
                responded = has_author_response(note_id, notes)
                for field, text in review_fields(note.get("comment", "")):
                    for index, span in enumerate(safe_sentence_split(text)):
                        span_norm = normalize(span)
                        included = any(
                            span_norm == retained_text
                            or span_norm in retained_text
                            or retained_text in span_norm
                            for retained_text in retained_texts
                            if retained_text
                        )
                        all_spans.append(
                            {
                                "conference": conference,
                                "paper_id": paper_id,
                                "review_note_id": note_id,
                                "review_field": field,
                                "span_index": index,
                                "comment": span,
                                "author_responded": responded,
                                "included": included,
                                "feedback_field": field in FEEDBACK_FIELDS or field == "unstructured",
                            }
                        )

    included = [span for span in all_spans if span["included"]]
    excluded = [span for span in all_spans if not span["included"]]
    responded = [span for span in all_spans if span["author_responded"]]
    included_responded = [span for span in responded if span["included"]]
    feedback = [span for span in all_spans if span["feedback_field"]]
    feedback_included = [span for span in feedback if span["included"]]
    feedback_responded = [span for span in feedback if span["author_responded"]]
    feedback_included_responded = [span for span in feedback_responded if span["included"]]
    retained_query_count = len(
        {
            (conference, paper_id, normalize(text))
            for (conference, paper_id), texts in retained.items()
            for text in texts
            if normalize(text)
        }
    )

    result = {
        "all_review_spans": len(all_spans),
        "review_spans_with_direct_author_response": len(responded),
        "retained_reference_bearing_queries": retained_query_count,
        "matched_included_review_spans": len(included),
        "matched_included_responded_spans": len(included_responded),
        "coverage_of_all_review_spans_pct": 100 * len(included) / len(all_spans),
        "coverage_of_responded_review_spans_pct": 100 * len(included_responded) / len(responded),
        "feedback_review_spans": len(feedback),
        "matched_included_feedback_spans": len(feedback_included),
        "coverage_of_feedback_review_spans_pct": 100 * len(feedback_included) / len(feedback),
        "feedback_spans_with_direct_author_response": len(feedback_responded),
        "matched_included_responded_feedback_spans": len(feedback_included_responded),
        "coverage_of_responded_feedback_spans_pct": 100 * len(feedback_included_responded) / len(feedback_responded),
        "note": "Broad coverage includes all substantive review fields. Feedback-only coverage uses summary_of_weaknesses and comments_suggestions_and_typos (plus unstructured reviews).",
    }
    return result, feedback_included, [span for span in feedback if not span["included"]]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_annotation_csv(path: Path, included: list[dict[str, Any]], excluded: list[dict[str, Any]]) -> None:
    fields = ["set", "conference", "paper_id", "review_note_id", "review_field", "span_index", "comment", "category"]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for set_name, records in (("included", included), ("excluded", excluded)):
            for record in records:
                writer.writerow({"set": set_name, **{field: record.get(field, "") for field in fields[1:-1]}, "category": ""})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--og", type=Path, default=ROOT / "reground.json")
    parser.add_argument("--threads", type=Path, default=ROOT / "data" / "threads")
    parser.add_argument("--output", type=Path, default=ROOT / "reference_coverage.json")
    parser.add_argument("--sample-dir", type=Path, default=ROOT / "reference_coverage_samples")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=200,
        help="Number sampled per group (200 included + 200 excluded by default).",
    )
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    og = json.loads(args.og.read_text(encoding="utf-8"))
    threads = load_threads(args.threads)
    result, included, excluded = analyze(og, threads)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    rng = random.Random(args.seed)
    included_sample = rng.sample(included, min(args.sample_size, len(included)))
    excluded_sample = rng.sample(excluded, min(args.sample_size, len(excluded)))
    write_jsonl(args.sample_dir / "included.jsonl", included_sample)
    write_jsonl(args.sample_dir / "excluded.jsonl", excluded_sample)
    write_annotation_csv(args.sample_dir / "taxonomy_annotation.csv", included_sample, excluded_sample)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
