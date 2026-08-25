#!/usr/bin/env python3
"""Classify included/excluded reviewer spans with the OpenAI Responses API.

Setup:
  1. Put your API key in OPENAI_API_KEY below, or export OPENAI_API_KEY.
  2. Install/update the SDK: uv pip install --python .reb/bin/python -U openai
  3. Run: .reb/bin/python code_and_data/scripts/classify_taxonomy_openai.py

The script is resumable. It checkpoints after every completed batch and skips
rows that already have a valid category in the output CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
MODEL = "gpt-5.6-luna"

# You may paste your key here. Prefer the OPENAI_API_KEY environment variable
# when possible so the key is not stored in source control.
OPENAI_API_KEY = ""

CATEGORIES = (
    "Conceptual & Novelty",
    "Technical Soundness",
    "Empirical Rigor",
    "Data & Reproducibility",
    "Presentation & Writing",
    "Deployment & Impact",
    "Other",
)

SYSTEM_PROMPT = """You classify peer-review feedback by its single dominant dimension.

Use exactly one of these mutually exclusive categories:
- Conceptual & Novelty: originality, significance, motivation, framing, or relation to prior work.
- Technical Soundness: correctness of methods, theory, assumptions, algorithms, or technical reasoning.
- Empirical Rigor: experiments, baselines, evaluation design, metrics, analysis, statistical support, or interpretation of results.
- Data & Reproducibility: dataset construction or quality, annotation, data coverage, implementation details needed for reproduction, code/data availability, or reproducibility.
- Presentation & Writing: clarity, organization, terminology, figures/tables as presentation, formatting, or writing quality.
- Deployment & Impact: efficiency, computational cost, scalability in use, ethics, risks, societal impact, or real-world applicability.
- Other: administrative, meta-review, vague, or not meaningfully covered above.

Choose the dominant issue even when a span mentions several dimensions. Classify the concern being raised, not isolated keywords. Return only the required JSON object."""

SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": list(CATEGORIES)},
    },
    "required": ["category"],
    "additionalProperties": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "reference_coverage_samples" / "taxonomy_annotation.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reference_coverage_samples" / "taxonomy_classified.csv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "reference_coverage_samples" / "taxonomy_comparison.json",
    )
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=8)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def row_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row["set"],
        row["conference"],
        row["paper_id"],
        row["review_note_id"],
        row.get("review_field", ""),
        row["span_index"],
    )


def merge_existing(rows: list[dict[str, str]], output: Path) -> None:
    if not output.exists():
        return
    existing = {row_key(row): row.get("category", "") for row in read_rows(output)}
    for row in rows:
        category = existing.get(row_key(row), "")
        if category in CATEGORIES:
            row["category"] = category


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(rows[0])
    with temporary.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def classify(client: OpenAI, model: str, comment: str, max_retries: int) -> str:
    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Reviewer feedback:\n{comment}"},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "review_taxonomy",
                        "strict": True,
                        "schema": SCHEMA,
                    }
                },
            )
            category = json.loads(response.output_text)["category"]
            if category not in CATEGORIES:
                raise ValueError(f"Unexpected category: {category!r}")
            return category
        except Exception:
            if attempt + 1 == max_retries:
                raise
            delay = min(60.0, (2**attempt) + random.random())
            time.sleep(delay)
    raise RuntimeError("Unreachable")


def summarize(rows: list[dict[str, str]], model: str) -> dict[str, Any]:
    sets = sorted({row["set"] for row in rows})
    counts = {set_name: Counter(row["category"] for row in rows if row["set"] == set_name) for set_name in sets}
    result: dict[str, Any] = {"model": model, "categories": {}}
    for category in CATEGORIES:
        result["categories"][category] = {}
        for set_name in sets:
            total = sum(counts[set_name].values())
            count = counts[set_name][category]
            result["categories"][category][set_name] = {
                "count": count,
                "percentage": 100 * count / total if total else 0.0,
            }
        if "included" in sets and "excluded" in sets:
            result["categories"][category]["difference_percentage_points"] = (
                result["categories"][category]["included"]["percentage"]
                - result["categories"][category]["excluded"]["percentage"]
            )
    return result


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("OPENAI_API_KEY") or OPENAI_API_KEY
    if not api_key:
        raise SystemExit(
            "Set the OPENAI_API_KEY environment variable or paste the key into "
            "OPENAI_API_KEY near the top of this script."
        )

    rows = read_rows(args.input)
    if not rows:
        raise SystemExit(f"No rows found in {args.input}")
    merge_existing(rows, args.output)
    pending = [index for index, row in enumerate(rows) if row.get("category") not in CATEGORIES]
    print(f"Rows: {len(rows)}; already classified: {len(rows) - len(pending)}; pending: {len(pending)}")

    client = OpenAI(api_key=api_key)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(classify, client, args.model, rows[index]["comment"], args.max_retries): index
            for index in pending
        }
        completed = len(rows) - len(pending)
        for future in as_completed(futures):
            index = futures[future]
            rows[index]["category"] = future.result()
            completed += 1
            write_rows(args.output, rows)
            print(f"Classified {completed}/{len(rows)}", flush=True)

    summary = summarize(rows, args.model)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
