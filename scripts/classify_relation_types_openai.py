#!/usr/bin/env python3
"""Classify the fixed 200-pair relation sample with the OpenAI Responses API.

Setup:
  1. Paste your API key into OPENAI_API_KEY below, or export OPENAI_API_KEY.
  2. Install/update the SDK: uv pip install --python .reb/bin/python -U openai
  3. Run: .reb/bin/python code_and_data/scripts/classify_relation_types_openai.py

The script is resumable and writes a checkpoint after every completed item.
It never modifies the manual-annotation input CSV.
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
OPENAI_API_KEY = ""

RELATIONS = ("corrective", "explanatory", "other_or_unclear")

SYSTEM_PROMPT = """Classify how an author's reference-bearing rebuttal sentence relates to a reviewer comment.

Use exactly one label:
- corrective: The author uses the cited paper content to contradict a reviewer claim, reject a criticism's premise, or show that something the reviewer claimed was missing or overlooked was already present. The defining feature is that the response treats the reviewer as substantively mistaken about the submitted paper.
- explanatory: The author uses the cited content to answer a reasonable question, clarify or disambiguate the work, provide justification or requested evidence, explain scope or a trade-off, or otherwise address a legitimate concern without primarily establishing that the reviewer was mistaken.
- other_or_unclear: The relation is procedural, mixed without a clear dominant function, not captured above, or cannot be determined confidently from the supplied pair.

Important boundary: merely pointing to content already in the paper is not automatically corrective. If that content answers a reasonable clarification request without contradicting the reviewer, label it explanatory. Classify the communicative relation, not sentiment or politeness. Base the decision only on the supplied text. Return the required JSON object."""

SCHEMA = {
    "type": "object",
    "properties": {
        "relation": {"type": "string", "enum": list(RELATIONS)},
        "rationale": {"type": "string", "description": "One brief sentence tied to the text."},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["relation", "rationale", "confidence"],
    "additionalProperties": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "relation_type_samples" / "relation_types_200.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "relation_type_samples" / "relation_types_200_llm.csv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "relation_type_samples" / "relation_types_200_llm_summary.json",
    )
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=8)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def merge_existing(rows: list[dict[str, str]], output: Path) -> None:
    if not output.exists():
        return
    existing = {row["sample_id"]: row for row in read_rows(output)}
    for row in rows:
        previous = existing.get(row["sample_id"], {})
        if previous.get("llm_relation") in RELATIONS:
            for field in ("llm_relation", "llm_confidence", "llm_rationale"):
                row[field] = previous.get(field, "")


def classify(client: OpenAI, model: str, row: dict[str, str], max_retries: int) -> dict[str, str]:
    user_prompt = (
        f"Reviewer comment:\n{row['review_comment']}\n\n"
        f"Author rebuttal sentence:\n{row['rebuttal_sentence']}\n\n"
        f"Explicit paper reference(s):\n{row['paper_references']}"
    )
    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "rebuttal_relation",
                        "strict": True,
                        "schema": SCHEMA,
                    }
                },
            )
            result = json.loads(response.output_text)
            return {
                "llm_relation": result["relation"],
                "llm_confidence": result["confidence"],
                "llm_rationale": result["rationale"],
            }
        except Exception:
            if attempt + 1 == max_retries:
                raise
            time.sleep(min(60.0, (2**attempt) + random.random()))
    raise RuntimeError("Unreachable")


def summarize(rows: list[dict[str, str]], model: str) -> dict[str, Any]:
    counts = Counter(row["llm_relation"] for row in rows)
    total = sum(counts.values())
    return {
        "model": model,
        "items": total,
        "relations": {
            relation: {
                "count": counts[relation],
                "percentage": 100 * counts[relation] / total if total else 0.0,
            }
            for relation in RELATIONS
        },
    }


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("OPENAI_API_KEY") or OPENAI_API_KEY
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY or paste it into OPENAI_API_KEY near the top of this script.")

    rows = read_rows(args.input)
    if not rows:
        raise SystemExit(f"No rows found in {args.input}")
    for row in rows:
        row.update({"llm_relation": "", "llm_confidence": "", "llm_rationale": ""})
    merge_existing(rows, args.output)
    pending = [index for index, row in enumerate(rows) if row["llm_relation"] not in RELATIONS]
    print(f"Rows: {len(rows)}; complete: {len(rows) - len(pending)}; pending: {len(pending)}")

    client = OpenAI(api_key=api_key)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(classify, client, args.model, rows[index], args.max_retries): index
            for index in pending
        }
        completed = len(rows) - len(pending)
        for future in as_completed(futures):
            index = futures[future]
            rows[index].update(future.result())
            completed += 1
            write_rows(args.output, rows)
            print(f"Classified {completed}/{len(rows)}", flush=True)

    args.summary.write_text(
        json.dumps(summarize(rows, args.model), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
