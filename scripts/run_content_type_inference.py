#!/usr/bin/env python3
"""Run content-type inference against an OpenAI-compatible vLLM server.

Input dataset format:
    {
      "<reviewer comment>": ["figure", "line"],
      ...
    }

Output is JSONL, one record per comment, suitable for
scripts/evaluate_content_type_predictions.py.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


FOUR_LABELS = ("figure", "table", "line", "section")
TEXT3_LABELS = ("figure", "table", "text")
DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B-Thinking-2507"
REASONING_TOKEN_BUDGET = 2048
STANDARD_TOKEN_BUDGET = 256


def normalize_gold(labels: list[str], label_mode: str) -> list[str]:
    normalized: set[str] = set()
    for label in labels:
        label = label.strip().lower()
        if label_mode == "text3" and label in {"line", "section"}:
            normalized.add("text")
        else:
            normalized.add(label)
    allowed = TEXT3_LABELS if label_mode == "text3" else FOUR_LABELS
    return [label for label in allowed if label in normalized]


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Parse the first JSON object from a model response."""
    text = text.strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def parse_prediction(raw_text: str, allowed_labels: tuple[str, ...]) -> list[str]:
    allowed = set(allowed_labels)
    parsed = extract_json_object(raw_text)
    values: list[str] = []

    if parsed is not None:
        prediction = parsed.get("content_types", parsed.get("content_type", []))
        if isinstance(prediction, str):
            values = [prediction]
        elif isinstance(prediction, list):
            values = [item for item in prediction if isinstance(item, str)]

    if not values:
        lowered = raw_text.lower()
        values = [label for label in allowed_labels if re.search(rf"\b{label}\b", lowered)]

    deduped = []
    seen = set()
    for value in values:
        label = value.strip().lower()
        if label in allowed and label not in seen:
            deduped.append(label)
            seen.add(label)
    return deduped


def model_profile(model: str, requested: str) -> str:
    if requested != "auto":
        return requested
    lowered = model.lower()
    if "gpt-oss" in lowered:
        return "gpt-oss"
    if "gemma-3" in lowered:
        return "gemma3"
    if "qwen3" in lowered and "thinking" in lowered:
        return "qwen3-thinking"
    if lowered.startswith("gpt-5"):
        return "openai-gpt5"
    return "standard"


def message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
        return "".join(text_parts)
    return ""


def make_content(text: str, profile: str) -> str | list[dict[str, str]]:
    # Gemma 3's official examples use typed content parts with chat templates.
    if profile == "gemma3":
        return [{"type": "text", "text": text}]
    return text


def build_messages(
    comment: str,
    label_mode: str,
    profile: str,
    gpt_oss_reasoning: str,
) -> list[dict[str, Any]]:
    if label_mode == "text3":
        labels = ", ".join(TEXT3_LABELS)
        description = (
            "Classify which content type(s) in the paper the reviewer comment "
            "implicitly relates to. Use text when the target is a line-level or "
            "section-level part of the paper."
        )
    else:
        labels = ", ".join(FOUR_LABELS)
        description = (
            "Classify which content type(s) in the paper the reviewer comment "
            "implicitly relates to."
        )

    system = (
        f"{description}\n"
        f"Allowed labels: {labels}.\n"
        "Return only valid JSON with this schema:\n"
        '{"content_types": ["<one or more allowed labels>"]}\n'
        "Do not include explanations."
    )
    if profile == "gpt-oss":
        system = f"Reasoning: {gpt_oss_reasoning}\n{system}"
    user = f"Reviewer comment:\n{comment}"
    return [
        {"role": "system", "content": make_content(system, profile)},
        {"role": "user", "content": make_content(user, profile)},
    ]


def post_chat_completion(
    *,
    base_url: str,
    provider: str,
    api_key: str | None,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    max_tokens: int,
    openai_reasoning_effort: str,
    timeout: float,
    extra_body: dict[str, Any] | None,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
    }
    headers = {"Content-Type": "application/json"}
    if provider == "openai":
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when --provider openai")
        headers["Authorization"] = f"Bearer {api_key}"
        payload["max_completion_tokens"] = max_tokens
        payload["reasoning_effort"] = openai_reasoning_effort
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "content_type_prediction",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "content_types": {
                            "type": "array",
                            "items": {"type": "string"},
                        }
                    },
                    "required": ["content_types"],
                },
            },
        }
    else:
        payload["temperature"] = temperature
        payload["max_tokens"] = max_tokens
    if extra_body:
        payload.update(extra_body)
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body["choices"][0]["message"]


def load_dataset(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        examples = []
        with path.open("r", encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                comment = record.get("comment")
                labels = record.get("labels", record.get("gold_raw", record.get("gold")))
                if not isinstance(comment, str) or not isinstance(labels, list):
                    raise ValueError(f"Bad JSONL example on line {line_number}: expected comment and labels")
                examples.append(
                    {
                        "id": record.get("id", len(examples)),
                        "comment": comment,
                        "gold_raw": labels,
                    }
                )
        return examples

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Expected dataset JSON to be an object: comment -> labels")

    examples = []
    for idx, (comment, labels) in enumerate(data.items()):
        if not isinstance(comment, str) or not isinstance(labels, list):
            raise ValueError(f"Bad example at index {idx}: expected string -> list")
        examples.append({"id": idx, "comment": comment, "gold_raw": labels})
    return examples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("dataset.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", choices=("vllm", "openai"), default="vllm")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--model-profile",
        choices=("auto", "qwen3-thinking", "gemma3", "gpt-oss", "openai-gpt5", "standard"),
        default="auto",
        help="Request formatting profile; auto infers the three supported model families.",
    )
    parser.add_argument(
        "--openai-reasoning-effort",
        choices=("none", "low", "medium", "high"),
        default="medium",
        help="Reasoning effort for OpenAI GPT-5.1 chat completions.",
    )
    parser.add_argument(
        "--gpt-oss-reasoning",
        choices=("low", "medium", "high"),
        default="medium",
        help="Reasoning level written into the GPT-OSS system prompt.",
    )
    parser.add_argument("--label-mode", choices=("four", "text3"), default="four")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Generation budget. Auto defaults higher for reasoning model profiles.",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument(
        "--extra-body-json",
        default=None,
        help="JSON object merged into each /v1/chat/completions request for vLLM-specific options.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent requests. Increase until vLLM GPU utilization is high.",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    profile = model_profile(args.model, args.model_profile)
    base_url = args.base_url or (
        "https://api.openai.com" if args.provider == "openai" else "http://localhost:8000"
    )
    api_key = os.environ.get(args.api_key_env) if args.provider == "openai" else None
    max_tokens = args.max_tokens
    if max_tokens is None:
        max_tokens = (
            REASONING_TOKEN_BUDGET
            if profile in {"qwen3-thinking", "gpt-oss", "openai-gpt5"}
            else STANDARD_TOKEN_BUDGET
        )
    extra_body = None
    if args.extra_body_json:
        extra_body = json.loads(args.extra_body_json)
        if not isinstance(extra_body, dict):
            raise ValueError("--extra-body-json must decode to a JSON object")

    examples = load_dataset(args.dataset)
    if args.shuffle:
        random.Random(args.seed).shuffle(examples)
    if args.limit is not None:
        examples = examples[: args.limit]

    seen_ids = set()
    if args.resume and args.output.exists():
        with args.output.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    seen_ids.add(json.loads(line)["id"])

    allowed = TEXT3_LABELS if args.label_mode == "text3" else FOUR_LABELS
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def run_one(example: dict[str, Any]) -> dict[str, Any]:
        messages = build_messages(
            example["comment"],
            args.label_mode,
            profile,
            args.gpt_oss_reasoning,
        )
        raw_content = ""
        raw_message: dict[str, Any] | None = None
        error = None
        for attempt in range(args.retries + 1):
            try:
                raw_message = post_chat_completion(
                    base_url=base_url,
                    provider=args.provider,
                    api_key=api_key,
                    model=args.model,
                    messages=messages,
                    temperature=args.temperature,
                    max_tokens=max_tokens,
                    openai_reasoning_effort=args.openai_reasoning_effort,
                    timeout=args.timeout,
                    extra_body=extra_body,
                )
                raw_content = message_text(raw_message.get("content"))
                break
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
                error = str(exc)
                if attempt >= args.retries:
                    break
                time.sleep(1.5 * (attempt + 1))

        prediction = parse_prediction(raw_content, allowed) if raw_content else []
        return {
            "id": example["id"],
            "label_mode": args.label_mode,
            "model": args.model,
            "model_profile": profile,
            "provider": args.provider,
            "max_tokens": max_tokens,
            "openai_reasoning_effort": (
                args.openai_reasoning_effort if args.provider == "openai" else None
            ),
            "gpt_oss_reasoning": args.gpt_oss_reasoning if profile == "gpt-oss" else None,
            "comment": example["comment"],
            "gold": normalize_gold(example["gold_raw"], args.label_mode),
            "gold_raw": example["gold_raw"],
            "prediction": prediction,
            "raw_response": raw_content,
            "raw_message": raw_message,
            "error": error if not raw_content else None,
        }

    pending = [example for example in examples if example["id"] not in seen_ids]
    total = len(pending)
    completed = 0

    with args.output.open("a" if args.resume else "w", encoding="utf-8") as out:
        if args.workers <= 1:
            for example in pending:
                record = run_one(example)
                completed += 1
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                print(
                    f"[{completed}/{total}] id={record['id']} "
                    f"gold={record['gold']} pred={record['prediction']}",
                    file=sys.stderr,
                )
                if args.sleep:
                    time.sleep(args.sleep)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(run_one, example) for example in pending]
                for future in concurrent.futures.as_completed(futures):
                    record = future.result()
                    completed += 1
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out.flush()
                    print(
                        f"[{completed}/{total}] id={record['id']} "
                        f"gold={record['gold']} pred={record['prediction']}",
                        file=sys.stderr,
                    )
                    if args.sleep:
                        time.sleep(args.sleep)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
