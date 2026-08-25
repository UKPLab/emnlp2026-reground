import json
import os
import re
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm
from utils import safe_sentence_split


random.seed(42)


# =========================================================
# vLLM client configuration (override via env for the release)
# =========================================================
VLLM_BASE_URL = os.environ.get("REGROUND_LLM_BASE_URL", "http://localhost:6531/v1")
VLLM_API_KEY = os.environ.get("REGROUND_LLM_API_KEY", "EMPTY")
VLLM_MODEL_NAME = os.environ.get("REGROUND_LLM_MODEL", "gpt-oss-120b")

client = OpenAI(
    base_url=VLLM_BASE_URL,
    api_key=VLLM_API_KEY,
)

TEMPERATURE = 1.0  # according to the model card this has to be 1.0 for gpt-oss models
SAVE_EVERY = 1


# =========================================================
# Reviewer span construction
# =========================================================
def build_reviewer_spans(review_text):
    sentences = safe_sentence_split(review_text)
    return [
        {"span_id": f"rev_sent_{i}", "span_text": s, "span_index": i}
        for i, s in enumerate(sentences)
    ]


# =========================================================
# Rebuttal window extraction
# =========================================================
def build_rebuttal_window(
    rebuttal_text,
    target_sentence,
    pre,
    post,
):
    rebuttal_sents = safe_sentence_split(rebuttal_text)
    if not rebuttal_sents:
        return rebuttal_text

    tgt = (target_sentence or "").strip()
    if not tgt:
        return rebuttal_text

    idx = None
    for i, s in enumerate(rebuttal_sents):
        if (s or "").strip() == tgt:
            idx = i
            break

    if idx is None:

        def norm(x):
            return " ".join((x or "").split())

        tgt_n = norm(tgt)
        for i, s in enumerate(rebuttal_sents):
            if norm(s) == tgt_n:
                idx = i
                break

    if idx is None:
        return rebuttal_text

    start = max(0, idx - pre)
    end = min(len(rebuttal_sents), idx + post + 1)
    return "\n".join(rebuttal_sents[start:end])


# =========================================================
# Prompt construction
# =========================================================
def build_prompt(rebuttal, target_sentence, spans):
    spans_str = "\n".join(f"{s['span_id']}: {s['span_text']}" for s in spans)

    return f"""
You are given a rebuttal context from an author responding to reviewer comments, and a specific sentence from that rebuttal.
Your task is to align a rebuttal sentence to ALL corresponding reviewer spans it's addressing.

### Reviewer Spans
Choose ONLY from these IDs.
{spans_str}

### Rebuttal Context
{rebuttal}

### Target Sentence
"{target_sentence}"

### Output
Return ONLY JSON:
{{"review_spans": ["rev_sent_#1", "rev_sent_#2"]}}
""".strip()


# =========================================================
# Response parsing
# =========================================================
def parse_response_text(text):
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"(\{.*?\}|\[.*?\])", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            return None

    return None


# =========================================================
# Single vLLM call with retries
# =========================================================
def _single_vllm_call_with_retries(prompt, effort="medium"):
    attempt = 0
    while True:
        attempt += 1
        try:
            resp = client.chat.completions.create(
                model=VLLM_MODEL_NAME,
                messages=[
                    {"role": "system", "content": "Only output JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=TEMPERATURE,
                reasoning_effort=effort,
            )

            msg = resp.choices[0].message
            if msg is None or msg.content is None:
                raise ValueError("Empty response")

            out = parse_response_text(msg.content)
            if out is None:
                raise ValueError("Invalid JSON")

            return out

        except Exception as e:
            tqdm.write(f"[WARN] Retry {attempt}: {e}")
            if attempt >= 10:
                return None


# =========================================================
# Batched parallel execution
# =========================================================
def run_vllm_batched(prompts, batch_size=8, effort="medium"):
    results = [None] * len(prompts)

    with ThreadPoolExecutor(max_workers=batch_size) as ex:
        future_to_idx = {
            ex.submit(_single_vllm_call_with_retries, p, effort): i
            for i, p in enumerate(prompts)
        }

        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            try:
                results[idx] = fut.result()
            except Exception:
                results[idx] = None

    return results


# =========================================================
# Persistence helpers
# =========================================================
def save_partial(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# =========================================================
# Main pipeline
# =========================================================
def main(results_dir="../results", batch_size=8):
    dataset = json.load(open(os.path.join(results_dir, "2_regex_no_quotes.json"), encoding="utf-8"))

    tasks = []
    spans_map = {}
    comment_meta = {}

    for paper_id, entries in tqdm(dataset.items(), desc="Building tasks"):
        for comment_id, entry in entries.items():
            key = (paper_id, comment_id)

            rebuttal = entry["rebuttal"]
            review = entry["review"]

            comment_meta[key] = {"rebuttal": rebuttal, "review": review}

            spans = build_reviewer_spans(review)
            spans_map[key] = {s["span_id"]: s["span_text"] for s in spans}

            for sent in entry["referencing_sentences"]:
                prompt = build_prompt(rebuttal, sent, spans)
                tasks.append(
                    {
                        "paper_id": paper_id,
                        "comment_id": comment_id,
                        "sentence": sent,
                        "prompt": prompt,
                    }
                )

    aligned_output = {}
    partial_path = os.path.join(results_dir, "3_review_alignment_partial.json")
    completed = set()

    if os.path.exists(partial_path):
        aligned_output = json.load(open(partial_path, encoding="utf-8"))
        for pid, comments in aligned_output.items():
            for cid, info in comments.items():
                for a in info.get("aligned", []):
                    completed.add((pid, cid, a.get("rebuttal_sentence")))

    tasks = [
        t
        for t in tasks
        if (t["paper_id"], t["comment_id"], t["sentence"]) not in completed
    ]

    failures = []

    for i in tqdm(range(0, len(tasks), batch_size), desc="Running batches"):
        batch = tasks[i : i + batch_size]
        prompts = [t["prompt"] for t in batch]
        outputs = run_vllm_batched(prompts, batch_size=batch_size)

        for task, out in zip(batch, outputs):
            pid = task["paper_id"]
            cid = task["comment_id"]
            sent = task["sentence"]
            key = (pid, cid)

            if not out or "review_spans" not in out:
                failures.append((pid, cid, sent))
                continue

            span_texts = [
                spans_map[key][sid]
                for sid in out["review_spans"]
                if sid in spans_map[key]
            ]
            if not span_texts:
                continue

            aligned_output.setdefault(pid, {})
            aligned_output[pid].setdefault(
                cid,
                {
                    "rebuttal": comment_meta[key]["rebuttal"],
                    "review": comment_meta[key]["review"],
                    "aligned": [],
                },
            )

            aligned_output[pid][cid]["aligned"].append(
                {
                    "rebuttal_sentence": sent,
                    "review_spans": span_texts,
                }
            )

        if (i // batch_size) % SAVE_EVERY == 0 and i > 0:
            save_partial(partial_path, aligned_output)

    save_partial(os.path.join(results_dir, "3_review_alignment.json"), aligned_output)

    if failures:
        with open(
            os.path.join(results_dir, "3_review_alignment_failures.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(failures, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="../results")
    ap.add_argument("--batch-size", type=int, default=50)
    a = ap.parse_args()
    main(a.results_dir, a.batch_size)
