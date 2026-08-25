import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm


# =========================================================
# vLLM CONFIG
# =========================================================
VLLM_BASE_URL = os.environ.get("REGROUND_LLM_BASE_URL", "http://localhost:6531/v1")
VLLM_API_KEY = os.environ.get("REGROUND_LLM_API_KEY", "EMPTY")
VLLM_MODEL_NAME = os.environ.get("REGROUND_FILTER_MODEL", "gpt-oss-20b")

client = OpenAI(
    base_url=VLLM_BASE_URL,
    api_key=VLLM_API_KEY,
)

TEMPERATURE = 1.0
SAVE_EVERY = 1


# =========================================================
# PROMPT BUILDER (ORIGINAL PROMPT)
# =========================================================
def build_filter_prompt(review_text, rebuttal_text, rebuttal_comment, target_sentence):
    return f"""
You are analyzing one rebuttal sentence from an academic peer-review exchange.
Your goal is to decide whether this sentence explicitly refers to content from the ORIGINAL ANONYMOUS SUBMISSION (i.e., paper content),
or whether it should be excluded for one of several reasons.

Read ALL provided text, but focus your decision ONLY on the TARGET REBUTTAL SENTENCE.

------------------------------------------------------------
REVIEW TEXT:
{review_text}

------------------------------------------------------------
REBUTTAL TEXT:
{rebuttal_text}

------------------------------------------------------------
RELEVANT REVIEW COMMENT:
{rebuttal_comment}

------------------------------------------------------------
TARGET REBUTTAL SENTENCE (the ONLY sentence to classify):
"{target_sentence}"
------------------------------------------------------------

IMPORTANT DECISION RULES (apply these strictly):
- If the TARGET sentence points to content in this rebuttal or uses deictic cues like "below", "following", "here", "in this response", "in the rebuttal", treat that as rebuttal-introduced material (Q2=true).
- If the TARGET sentence points to paper anchors like Section/Figure/Table/Equation/Appendix/Page/Line WITHOUT such rebuttal-deictic cues, treat that as paper-grounded (Q4=true), unless it is clearly about future changes (Q1=true).
- Treat ANY explicit external citation as external references (Q3=true), including author-year (e.g., Smith et al., 2020), bracketed/numbered citations (e.g., [12]), or explicit mentions of external papers/authors.

Answer the following FOUR questions. Each must be strictly true or false.
Interpret each question EXACTLY as defined.

### Q1: Future Changes / Camera-Ready
Does the sentence describe something the authors plan to add, modify, remove, clarify, expand, or correct in the future, including in the camera-ready version or in any revised version?

### Q2: New Experiments or Newly Introduced Results (During Rebuttal)
Does the sentence report or describe new experiments, analyses, tables, figures, metrics, or results that were NOT part of the original anonymous submission, including material introduced in this rebuttal/response?

### Q3: External Papers / Citations
Does the sentence explicitly reference external papers, authors, or citations (author-year OR bracketed/numbered citations), as opposed to pointing to the paper's included sections/figures/equations/tables?

### Q4: Grounded in the Original Anonymous Submission
Does the sentence explicitly refer to specific, identifiable content in the paper (e.g., Section/Figure/Table/Equation/Appendix/Page/Line or similarly locatable content), as opposed to being generic discussion not anchored to paper content?

------------------------------------------------------------
### REQUIRED OUTPUT FORMAT
Return ONLY valid JSON with ALL fields present:

{{
  "future_changes": true/false,
  "new_experiments": true/false,
  "external_references": true/false,
  "grounded_in_paper": true/false
}}

Do NOT provide explanations.
Do NOT include text outside the JSON.
Return ONLY the JSON object.
""".strip()


# =========================================================
# RESPONSE PARSING
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
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    match = re.search(r"(\{.*?\})", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None

    return None


# =========================================================
# OUTPUT NORMALIZATION (filter_out computed locally)
# =========================================================
BASE_KEYS = (
    "future_changes",
    "new_experiments",
    "external_references",
    "grounded_in_paper",
)

def normalize_labels(out):
    for k in BASE_KEYS:
        if k not in out or not isinstance(out[k], bool):
            return None

    return {
        "future_changes": out["future_changes"],
        "new_experiments": out["new_experiments"],
        "external_references": out["external_references"],
        "grounded_in_paper": out["grounded_in_paper"],
        "filter_out": (
            out["future_changes"]
            or out["new_experiments"]
            or out["external_references"]
            or (not out["grounded_in_paper"])
        ),
    }


# =========================================================
# SINGLE CALL WITH RETRIES
# =========================================================
def _single_vllm_call_with_retries(prompt):
    attempt = 0
    while True:
        attempt += 1
        try:
            resp = client.chat.completions.create(
                model=VLLM_MODEL_NAME,
                messages=[
                    {"role": "system", "content": "Return ONLY the requested JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=TEMPERATURE,
                reasoning_effort="medium",
            )

            msg = resp.choices[0].message
            if msg is None or msg.content is None:
                raise ValueError("Empty response")

            raw = parse_response_text(msg.content)
            if raw is None:
                raise ValueError("Invalid JSON")

            norm = normalize_labels(raw)
            if norm is None:
                raise ValueError("Invalid schema")

            return norm

        except Exception as e:
            tqdm.write(f"[WARN] Retry {attempt}: {e}")
            if attempt >= 10:
                return None


# =========================================================
# BATCHED EXECUTION
# =========================================================
def run_vllm_batched(prompts, batch_size=8):
    results = [None] * len(prompts)

    with ThreadPoolExecutor(max_workers=batch_size) as ex:
        future_to_idx = {
            ex.submit(_single_vllm_call_with_retries, p): i
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
# SAVE HELPERS
# =========================================================
def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# =========================================================
# MAIN PIPELINE
# =========================================================
def main(results_dir="../results", batch_size=8):
    with open(os.path.join(results_dir, "4_unique_refs.json"), encoding="utf-8") as f:
        aligned = json.load(f)

    tasks = []

    for paper_id, entries in tqdm(aligned.items(), desc="Building tasks"):
        for comment_id, data in entries.items():
            for a in data["aligned"]:
                prompt = build_filter_prompt(
                    review_text=data["review"],
                    rebuttal_text=data["rebuttal"],
                    rebuttal_comment=" ".join(a["review_spans"]),
                    target_sentence=a["rebuttal_sentence"],
                )

                tasks.append(
                    {
                        "paper_id": paper_id,
                        "comment_id": comment_id,
                        "sentence": a["rebuttal_sentence"],
                        "review": data["review"],
                        "rebuttal": data["rebuttal"],
                        "review_spans": " ".join(a["review_spans"]),
                        "prompt": prompt,
                    }
                )

    output = {}
    partial_path = os.path.join(results_dir, "5_grounded_references_partial.json")
    completed = set()

    if os.path.exists(partial_path):
        output = json.load(open(partial_path, encoding="utf-8"))
        for pid, comments in output.items():
            for cid, info in comments.items():
                for a in info.get("aligned", []):
                    completed.add((pid, cid, a["rebuttal_sentence"]))

    tasks = [
        t for t in tasks
        if (t["paper_id"], t["comment_id"], t["sentence"]) not in completed
    ]

    failures = []

    for i in tqdm(range(0, len(tasks), batch_size), desc="Running vLLM"):
        batch = tasks[i:i + batch_size]
        prompts = [t["prompt"] for t in batch]
        results = run_vllm_batched(prompts, batch_size=batch_size)

        for task, out in zip(batch, results):
            pid = task["paper_id"]
            cid = task["comment_id"]

            if out is None:
                failures.append([pid, cid, task["sentence"]])
                continue

            output.setdefault(pid, {})
            output[pid].setdefault(
                cid,
                {
                    "review": task["review"],
                    "rebuttal": task["rebuttal"],
                    "aligned": [],
                },
            )

            output[pid][cid]["aligned"].append(
                {
                    "rebuttal_sentence": task["sentence"],
                    "review_spans": task["review_spans"],
                    "labels": out,
                }
            )

        if (i // batch_size) % SAVE_EVERY == 0:
            save_json(partial_path, output)

    save_json(os.path.join(results_dir, "5_grounded_references.json"), output)

    if failures:
        save_json(os.path.join(results_dir, "5_grounded_references_failures.json"), failures)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="../results")
    ap.add_argument("--batch-size", type=int, default=50)
    a = ap.parse_args()
    main(a.results_dir, a.batch_size)
