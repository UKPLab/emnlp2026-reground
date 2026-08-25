import json
import math
import os
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Tuple

import get_from_paper
import openai
import threading
from openai import OpenAI
from tqdm import tqdm

random.seed(42)

# ---------------------------------------------------------------------
# Token usage accounting (thread-safe)
# ---------------------------------------------------------------------
_TOKEN_LOCK = threading.Lock()
TOKEN_USAGE = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "num_requests": 0,
}


# ---------------------------------------------------------------------
# OpenAI (commercial) client config
# ---------------------------------------------------------------------
# IMPORTANT: set OPENAI_API_KEY in your environment.
#   export OPENAI_API_KEY="..."
#
# Optional (only if you use a proxy / gateway):
#   export OPENAI_BASE_URL="https://api.openai.com/v1"
#
OPENAI_API_KEY = ""
OPENAI_BASE_URL = ""
OPENAI_MODEL_NAME = os.environ.get("OPENAI_MODEL_NAME", "gpt-5.1")

MAX_RETRIES = 5

# For OpenAI Chat Completions, top_logprobs must be 0..20
OPENAI_TOP_LOGPROBS_CAP = 20


# ---------------------------------------------------------------------
# Candidate construction from SDR (unchanged)
# ---------------------------------------------------------------------
def get_by_type(sdr: Dict, ref_type: str):
    doc_ids = []
    doc_texts = []

    if ref_type == "Line":
        for ln, text in sdr["lines"].items():
            doc_ids.append(f"Line {ln}")
            doc_texts.append(text)
        return doc_ids, doc_texts

    elif ref_type == "Paragraph":
        for para in sdr["paragraphs"]:
            doc_ids.append(f"Paragraph {para['paragraph_id']}")
            para_text = get_from_paper.get_text_by_line(
                sdr, (para["start_line"], para["end_line"])
            )
            doc_texts.append(para_text)
        return doc_ids, doc_texts

    elif ref_type == "Section":
        for section in sdr["sections"]:
            doc_ids.append(f"Section {section['number']} / {section['name']}")
            sec_text = get_from_paper.get_section_text_by_number(sdr, section["number"])
            doc_texts.append(sec_text)
        return doc_ids, doc_texts

    elif ref_type == "Caption":
        doc_ids = list(sdr["captions"].keys())
        doc_texts = list(sdr["captions"].values())
        return doc_ids, doc_texts

    else:
        raise ValueError(f"Unknown reference type: {ref_type}")


def build_candidates_from_sdr(
    sdr: Dict[str, Any],
    include_types: Iterable[str] = ("Line", "Paragraph", "Section", "Caption"),
) -> List[Dict[str, str]]:
    candidates: List[Dict[str, str]] = []

    try:
        for ref_type in include_types:
            doc_ids, doc_texts = get_by_type(sdr, ref_type)

            for doc_id, doc_text in zip(doc_ids, doc_texts):
                if not doc_text:
                    continue
                passage = doc_text.strip()
                if not passage:
                    continue
                candidates.append({"id": doc_id, "text": passage})
    except Exception:
        return []

    return candidates


# ---------------------------------------------------------------------
# Prompt builder (unchanged)
# ---------------------------------------------------------------------
def build_pointwise_prompt(
    query: str,
    passage: str,
    label_yes: str = "A",
    label_no: str = "B",
    cot: bool = False,
) -> str:
    if cot:
        instructions = (
            f"- First, write a short reasoning (1 paragraph).\n"
            f"- Then output a final answer on a new line in EXACTLY ONE character: {label_yes} or {label_no}.\n"
            f"- Do not output anything after that one character.\n"
        )
    else:
        instructions = (
            f"- Output a final answer on a new line in EXACTLY ONE character: {label_yes} or {label_no}.\n"
            f"- Do not output anything after that one character.\n"
        )

    prompt = f"""You are evaluating whether a passage from an NLP paper helps addressing a comment from its corresponding peer-review.

Review comment:
{query}

Paper passage:
{passage}

Instructions:
{instructions}

Meaning:
{label_yes} = Yes, the passage helps addressing the comment.
{label_no} = No, the passage does not help addressing the comment.

Output:""".strip()

    return prompt


# ---------------------------------------------------------------------
# Logprob extraction utilities (unchanged)
# ---------------------------------------------------------------------
def _as_dict(obj) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return obj


def _normalize_token(t: str) -> str:
    return t.strip()


def _logsumexp(a: float, b: float) -> float:
    m = max(a, b)
    if m == -math.inf:
        return -math.inf
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def _extract_token_logprobs(choice) -> List[Dict[str, Any]]:
    lp = getattr(choice, "logprobs", None)
    lp = _as_dict(lp)

    if not lp:
        return []

    content = lp.get("content") if isinstance(lp, dict) else None
    if content is None and hasattr(getattr(choice, "logprobs", None), "content"):
        content = _as_dict(choice.logprobs.content)

    if not content or not isinstance(content, list):
        return []

    out: List[Dict[str, Any]] = []
    for item in content:
        item = _as_dict(item)
        if not isinstance(item, dict):
            continue

        token = item.get("token")
        logprob = item.get("logprob")

        top = item.get("top_logprobs", [])
        top_map: Dict[str, float] = {}

        if isinstance(top, dict):
            for k, v in top.items():
                if isinstance(k, str) and isinstance(v, (int, float)):
                    top_map[k] = float(v)
        elif isinstance(top, list):
            for titem in top:
                titem = _as_dict(titem)
                if not isinstance(titem, dict):
                    continue
                tt = titem.get("token")
                tlp = titem.get("logprob")
                if isinstance(tt, str) and isinstance(tlp, (int, float)):
                    top_map[tt] = float(tlp)

        if isinstance(token, str) and isinstance(logprob, (int, float)):
            out.append(
                {"token": token, "logprob": float(logprob), "top_logprobs": top_map}
            )

    return out


def _pick_decision_index(
    completion_text: str,
    token_items: List[Dict[str, Any]],
    label_yes: str,
    label_no: str,
) -> Optional[int]:
    text = (completion_text or "").rstrip()
    target = text[-1] if text else ""

    labels = {label_yes, label_no}

    def is_label(tok: str) -> bool:
        return _normalize_token(tok) in labels

    if target in labels:
        for i in range(len(token_items) - 1, -1, -1):
            if _normalize_token(token_items[i]["token"]) == target:
                return i

    for i in range(len(token_items) - 1, -1, -1):
        if is_label(token_items[i]["token"]):
            return i

    return None


def _prob_yes_from_choice(
    choice,
    label_yes: str = "A",
    label_no: str = "B",
) -> Tuple[Optional[float], Dict[str, Any]]:
    msg = getattr(choice, "message", None)
    completion_text = getattr(msg, "content", "") if msg else ""

    token_items = _extract_token_logprobs(choice)
    if not token_items:
        return None, {"error": "No token logprobs returned by server/model."}

    decision_idx = _pick_decision_index(
        completion_text, token_items, label_yes, label_no
    )
    if decision_idx is None:
        return None, {
            "error": "Could not locate decision token (A/B) in generated tokens.",
            "completion_text_tail": (completion_text or "")[-200:],
            "last_tokens": [ti["token"] for ti in token_items[-20:]],
        }

    ti = token_items[decision_idx]
    top_map: Dict[str, float] = ti.get("top_logprobs", {}) or {}

    norm_top: Dict[str, float] = {}
    for k, v in top_map.items():
        nk = _normalize_token(k)
        norm_top[nk] = max(norm_top.get(nk, -math.inf), float(v))

    lp_yes = norm_top.get(label_yes, -math.inf)
    lp_no = norm_top.get(label_no, -math.inf)

    emitted_norm = _normalize_token(ti["token"])
    emitted_lp = float(ti.get("logprob", -math.inf))

    if lp_yes == -math.inf and emitted_norm == label_yes:
        lp_yes = emitted_lp
    if lp_no == -math.inf and emitted_norm == label_no:
        lp_no = emitted_lp

    if lp_yes == -math.inf and lp_no == -math.inf:
        return None, {
            "error": "Neither label found in top_logprobs; increase top_logprobs (<=20 for OpenAI).",
            "decision_token": ti["token"],
            "decision_logprob": emitted_lp,
            "top_logprobs_keys_tail": list(norm_top.keys())[:50],
        }

    denom = _logsumexp(lp_yes, lp_no)
    p_yes = math.exp(lp_yes - denom) if denom != -math.inf else None

    debug = {
        "decision_index": decision_idx,
        "decision_token_raw": ti["token"],
        "decision_token_norm": emitted_norm,
        "logprob_yes": lp_yes,
        "logprob_no": lp_no,
        "p_yes": p_yes,
        "completion_text_tail": (completion_text or "")[-200:],
    }
    return p_yes, debug


def reset_token_usage():
    with _TOKEN_LOCK:
        for k in TOKEN_USAGE:
            TOKEN_USAGE[k] = 0

def snapshot_token_usage():
    with _TOKEN_LOCK:
        return dict(TOKEN_USAGE)


# ---------------------------------------------------------------------
# OpenAI call: pointwise + logprobs (UPDATED for GPT-5.1)
# ---------------------------------------------------------------------
def _sleep_backoff_seconds(attempt: int, base: float = 1.0, cap: float = 60.0) -> float:
    # exponential backoff + jitter; deterministic jitter because random.seed(42)
    return min(cap, base * (2 ** (attempt - 1))) + random.random()


def _call_llm_pointwise_logprobs(
    client: OpenAI,
    model_name: str,
    prompt: str,
    top_logprobs: int = 20,
) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    Single-call: return P(Yes) from token logprobs at the decision token.
    Works with OpenAI Chat Completions + GPT-5.1.

    IMPORTANT:
      - GPT-5.1 + {temperature, top_p, logprobs} requires reasoning_effort='none'.
    """
    # OpenAI cap
    top_logprobs = max(0, min(int(top_logprobs), OPENAI_TOP_LOGPROBS_CAP))

    attempt = 0
    while True:
        attempt += 1
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a careful evaluator. Follow instructions exactly.",
                    },
                    {"role": "user", "content": prompt},
                ],
                # Keep your benchmarking comparable to vLLM runs:
                temperature=0.0,
                # Required when requesting logprobs/temperature on GPT-5.1:
                reasoning_effort="none",
                logprobs=True,
                top_logprobs=top_logprobs,
                # Only need a tiny output budget (newline + A/B); keeps cost bounded.
                max_completion_tokens=16,
            )


            choice = resp.choices[0]
            p_yes, debug = _prob_yes_from_choice(choice, label_yes="A", label_no="B")

            # ---------------- TOKEN ACCOUNTING ----------------
            if hasattr(resp, "usage") and resp.usage is not None:
                with _TOKEN_LOCK:
                    TOKEN_USAGE["prompt_tokens"] += resp.usage.prompt_tokens or 0
                    TOKEN_USAGE["completion_tokens"] += resp.usage.completion_tokens or 0
                    TOKEN_USAGE["total_tokens"] += resp.usage.total_tokens or 0
                    TOKEN_USAGE["num_requests"] += 1
            # --------------------------------------------------


            # extra metadata can help your paper writeup/repro:
            debug["model"] = getattr(resp, "model", None)
            debug["system_fingerprint"] = getattr(resp, "system_fingerprint", None)

            if p_yes is None:
                if attempt >= MAX_RETRIES:
                    tqdm.write(
                        f"[WARN] Scoring failed on attempt {attempt} because {debug.get('error')}. "
                        f"Output is: {getattr(choice.message, 'content', '')[:200]}"
                    )
                    return None, debug
            else:
                msg = getattr(choice, "message", None)
                debug["raw_text"] = getattr(msg, "content", "") if msg else ""
                return p_yes, debug

        except openai.RateLimitError as e:
            if attempt >= MAX_RETRIES:
                return None, {
                    "error": f"rate_limited_after_{attempt}_attempts",
                    "exception": str(e),
                }
            time.sleep(_sleep_backoff_seconds(attempt))
            continue

        except (
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.InternalServerError,
        ) as e:
            if attempt >= MAX_RETRIES:
                return None, {
                    "error": f"transient_error_after_{attempt}_attempts",
                    "exception": str(e),
                }
            time.sleep(_sleep_backoff_seconds(attempt))
            continue

        except openai.BadRequestError as e:
            msg = str(e).lower()
            # context length exceeded is a hard fail for a given (query, passage) pair
            if "context_length_exceeded" in msg or (
                "maximum context length" in msg and "messages resulted" in msg
            ):
                return -100.0, {"error": "context_length_exceeded"}
            return None, {"error": "bad_request", "exception": str(e)}

        except Exception as e:
            if attempt >= MAX_RETRIES:
                tqdm.write(f"[ERROR] LLM call failed after {attempt} attempts: {e}")
                return None, {
                    "error": f"LLM call failed after {attempt} attempts",
                    "exception": str(e),
                }
            time.sleep(_sleep_backoff_seconds(attempt))
            continue


# ---------------------------------------------------------------------
# Main ranker: pointwise scores -> ranking (UPDATED init/config)
# ---------------------------------------------------------------------
class LLMPointwiseLogprobRanker:
    """
    Pointwise scoring baseline:
      score(candidate) = P(Yes | query, candidate) computed from token logprobs,
      where the model commits to A/B.

    Works with OpenAI GPT-5.1 via Chat Completions.
    """

    def __init__(
        self,
        model_name: str = OPENAI_MODEL_NAME,
        api_key: str = OPENAI_API_KEY,
        base_url: str = OPENAI_BASE_URL,
        timeout_s: float = 60.0,
    ):
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is missing. Set it in your environment, e.g.:\n"
                "  export OPENAI_API_KEY='...'\n"
            )

        kwargs = {"api_key": api_key, "timeout": timeout_s, "max_retries": 0}
        if base_url:
            kwargs["base_url"] = base_url

        self.client = OpenAI(**kwargs)
        self.model_name = model_name

    def build_candidates(
        self,
        sdr: Dict[str, Any],
        include_types: Iterable[str],
    ) -> List[Dict[str, Any]]:
        return build_candidates_from_sdr(sdr, include_types=include_types)

    def _score_one_candidate(
        self,
        query: str,
        candidate: Dict[str, Any],
        top_logprobs: int,
    ) -> Tuple[str, float]:
        cid = candidate["id"]
        passage = candidate["text"]

        prompt = build_pointwise_prompt(query, passage, label_yes="A", label_no="B")

        p_yes, _debug = _call_llm_pointwise_logprobs(
            self.client,
            self.model_name,
            prompt,
            top_logprobs=top_logprobs,
        )

        if p_yes is None:
            print(f"[WARN] Scoring candidate {cid} failed because {_debug.get('exception')}.")
            return cid, -100.0

        return cid, float(p_yes)

    def rank(
        self,
        query: str,
        sdr: Dict[str, Any],
        include_types: Iterable[str] = ("Paragraph", "Section", "Caption"),
        top_logprobs: int = 20,  # OpenAI supports up to 20
        max_workers: int = 50,
    ) -> Dict[str, float]:
        candidates = self.build_candidates(sdr, include_types=include_types)
        if not candidates:
            return {}

        id2score: Dict[str, float] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._score_one_candidate, query, c, top_logprobs)
                for c in candidates
            ]

            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="LLM pointwise scoring",
                leave=False,
                disable=True,
            ):
                cid, score = fut.result()
                id2score[cid] = score

        return dict(sorted(id2score.items(), key=lambda x: x[1], reverse=True))

    def rank_many(
        self,
        queries,
        sdr,
        top_logprobs: int = 5,
        max_workers: int = 20,
    ) -> Dict[int, Dict[str, float]]:
        
        reset_token_usage()

        results = defaultdict(dict)

        def _task(query_id, query_text, candidate):
            cid = candidate["id"]
            passage = candidate["text"]

            prompt = build_pointwise_prompt(
                query_text, passage, label_yes="A", label_no="B"
            )

            p_yes, _ = _call_llm_pointwise_logprobs(
                self.client,
                self.model_name,
                prompt,
                top_logprobs=top_logprobs,
            )

            if p_yes is None:
                return query_id, cid, -100.0

            return query_id, cid, float(p_yes)

        futures = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for query_id, query_text, _, q_include_types in queries:
                candidates = self.build_candidates(sdr, include_types=q_include_types)
                if not candidates:
                    continue
                for c in candidates:
                    futures.append(executor.submit(_task, query_id, query_text, c))

            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="LLM pointwise (multi-query)",
                leave=False,
                disable=True,
            ):
                qid, cid, score = fut.result()
                results[qid][cid] = score

        for qid in results:
            results[qid] = dict(
                sorted(results[qid].items(), key=lambda x: x[1], reverse=True)
            )

        usage = snapshot_token_usage()
        return results, usage


if __name__ == "__main__":

    with open("example_sdr.json", "r") as f:
        example_sdr = json.load(f)

    query = "The authors didn't explain the influence of freezing different numbers of layers on the performance."

    include_types_to_run = {"Caption"}

    ranker = LLMPointwiseLogprobRanker()
    scores = ranker.rank(
        query,
        example_sdr,
        include_types=include_types_to_run,
        top_logprobs=5,
        max_workers=50,
    )

    tqdm.write("Top ranked candidates:")
    for cid, score in list(scores.items())[:10]:
        tqdm.write(f"  {cid}: {score:.4f}")

    print("Token usage:")
    for k, v in TOKEN_USAGE.items():
        print(f"  {k}: {v}")