import json
import math
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import get_from_paper
from openai import OpenAI
from tqdm import tqdm

random.seed(42)

VLLM_BASE_URL = "http://localhost:6531/v1"
VLLM_API_KEY = ""
MAX_RETRIES = 5


def get_by_type(sdr, ref_type):
    doc_ids = []
    doc_texts = []

    if ref_type == "Line":
        for ln, text in sdr["lines"].items():
            doc_ids.append(f"Line {ln}")
            doc_texts.append(text)
        return doc_ids, doc_texts

    if ref_type == "Paragraph":
        for para in sdr["paragraphs"]:
            doc_ids.append(f"Paragraph {para['paragraph_id']}")
            para_text = get_from_paper.get_text_by_line(
                sdr, (para["start_line"], para["end_line"])
            )
            doc_texts.append(para_text)
        return doc_ids, doc_texts

    if ref_type == "Section":
        for section in sdr["sections"]:
            doc_ids.append(f"Section {section['number']} / {section['name']}")
            sec_text = get_from_paper.get_section_text_by_number(
                sdr, section["number"]
            )
            doc_texts.append(sec_text)
        return doc_ids, doc_texts

    if ref_type == "Caption":
        return list(sdr["captions"].keys()), list(sdr["captions"].values())

    raise ValueError(ref_type)


def build_candidates_from_sdr(sdr, include_types):
    candidates = []
    for ref_type in include_types:
        try:
            doc_ids, doc_texts = get_by_type(sdr, ref_type)
        except Exception:
            continue

        for cid, text in zip(doc_ids, doc_texts):
            if text:
                t = text.strip()
                if t:
                    candidates.append({"id": cid, "text": t})
    return candidates


def build_pointwise_prompt(query, content, label_yes="A", label_no="B"):
    return f"""You are evaluating whether content from an NLP paper helps addressing a peer-review comment.

Review comment:
{query}

Paper content:
{content}

Instructions:
- Output a final answer on a new line in EXACTLY ONE character: {label_yes} or {label_no}.
- Do not output anything after that one character.

Meaning:
{label_yes} = Yes
{label_no} = No

Output:"""


def _as_dict(obj):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return obj


def _normalize_token(t):
    return t.strip()


def _logsumexp(a, b):
    m = max(a, b)
    if m == -math.inf:
        return -math.inf
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def _extract_token_logprobs(choice):
    lp = _as_dict(getattr(choice, "logprobs", None))
    if not lp:
        return []

    content = lp.get("content")
    if not isinstance(content, list):
        return []

    out = []
    for item in content:
        item = _as_dict(item)
        token = item.get("token")
        logprob = item.get("logprob")
        top = item.get("top_logprobs", [])

        top_map = {}
        if isinstance(top, dict):
            for k, v in top.items():
                if isinstance(k, str):
                    top_map[k] = float(v)
        elif isinstance(top, list):
            for t in top:
                t = _as_dict(t)
                if "token" in t and "logprob" in t:
                    top_map[t["token"]] = float(t["logprob"])

        if isinstance(token, str):
            out.append(
                {
                    "token": token,
                    "logprob": float(logprob),
                    "top_logprobs": top_map,
                }
            )
    return out


def _pick_decision_index(text, tokens, yes, no):
    labels = {yes, no}
    text = (text or "").rstrip()
    if text and text[-1] in labels:
        for i in range(len(tokens) - 1, -1, -1):
            if _normalize_token(tokens[i]["token"]) == text[-1]:
                return i

    for i in range(len(tokens) - 1, -1, -1):
        if _normalize_token(tokens[i]["token"]) in labels:
            return i
    return None


def _prob_yes_from_choice(choice):
    msg = getattr(choice, "message", None)
    text = getattr(msg, "content", "") if msg else ""

    tokens = _extract_token_logprobs(choice)
    if not tokens:
        return None

    idx = _pick_decision_index(text, tokens, "A", "B")
    if idx is None:
        return None

    ti = tokens[idx]
    norm_top = {}

    for k, v in ti["top_logprobs"].items():
        nk = _normalize_token(k)
        norm_top[nk] = max(norm_top.get(nk, -math.inf), v)

    lp_a = norm_top.get("A", -math.inf)
    lp_b = norm_top.get("B", -math.inf)

    emitted = _normalize_token(ti["token"])
    if lp_a == -math.inf and emitted == "A":
        lp_a = ti["logprob"]
    if lp_b == -math.inf and emitted == "B":
        lp_b = ti["logprob"]

    if lp_a == -math.inf and lp_b == -math.inf:
        return None

    return math.exp(lp_a - _logsumexp(lp_a, lp_b))


def _call_llm(client, model, prompt, top_logprobs):
    for _ in range(MAX_RETRIES):
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Follow instructions exactly."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,
            top_p=0.95,
            logprobs=True,
            top_logprobs=top_logprobs,
        )
        p = _prob_yes_from_choice(resp.choices[0])
        if p is not None:
            return p
    return None


class LLMPointwiseLogprobRanker:
    def __init__(self, model_name, base_url=VLLM_BASE_URL):
        self.client = OpenAI(base_url=base_url, api_key=VLLM_API_KEY)
        self.model_name = model_name

    def rank_many(self, queries, sdr, top_logprobs=10, max_workers=50):
        results = defaultdict(dict)
        candidates = build_candidates_from_sdr(
            sdr, {"Paragraph", "Section", "Caption"}
        )

        def task(qid, query, cand):
            p = _call_llm(
                self.client,
                self.model_name,
                build_pointwise_prompt(query, cand["text"]),
                top_logprobs,
            )
            return qid, cand["id"], p if p is not None else -100.0

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [
                ex.submit(task, qid, q, c)
                for qid, q, _, _ in queries
                for c in candidates
            ]

            for f in as_completed(futs):
                qid, cid, score = f.result()
                results[qid][cid] = score

        for qid in results:
            results[qid] = dict(
                sorted(results[qid].items(), key=lambda x: x[1], reverse=True)
            )
        return results
