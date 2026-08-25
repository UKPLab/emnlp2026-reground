import argparse
import json
import os
import random

from llm_retrieval_logprobs import LLMPointwiseLogprobRanker
from tqdm import tqdm
from utils import extract_and_normalize_references  # canonical shared gold extraction

random.seed(42)

# Hugging Face model IDs (override with a local path via env if you have one cached).
MODEL_MAPPING = {
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "qwen3-30b": "Qwen/Qwen3-30B-A3B",
    "gemma-3-12b": "google/gemma-3-12b-it",
    "gemma-3-27b": "google/gemma-3-27b-it",
    "qwen3-30b-instruct": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "qwen3-30b-thinking": "Qwen/Qwen3-30B-A3B-Thinking-2507",
}

NLPEER_ROOT = os.environ.get("REGROUND_NLPEER", "nlpeer")


def enrich(dataset):
    out = {}
    for pid, pdata in dataset.items():
        out[pid] = []
        for _, r in pdata.items():
            if r.get("aligned"):
                for a in r["aligned"]:
                    a["refs"] = extract_and_normalize_references(a["rebuttal_sentence"])
                    out[pid].append(a)
    return out


def load(path):
    with open(path) as f:
        return json.load(f)


def save(rankings, dataset, model, out_root="results"):
    p = os.path.join(out_root, dataset, "text_retrieval")
    os.makedirs(p, exist_ok=True)
    with open(os.path.join(p, f"{model}_logprobs.json"), "w") as f:
        json.dump(rankings, f, indent=2)


def load_sdr(pid, dataset):
    """dataset is the venue: nlpeer/<venue>/<pid>/v1/paper.sdr.json."""
    try:
        with open(os.path.join(NLPEER_ROOT, dataset, pid, "v1", "paper.sdr.json")) as f:
            return json.load(f)
    except Exception:
        return None


def run(dataset, dataset_name, model, out_root="results"):
    ranker = LLMPointwiseLogprobRanker(MODEL_MAPPING[model])
    rankings = {}

    for pid, rebuttals in tqdm(dataset.items()):
        sdr = load_sdr(pid, dataset_name)
        if not sdr:
            continue

        queries = []
        for i, r in enumerate(rebuttals):
            if r["refs"]:
                queries.append(
                    (i, r["review_spans"], r["rebuttal_sentence"], None)
                )

        if not queries:
            continue

        scores = ranker.rank_many(queries, sdr)
        rankings[pid] = {}

        for i, q, rs, _ in queries:
            rankings[pid][i] = {
                "query": q,
                "rebuttal_sentence": rs,
                "scores": scores.get(i, {}),
            }

        save(rankings, dataset_name, model, out_root)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Open-model logprob retrieval over reground.json queries + nlpeer sdr pools.")
    ap.add_argument("--model_name", default="qwen3-30b-thinking", choices=list(MODEL_MAPPING))
    ap.add_argument("--dataset", default="reground.json")
    ap.add_argument("--venues", nargs="+", default=None, help="default: all venues in the dataset")
    ap.add_argument("--nlpeer", default=None, help="override REGROUND_NLPEER / nlpeer root")
    ap.add_argument("--out-root", default="results")
    args = ap.parse_args()

    if args.nlpeer:
        NLPEER_ROOT = args.nlpeer  # module global used by load_sdr

    data = load(args.dataset)
    venues = args.venues or list(data.keys())
    for name in venues:
        run(enrich(data[name]), name, args.model_name, args.out_root)
