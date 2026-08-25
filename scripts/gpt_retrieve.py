import argparse
import json
import os
import random

from gpt5 import LLMPointwiseLogprobRanker
from tqdm import tqdm
from utils import extract_and_normalize_references  # canonical shared gold extraction

random.seed(42)


def enrich(dataset):
    enriched = {}
    empty = 0
    for paper_id, paper_data in dataset.items():
        enriched[paper_id] = []
        for review_id, review_data in paper_data.items():
            if review_data.get("aligned") is None:
                continue
            for ref in review_data["aligned"]:
                ref["refs"] = extract_and_normalize_references(ref["rebuttal_sentence"])
                ref["id"] = review_id
                enriched[paper_id].append(ref)
        if len(enriched[paper_id]) == 0:
            empty += 1

    if empty > 0:
        print(f"Papers with no extracted references: {empty} out of {len(dataset)}")
    return enriched


def load_dataset(path: str):
    with open(path, "r") as f:
        dataset = json.load(f)
    return dataset


def save_results(rankings, dataset_name, model, out_root="results"):
    output_path = os.path.join(out_root, dataset_name, "text_retrieval")
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, f"{model}_logprobs.json"), "w") as f:
        json.dump(rankings, f, indent=2)

def save_token_usage(token_usage, dataset_name, model, out_root="results"):
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, f"{dataset_name}_token_usage.json"), "w") as f:
        json.dump(token_usage, f, indent=2)


NLPEER_ROOT = os.environ.get("REGROUND_NLPEER", "nlpeer")


def load_sdr(paper_id: str, dataset_name: str) -> dict:
    """Structured document representation for a paper.
    dataset_name is the venue (nlpeer/<venue>/<paper_id>/v1/paper.sdr.json)."""
    path = os.path.join(NLPEER_ROOT, dataset_name, paper_id, "v1", "paper.sdr.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def cot_str(cot: bool) -> str:
    return ("CoT", "_CoT") if cot else ("NoCoT", "_NoCoT")


def print_costs(ds_name, input_tokens, output_tokens):
    input_cost_per_1m = 1.25
    output_cost_per_1m = 10
    input_cost = (input_tokens / 1_000_000) * input_cost_per_1m
    output_cost = (output_tokens / 1_000_000) * output_cost_per_1m
    total_cost = input_cost + output_cost
    print(f"[{ds_name}] Total cost estimate: ${total_cost:.4f}")
    return total_cost

def rank_data_batch_queries(dataset, dataset_name, model_name, max_workers=20, out_root="results"):
    total_token_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "num_requests": 0,
    }
    rankings = {}
    ranker = LLMPointwiseLogprobRanker()

    for paper_id, rebuttals in tqdm(
        dataset.items(),
        desc=f"Ranking {dataset_name} with {model_name}",
    ):
        rankings[paper_id] = {}
        sdr = load_sdr(paper_id=paper_id, dataset_name=dataset_name)

        if sdr is None:
            tqdm.write(f"Skipping paper {paper_id[:30]} due to missing SDR.")
            continue

        # ---- build queries ----
        queries = []

        for idx, entry in enumerate(rebuttals):
            gold = entry.get("refs", [])
            if not gold:
                continue

            gold_types = set([g["type"] for g in gold])
            for gt in gold_types:
                if gt not in {"Paragraph", "Section", "Caption"}:
                    continue

            include_types_to_run = set()
            include_types_to_run.add("Paragraph")
            include_types_to_run.add("Section")
            include_types_to_run.add("Caption")

            queries.append(
                (
                    idx,
                    entry["review_spans"],
                    entry["rebuttal_sentence"],
                    include_types_to_run,
                )
            )

        if not queries or not include_types_to_run:
            continue

        # ---- ONE big batched call ----
        all_scores, promot_token_usage = ranker.rank_many(
            queries=queries,
            sdr=sdr,
            top_logprobs=5,
            max_workers=max_workers,
        )

        # ---- unpack ----
        for idx, query_text, rebuttal_sentence, _ in queries:
            rankings[paper_id][idx] = {
                "id": rebuttals[idx]["id"],
                "query": query_text,
                "rebuttal_sentence": rebuttal_sentence,
                "gold": rebuttals[idx]["refs"],
                "scores": all_scores.get(idx, {}),
            }

        for k, v in promot_token_usage.items():
            total_token_usage[k] += v

        save_results(rankings, dataset_name, model=model_name, out_root=out_root)

    save_results(rankings, dataset_name, model=model_name, out_root=out_root)
    print_costs(dataset_name, total_token_usage["prompt_tokens"], total_token_usage["completion_tokens"])
    save_token_usage(total_token_usage, dataset_name, model=model_name, out_root=out_root)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="GPT-5.1 pointwise-logprob retrieval over reground.json queries + nlpeer sdr pools.")
    ap.add_argument("--dataset", default="reground.json")
    ap.add_argument("--venues", nargs="+", default=None, help="default: all venues in the dataset")
    ap.add_argument("--nlpeer", default=None, help="override REGROUND_NLPEER / nlpeer root")
    ap.add_argument("--out-root", default="results")
    ap.add_argument("--model-name", default="GPT-5.1")
    ap.add_argument("--max-workers", type=int, default=20)
    args = ap.parse_args()

    if args.nlpeer:
        NLPEER_ROOT = args.nlpeer  # noqa: F811  (module global used by load_sdr)

    reground = load_dataset(args.dataset)
    venues = args.venues or list(reground.keys())

    # NOTE: GPT-5.1 is a paid API; the paper ran it on a random subset for cost.
    # Pass --venues and/or pre-subset reground.json if you want to limit the run.
    for dataset_name in venues:
        enriched_dataset = enrich(reground[dataset_name])
        tqdm.write(f"Starting ranking for venue: {dataset_name}")
        rank_data_batch_queries(
            enriched_dataset, dataset_name,
            model_name=args.model_name, max_workers=args.max_workers, out_root=args.out_root,
        )
