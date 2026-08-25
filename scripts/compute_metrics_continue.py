import argparse
import gc
import json
import os

import retrieval_methods
import torch
from tqdm import tqdm
from utils import extract_and_normalize_references  # canonical shared gold extraction


def enrich(dataset):
    enriched = {}
    empty = 0
    for paper_id, paper_data in dataset.items():
        enriched[paper_id] = []
        for _, review_data in paper_data.items():
            if review_data.get("aligned") is None:
                continue
            for ref in review_data["aligned"]:
                ref["refs"] = extract_and_normalize_references(ref["rebuttal_sentence"])
                enriched[paper_id].append(ref)
        if len(enriched[paper_id]) == 0:
            empty += 1

    tqdm.write(f"Papers with no extracted references: {empty} out of {len(dataset)}")
    return enriched


def rank_model_by_model(enriched_data, dataset_name, out_dir, retrievers):
    os.makedirs(out_dir, exist_ok=True)

    for name, factory in retrievers:
        tqdm.write(f"[INFO] Running retriever: {name}")
        out_path = os.path.join(out_dir, f"{name}.json")
        outputs = {}

        if os.path.exists(out_path):
            with open(out_path, "r", encoding="utf-8") as f:
                outputs = json.load(f)
            tqdm.write(f"[INFO] Resuming from {out_path}, {len(outputs)} papers done")

        retriever = None  # instantiate once (first valid paper)

        for paper_id, rebuttals in tqdm(
            enriched_data.items(), desc=f"Ranking ({name})"
        ):
            
            if paper_id in outputs:
                continue  # already done

            sdr = retrieval_methods.load_sdr(paper_id, dataset_name)
            if sdr is None:
                tqdm.write(f"[WARNING] Skipping paper {paper_id} due to missing SDR.")
                continue

            if retriever is None:
                retriever = factory(sdr)
            else:
                # reuse same model; MUST clear cache per paper
                if hasattr(retriever, "set_sdr"):
                    retriever.set_sdr(sdr)
                else:
                    # fallback: replace SDR + nuke known cache field if present
                    retriever.sdr = sdr
                    if hasattr(retriever, "cache"):
                        retriever.cache.clear()

            outputs.setdefault(paper_id, {})

            for q_idx, entry in enumerate(rebuttals):
                query_text = entry["review_spans"]
                gold = entry.get("refs", [])

                with torch.inference_mode():
                    outputs[paper_id][q_idx] = {
                        "query": query_text,
                        "gold": gold,
                        "scores": retriever.rank(query_text),
                    }

            # save after each paper (crash-safe)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(outputs, f, indent=2, ensure_ascii=False)

        # hard cleanup between models
        del retriever
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


def main(dataset_name, dataset, retrievers, out_root="results"):
    enriched_data = enrich(dataset)
    rank_model_by_model(
        enriched_data,
        dataset_name,
        os.path.join(out_root, dataset_name, "text_retrieval"),
        retrievers,
    )


if __name__ == "__main__":

    retrievers_all = [
        ("mpnet", lambda sdr: retrieval_methods.MPNetRetriever(sdr)),
        ("bgem3", lambda sdr: retrieval_methods.BGEM3Retriever(sdr)),
        ("minilm_ce", lambda sdr: retrieval_methods.MiniLMCrossEncoderRetriever(sdr)),
        ("bgem3_ce", lambda sdr: retrieval_methods.BGEM3RerankerRetriever(sdr)),
        ("bm25", lambda sdr: retrieval_methods.BM25Retriever(sdr)),
        ("splade", lambda sdr: retrieval_methods.SPLADEv3Retriever(sdr)),
        ("qwen3", lambda sdr: retrieval_methods.Qwen3EmbeddingRetriever(sdr)),
        ("gemma300", lambda sdr: retrieval_methods.GEMMA300Retriever(sdr)),
    ]

    argparser = argparse.ArgumentParser(
        description="Text retrieval over reground.json queries + nlpeer sdr pools."
    )
    argparser.add_argument("--retrievers", nargs="+", required=True,
                           help=f"any of: {', '.join(n for n, _ in retrievers_all)}")
    argparser.add_argument("--dataset", default="reground.json")
    argparser.add_argument("--venues", nargs="+", default=None, help="default: all venues in the dataset")
    argparser.add_argument("--nlpeer", default=None, help="override REGROUND_NLPEER / nlpeer root")
    argparser.add_argument("--out-root", default="results")
    args = argparser.parse_args()

    if args.nlpeer:
        retrieval_methods.NLPEER_ROOT = args.nlpeer

    selected = set(args.retrievers)
    retrievers = [(n, f) for (n, f) in retrievers_all if n in selected]

    reground = json.load(open(args.dataset, "r", encoding="utf-8"))
    venues = args.venues or list(reground.keys())

    for dataset_name in venues:
        tqdm.write(f"[INFO] Processing venue: {dataset_name}")
        main(dataset_name, reground[dataset_name], retrievers, args.out_root)
        tqdm.write(f"[INFO] Completed venue: {dataset_name}")
        tqdm.write("-" * 40)
