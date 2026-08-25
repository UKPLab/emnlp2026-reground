import json
import os
import re

import get_from_paper as gfp
import numpy as np
import tiktoken
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from utils import extract_and_normalize_references  # canonical shared gold extraction

# Uses TYPE_MAP from your code
TYPE_MAP = {
    "Line": ["Paragraph"],
    "Section": ["Section"],
    "Table": ["Table"],
    "Figure": ["Figure"],
}


def expand_line_gold_tmp(gold_label):
    label = gold_label.replace("Line", "").strip()

    # Parse line numbers
    if "-" in label:
        start, end = map(int, label.split("-"))
        gold_lines = list(range(start, end + 1))
    else:
        gold_lines = [int(label)]

    return gold_lines


def get_by_type(sdr, ref_type, ref_label):
    doc_texts = []

    if ref_type == "Line":
        gold_lines = expand_line_gold_tmp(ref_label)
        para_ids = gfp.get_paragraph_ids_by_line(sdr, gold_lines)
        for para in sdr["paragraphs"]:
            if para["paragraph_id"] in para_ids:
                para_text = gfp.get_paragraph_by_id(sdr, para["paragraph_id"])["text"]
                doc_texts.append(para_text)
        return doc_texts

    elif ref_type == "Section":
        sec_text = gfp.get_section_text_by_number(sdr, ref_label)
        doc_texts.append(sec_text)
        return doc_texts

    elif ref_type in ["Figure", "Table"]:
        for caption_id, caption_text in sdr["captions"].items():
            if caption_id == ref_type + " " + ref_label:
                doc_texts.append(caption_text)
        return doc_texts

    else:
        return None




NLPEER_ROOT = os.environ.get("REGROUND_NLPEER", "nlpeer")


def load_sdr(paper_id, dataset_name):
    """dataset_name is the venue: nlpeer/<venue>/<paper_id>/v1/paper.sdr.json."""
    path = os.path.join(NLPEER_ROOT, dataset_name, paper_id, "v1", "paper.sdr.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _empty_type_stats():
    return {"Paragraph": [], "Section": [], "Figure": [], "Table": []}


def flatten_stats(stats_by_dataset):
    """Flatten stats[dataset_name][type] into one list of cosine values."""
    vals = []
    for type_stats in stats_by_dataset.values():
        for sims in type_stats.values():
            vals.extend(sims)
    return vals


def average_cosine(stats_by_dataset):
    vals = flatten_stats(stats_by_dataset)
    return float(np.mean(vals)) if vals else None


def count_cosines(stats_by_dataset):
    return len(flatten_stats(stats_by_dataset))


def summarize_stats(stats_by_dataset):
    """
    Returns a compact summary with overall and per-type mean/count.
    Useful for directly reporting the paper number.
    """
    summary = {
        "overall": {
            "mean": average_cosine(stats_by_dataset),
            "n": count_cosines(stats_by_dataset),
        },
        "by_type": {},
    }

    for key in ["Paragraph", "Section", "Figure", "Table"]:
        vals = []
        for dataset_stats in stats_by_dataset.values():
            vals.extend(dataset_stats.get(key, []))
        summary["by_type"][key] = {
            "mean": float(np.mean(vals)) if vals else None,
            "n": len(vals),
        }

    return summary


def compute_similarity_stats_by_type(
    complete_dataset,
    query_field="review_spans",
    mean_length=None,
    std_length=None,
    outlier_sigma=4.0,
    batch_size=32,
    device=None,  # e.g. "cuda" if you want to force it
    length_filter_field="review_spans",
):
    """
    Backward-compatible single-field version.

    By default this reproduces your original computation:
      review_spans -> referenced paper text.

    To compute the rebuttal/reference-sentence baseline, call with:
      query_field="rebuttal_sentence"

    Important:
      length_filter_field defaults to "review_spans" so the rebuttal-sentence
      comparison is computed on the same filtered alignment set as the original
      review-span comparison. That makes the two averages more comparable.
    """
    stats_by_field = compute_similarity_stats_by_type_for_fields(
        complete_dataset,
        query_fields=(query_field,),
        mean_length=mean_length,
        std_length=std_length,
        outlier_sigma=outlier_sigma,
        batch_size=batch_size,
        device=device,
        length_filter_field=length_filter_field,
    )
    return stats_by_field[query_field]


def compute_similarity_stats_by_type_for_fields(
    complete_dataset,
    query_fields=("review_spans", "rebuttal_sentence"),
    mean_length=None,
    std_length=None,
    outlier_sigma=4.0,
    batch_size=32,
    device=None,  # e.g. "cuda" if you want to force it
    length_filter_field="review_spans",
):
    """
    Computes cosine similarity between one or more alignment text fields and
    the paper text referenced by the rebuttal sentence.

    Returns:
      stats_by_field[field][dataset_name][key] = [cos_sim, cos_sim, ...]
      where key in {"Paragraph", "Section", "Figure", "Table"}
      and "Line" references are mapped to "Paragraph".

    Example fields:
      - "review_spans": original reviewer comment span
      - "rebuttal_sentence": sentence that contains the evidence reference

    Notes:
      - References are still extracted from alignment["rebuttal_sentence"],
        exactly as in your original code.
      - length_filter_field="review_spans" preserves the original sample filter,
        so review-span and rebuttal-sentence averages are directly comparable.
    """

    model = SentenceTransformer(
        "sentence-transformers/all-mpnet-base-v2", device=device
    )

    enc = tiktoken.get_encoding("cl100k_base")

    # Cache embeddings by exact text. Shared across fields to avoid duplicate work.
    text_cache = {}
    doc_cache = {}

    stats_by_field = {field: {} for field in query_fields}

    for dataset_name, dataset in complete_dataset.items():
        for field in query_fields:
            stats_by_field[field].setdefault(dataset_name, _empty_type_stats())

        for paper_id, paper_data in tqdm(
            dataset.items(), desc=f"Processing {dataset_name}"
        ):
            # Load SDR once per paper.
            sdr = load_sdr(paper_id, dataset_name)
            if sdr is None:
                continue

            for _, rebuttal in paper_data.items():
                for alignment in rebuttal.get("aligned", []):

                    # -------------------------
                    # Optional outlier filter
                    # -------------------------
                    # Default: filter using review_spans, as in the original code.
                    # This keeps review_spans and rebuttal_sentence comparisons
                    # on the same set of alignments.
                    if length_filter_field:
                        filter_text = alignment.get(length_filter_field, "")
                        if not filter_text:
                            continue

                        if (
                            mean_length is not None
                            and std_length is not None
                            and std_length > 0
                        ):
                            q_len = len(enc.encode(filter_text))
                            if abs(q_len - mean_length) > outlier_sigma * std_length:
                                continue

                    # -------------------------
                    # Extract references from rebuttal sentence
                    # -------------------------
                    rebuttal_sentence = alignment.get("rebuttal_sentence", "")
                    refs = extract_and_normalize_references(rebuttal_sentence)
                    if not refs:
                        continue

                    for ref in refs:
                        ref_type = ref.get("type")
                        ref_label = ref.get("label")

                        # Only types you care about.
                        if ref_type not in TYPE_MAP:
                            continue

                        key = "Paragraph" if ref_type == "Line" else ref_type

                        # -------------------------
                        # Resolve reference -> doc_texts
                        # -------------------------
                        try:
                            doc_texts = get_by_type(sdr, ref_type, ref_label)
                        except Exception:
                            continue
                        if not doc_texts:
                            continue

                        # -------------------------
                        # Embed referenced paper text, cached + batched
                        # -------------------------
                        to_encode = []
                        cached_doc_embs = []

                        for dt in doc_texts:
                            if not dt:
                                continue
                            emb = doc_cache.get(dt)
                            if emb is None:
                                to_encode.append(dt)
                            else:
                                cached_doc_embs.append(emb)

                        new_doc_embs = []
                        if to_encode:
                            new_doc_embs = model.encode(
                                to_encode,
                                normalize_embeddings=True,
                                batch_size=batch_size,
                            )
                            for dt, emb in zip(to_encode, new_doc_embs):
                                doc_cache[dt] = emb

                        all_doc_embs = cached_doc_embs + (
                            list(new_doc_embs) if len(to_encode) else []
                        )
                        if not all_doc_embs:
                            continue

                        # -------------------------
                        # Compute cosine for each requested query field
                        # -------------------------
                        for field in query_fields:
                            query_text = alignment.get(field, "")
                            if not query_text:
                                continue

                            q_emb = text_cache.get(query_text)
                            if q_emb is None:
                                q_emb = model.encode(
                                    query_text, normalize_embeddings=True
                                )
                                text_cache[query_text] = q_emb

                            for d_emb in all_doc_embs:
                                sim = float(np.dot(q_emb, d_emb))
                                if sim < 0:
                                    sim = 0.0  # clamp negatives due to numerical issues
                                stats_by_field[field][dataset_name][key].append(sim)

    return stats_by_field


def print_summary(stats_by_field):
    print("\nAverage cosine similarity against referenced paper text")
    print("=" * 64)

    for field, stats_by_dataset in stats_by_field.items():
        summary = summarize_stats(stats_by_dataset)
        overall = summary["overall"]
        mean = overall["mean"]
        n = overall["n"]

        if mean is None:
            print(f"{field}: no resolved references")
            continue

        print(f"{field}: {mean:.3f}  (n={n})")

        for ref_type, type_summary in summary["by_type"].items():
            type_mean = type_summary["mean"]
            type_n = type_summary["n"]
            if type_mean is not None:
                print(f"  {ref_type}: {type_mean:.3f}  (n={type_n})")
        print()


def main():
    dataset_path = os.environ.get("REGROUND_DATASET", "reground.json")
    complete_dataset = json.load(open(dataset_path, "r", encoding="utf-8"))

    mean_length, std_length = (
        np.float64(52.351212576032864),
        np.float64(33.70640667245876),
    )

    # Computes both numbers in one pass:
    #   1. review_spans -> referenced paper text  [your original number]
    #   2. rebuttal_sentence -> referenced paper text  [baseline/context number]
    #
    # If your dataset field is literally called "referenced_sentence" instead,
    # change "rebuttal_sentence" below to "referenced_sentence".
    query_fields = ("review_spans", "rebuttal_sentence")

    stats_by_field = compute_similarity_stats_by_type_for_fields(
        complete_dataset,
        query_fields=query_fields,
        mean_length=mean_length,
        std_length=std_length,
        outlier_sigma=4.0,
        batch_size=32,
        device="cuda" if torch.cuda.is_available() else "cpu",
        # Keep this as review_spans for fair comparison to your original 0.395.
        length_filter_field="review_spans",
    )

    # Store full stats for both sources.
    with open("similarity_stats_by_field_and_type.json", "w", encoding="utf-8") as f:
        json.dump(stats_by_field, f, indent=2)

    # Store your original output shape too, for compatibility with old plotting code.
    if "review_spans" in stats_by_field:
        with open("similarity_stats_by_type.json", "w", encoding="utf-8") as f:
            json.dump(stats_by_field["review_spans"], f, indent=2)

    # Store compact reportable summary.
    summary_by_field = {
        field: summarize_stats(stats) for field, stats in stats_by_field.items()
    }
    with open("similarity_summary_by_field.json", "w", encoding="utf-8") as f:
        json.dump(summary_by_field, f, indent=2)

    # Print the numbers you need for the paper immediately.
    print_summary(stats_by_field)


if __name__ == "__main__":
    main()
