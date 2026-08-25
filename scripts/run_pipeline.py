#!/usr/bin/env python3
"""Reproduce reground.json from nlpeer/ by running the construction pipeline per venue.

Chain (all flat / paper-keyed inside a venue; the driver nests the result by venue):
  threads_builder -> regex_detector -> regex_no_quotes (Filter 1: quotes)
  -> review_alignment (LLM) -> unique_refs (Filter 2) -> llm_filter (LLM, Filter 3)
  -> merge venues -> reground.json

Prerequisites
-------------
- Input per paper: nlpeer/<venue>/<id>/v1/{reviews.json, comments.json}. These are part of
  the original NLPeer release. paper.sdr.json, extracted images, paragraphs.json and output_json
  are NOT needed to build reground.json — they are produced by the PDF-processing scripts
  (parse_papers.py, run_grob.py, cleanpdf.py + PDFFigures 2.0) and are only used downstream for
  evidence resolution (build_evidence.py) and the retrieval experiments. If you start from a raw
  NLPeer (no sdr/images) run that PDF-processing step first for the downstream tasks.
- LLM stages (review_alignment, llm_filter) need an OpenAI-compatible endpoint. Configure via env:
    REGROUND_LLM_BASE_URL   (default http://localhost:6531/v1)
    REGROUND_LLM_API_KEY    (default "")
    REGROUND_LLM_MODEL      alignment model (default gpt-oss-120b)
    REGROUND_FILTER_MODEL   filter model    (default gpt-oss-20b)

Use --stop-after quotes to run only the non-LLM prefix (no endpoint required).
"""

import argparse
import json
import os

import threads_builder
import regex_detector
import regex_no_quotes
import review_alignment
import unique_refs
import llm_filter

DEFAULT_VENUES = ["EMNLP24", "NAACL25", "ACL25", "COLING25", "EMNLP25"]


def run_venue(venue, nlpeer, workdir, batch_size, stop_after=None):
    """Run the pipeline for one venue; return its grounded-references dict (or None
    if stopped before the LLM stages)."""
    rd = os.path.join(workdir, venue)
    os.makedirs(rd, exist_ok=True)
    venue_dir = os.path.join(nlpeer, venue)

    threads_builder.main(data_path=venue_dir, results_dir=rd)
    regex_detector.main(results_dir=rd)
    regex_no_quotes.main(results_dir=rd)
    if stop_after == "quotes":
        return None

    review_alignment.main(results_dir=rd, batch_size=batch_size)
    unique_refs.main(results_dir=rd)
    llm_filter.main(results_dir=rd, batch_size=batch_size)
    with open(os.path.join(rd, "5_grounded_references.json"), encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nlpeer", default="nlpeer", help="path to the nlpeer/ root")
    ap.add_argument("--out", default="reground.json", help="merged, venue-keyed output")
    ap.add_argument("--workdir", default="_pipeline", help="per-venue intermediate files")
    ap.add_argument("--venues", nargs="+", default=DEFAULT_VENUES)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--stop-after", choices=["quotes"], default=None,
                    help="'quotes' runs only the non-LLM prefix (no endpoint needed)")
    args = ap.parse_args()

    merged = {}
    for venue in args.venues:
        print(f"\n===== {venue} =====")
        res = run_venue(venue, args.nlpeer, args.workdir, args.batch_size, args.stop_after)
        if res is not None:
            merged[venue] = res

    if args.stop_after is None:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False)
        papers = sum(len(v) for v in merged.values())
        print(f"\nwrote {args.out}: {len(merged)} venues, {papers} papers")


if __name__ == "__main__":
    main()
