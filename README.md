# ReGround

Reviewer comment grounding as retrieval. ReGround links reviewer comments to localized,
multimodal evidence (paragraphs, sections, figures, tables) in the original anonymous
submission, mined automatically from author rebuttals. This repository contains the dataset
and the code to reproduce it and to run every retrieval task in the paper.

## Contents

```
reground.json            # the dataset: venue -> paper_id -> review -> {review, rebuttal, aligned[...]}
nlpeer/                  # per paper: v1/{reviews.json, comments.json, meta.json, paper.pdf,
                         #             paper.sdr.json, images/, paragraphs.json}
scripts/                 # dataset construction, retrieval runners, evaluation, stats
requirements.txt
SCHEMA.md                # dataset schema (also the derived dataset.jsonl view)
CODEBASE_MAP.md          # what each script is and how the pieces fit
```

`nlpeer/` is NLPeer v2 (EMNLP 24/25, COLING 25, NAACL 25, ACL 25) augmented with the parsed
`paper.sdr.json`, extracted figure/table `images/`, and GROBID `paragraphs.json`. The original
NLPeer release does not include those; we produce them in the preprocessing step below and ship
them here so the retrieval tasks run without re-parsing PDFs.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The dense, cross-encoder, and multimodal retrievers require a GPU. The two LLM construction
stages and the open-model retrievers require an OpenAI-compatible endpoint (e.g. vLLM serving
`gpt-oss`); GPT-5.1 retrieval uses the OpenAI API.

## Environment variables

| Variable | Used by | Default |
|---|---|---|
| `REGROUND_NLPEER` | all runners (sdr/image paths) | `nlpeer` |
| `REGROUND_DATASET` | stats script | `reground.json` |
| `REGROUND_LLM_BASE_URL` | construction LLM stages | `http://localhost:6531/v1` |
| `REGROUND_LLM_API_KEY` | construction LLM stages | `EMPTY` |
| `REGROUND_LLM_MODEL` | alignment stage | `gpt-oss-120b` |
| `REGROUND_FILTER_MODEL` | filter stage | `gpt-oss-120b` |

Most runners also accept `--nlpeer`, `--venues`, and `--out-root` on the command line.

*NOTE:* for reproducibility you can use `gpt-oss-120b`. However, if you are extending the dataset with the latest releases from NLPeer, then we recommend replacing it with the latest SOTA open-weights model available to you.

## 1. Reproduce the dataset (optional)

`reground.json` is provided. To rebuild it from `nlpeer/`:

```bash
cd scripts
python run_pipeline.py --nlpeer ../nlpeer --out ../reground.json
# non-LLM prefix only (no endpoint needed):
python run_pipeline.py --nlpeer ../nlpeer --stop-after quotes
```

Stages: threads -> reference detection -> quote filter -> LLM alignment -> trivial-reference
filter -> LLM evidence filter, per venue, merged by venue. Only `reviews.json` and
`comments.json` are required as input; sdr/images are not needed for this step.

### Preprocessing from raw PDFs (only if starting from a fresh/newer NLPeer)

If your `nlpeer/` lacks `paper.sdr.json` and `images/`, generate them first:
`run_grob.py` (GROBID -> `paragraphs.json`), `parse_papers.py` (pypdf -> `paper.sdr.json`),
plus PDFFigures 2.0 for the figure/table images.

## 2. Retrieval tasks

All runners read `reground.json` as the query/gold source and the candidate pool from
`nlpeer/<venue>/<id>/v1/`. Rankings are written to `<out-root>/<venue>/{text,image}_retrieval/`.

```bash
cd scripts

# Sparse / dense / cross-encoder (needs GPU for dense/CE)
python compute_metrics_continue.py --retrievers bm25 splade bgem3 mpnet minilm_ce bgem3_ce qwen3 gemma300

# Open-model LLM retrieval (needs an OpenAI-compatible server)
python run_llm_retrieval.py --model_name qwen3-30b-instruct

# GPT-5.1 retrieval (paid API; the paper used a random subset for cost)
OPENAI_API_KEY=... python gpt_retrieve.py --venues COLING25

# Multimodal (figures/tables; needs GPU)
python image_compute_metrics.py --models siglip2 openclip_l14 eva02_clip_l14 jina_embeddings_v4 colpali_v1_3 colqwen2_5_v0_2 qwen3_vl_8b_yesno qwen3_vl_32b_yesno
```

Retriever/model keys: text retrievers are listed by `compute_metrics_continue.py --help`; open
LLM models by `run_llm_retrieval.py --help`; image model keys live in
`image_methods.RANKER_FACTORIES`.

## 3. Evaluation

The `evaluate_*` scripts read the per-model ranking files from step 2 and report Recall@k with
exact and relaxed (ancestor / container) matching:

```bash
python evaluate_type_filtered_retrieval.py --results-dir ../results
python evaluate_ancestor_relaxed.py --results ../results --data ../nlpeer
python evaluate_oracle_section_container.py
```

## 4. Evidence-type inference

`run_content_type_inference.py` (LLM content-type prediction), `train_mpnet_content_type_classifier.py`
/ `train_embedding_content_type_classifier.py` (trained baselines), with
`prepare_retrieval_content_type_data.py`, `split_content_type_dataset.py`, and
`evaluate_content_type_predictions.py` for data prep and scoring.

## 5. Dataset statistics and figures

```bash
python analyze_evidence_statistics.py     # Table 2 (papers, queries, evidence, combinations)
python analyze_reference_coverage.py
python similarity_review_rebuttal_stats.py
```