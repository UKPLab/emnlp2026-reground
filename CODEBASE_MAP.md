# Code overview

All scripts live in `scripts/` and are run from there. They read `reground.json` (the annotations)
and `nlpeer/<venue>/<paper_id>/v1/` (papers, reviews, parsed structure, images); see
[SCHEMA.md](SCHEMA.md) for both formats. Paths default to `reground.json` and `nlpeer` and can be
changed with `--dataset`, `--nlpeer` or the `REGROUND_*` environment variables listed in the
[README](README.md).

```
nlpeer/  ──(1) PDF processing──▶  paper.sdr.json, images/
   │
   └──(2) dataset construction──▶  reground.json
                                        │
                     (3) retrieval ◀────┘  queries = review spans, gold = references in rebuttal sentences
                          │
                     results/<venue>/{text,image}_retrieval/<model>.json
                          │
                     (4) evaluation, (5) evidence-type inference, (6) statistics
```

## 1. PDF processing

Only needed when starting from a raw NLPeer release; the shipped `nlpeer/` already contains the
outputs.

| Script | What it does |
|---|---|
| `cleanpdf.py` | Removes the line-number ruler from `paper.pdf` → `paper_clean.pdf`. |
| `grobid_script.sh` | Starts a GROBID Docker container and checks it is reachable. |
| `run_grob.py` | Sends `paper_clean.pdf` to GROBID → `paragraphs.json`. |
| `parse_papers.py` | Builds `paper.sdr.json`: line-numbered text from `paper.pdf`, GROBID paragraphs mapped to line ranges, sections, captions. |
| `get_from_paper.py` | Library: fetch text from an sdr by lines, paragraph, section or caption. |

Figure and table images come from [PDFFigures 2.0](https://github.com/allenai/pdffigures2),
which is not part of this repository.

## 2. Dataset construction

`run_pipeline.py` runs these stages per venue and merges the results into `reground.json`.
Intermediate files go to `_pipeline/<venue>/`.

| Stage | Script | What it does |
|---|---|---|
| Threads | `threads_builder.py` | Pairs each author rebuttal with the reviewer turn it replies to (`reviews.json` + `comments.json`). |
| Detection | `regex_detector.py` | Keeps rebuttals whose sentences refer explicitly to the paper (section, line, table, figure, …). |
| Filter 1 | `regex_no_quotes.py` | Drops rebuttal sentences that (fuzzily) quote the review, so the reference is the reviewer's, not the authors'. |
| Alignment | `review_alignment.py` | LLM: links each reference-bearing rebuttal sentence to the review span it answers. |
| Filter 2 | `unique_refs.py` | Drops pairs where the review span already mentions a reference the rebuttal sentence cites (trivially solvable). |
| Filter 3 | `llm_filter.py` | LLM: keeps sentences grounded in the original submission; drops promised changes, new rebuttal results and external citations. |

`utils.py` holds the shared reference regex, the sentence splitter that does not split inside
references, and `extract_and_normalize_references`, which turns a rebuttal sentence into gold
references for all later steps.

The two LLM stages need an OpenAI-compatible endpoint (e.g. vLLM serving `gpt-oss`).
`--stop-after quotes` runs only the stages before them.

## 3. Retrieval

Each runner writes one ranking file per model to `results/<venue>/{text,image}_retrieval/`.

| Family | Runner | Models (library) |
|---|---|---|
| Sparse, dense, cross-encoder | `compute_metrics_continue.py` | BM25, SPLADE-v3, BGE-M3, MPNet, MiniLM and BGE-M3 cross-encoders, Qwen3-Embedding, EmbeddingGemma (`retrieval_methods.py`) |
| Open LLMs | `run_llm_retrieval.py` | Qwen3-30B, Gemma 3, gpt-oss, scored by yes/no log-probabilities (`llm_retrieval_logprobs.py`) |
| GPT-5.1 | `gpt_retrieve.py` | GPT-5.1 via the OpenAI API (`gpt5.py`); paid, so the paper used a subset |
| Multimodal | `image_compute_metrics.py` | SigLIP 2, OpenCLIP, EVA02-CLIP, Jina Embeddings v4, ColPali, ColQwen2.5, Qwen3-VL (`image_methods.py`) |

Text retrieval ranks lines, paragraphs, sections and captions from `paper.sdr.json`. Multimodal
retrieval ranks the images in `images/` and uses only queries whose gold includes a figure or
table. `compute_metrics_continue.py` resumes from existing output files.

## 4. Evaluation

All report Recall@k on the stored rankings.

| Script | Setting |
|---|---|
| `evaluate_type_filtered_retrieval.py` | Candidates filtered by the predicted evidence type (from §5). |
| `evaluate_ancestor_relaxed.py` | Exact match, and relaxed match that also credits a containing unit (line → paragraph → section). |
| `evaluate_oracle_section_container.py` | Oracle section pool; a section counts as correct when it contains the gold text. |
| `evaluate_oracle_section_ancestor.py` | Oracle section pool with ancestor relaxation. |
| `evaluate_oracle_paragraph_container.py` | Exact paragraph retrieval compared with descendant-based relaxation. |

## 5. Evidence-type inference

Predicts which kind of evidence (figure, table, section, line, …) a reviewer comment needs.

| Script | What it does |
|---|---|
| `prepare_retrieval_content_type_data.py` | Extracts the queries that have rankings, with their gold evidence types. |
| `split_content_type_dataset.py` | Deterministic train/validation/test split. |
| `run_content_type_inference.py` | LLM prediction via an OpenAI-compatible endpoint. |
| `train_mpnet_content_type_classifier.py` | Fine-tuned MPNet baseline. |
| `train_embedding_content_type_classifier.py` | Frozen-embedding baseline (e.g. EmbeddingGemma). |
| `evaluate_content_type_predictions.py` | Scores predictions from any of the above. |

## 6. Statistics and analysis

| Script | Output |
|---|---|
| `analyze_evidence_statistics.py` | Dataset statistics (Table 2): papers, queries, evidence, evidence combinations. |
| `similarity_review_rebuttal_stats.py` | Review–evidence similarity histograms by evidence type. |
