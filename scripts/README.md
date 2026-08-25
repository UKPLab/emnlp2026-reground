# Rebuttal–Review Alignment Pipeline

This repository contains a multi-stage pipeline for processing academic papers, reviews, and rebuttals, including PDF cleaning, structure extraction, reference detection, alignment, and LLM-based filtering.

The pipeline is intentionally split into preparation (PDF + GROBID) and processing (regex + LLM) to keep failures localized and debuggable.

---

## Requirements

### System
- Python 3.10 (required)
- Docker (for GROBID)
- curl
- Linux or macOS

### Python
- Dependencies are listed in requirements.txt
- A local virtual environment is created automatically

### LLM Inference
- A running vLLM server at http://localhost:6531
- OpenAI-compatible API
- Model loaded: gpt-oss-120b

---

## Repository Structure

```text
.
├── run.sh
├── run_grobid.sh
├── run_grob.py
├── cleanpdf.py
├── parse_papers.py
├── threads_builder.py
├── regex_detector.py
├── regex_no_quotes.py
├── review_alignment.py
├── unique_refs.py
├── llm_filter.py
├── run_llm_retrieval.py
├── run_llm_retrieval.sh
├── llm_retrieval_logprobs.py
├── compute_metrics.py
├── run_normal_retrieval.sh
├── retrieval_methods.py
├── requirements.txt
```

---

## Workflow

### Phase 1: Preparation
1. Clean PDFs
2. Run GROBID
3. Parse papers
4. Build comment threads

### Phase 2: Processing
5. Regex reference detection
6. Remove quoted references
7. Align rebuttals to reviews
8. Deduplicate references
9. LLM-based filtering

---

## Usage

### Extract data.zip into a directory called data

#### IMPORTANT NOTE

Due to size restrictions, we are only uploading a representative subset of the whole dataset. Unfortunately, the whole dataset is 22GB, so we cannot upload it, and the ACL policies prohibit links to external cloud storages. We fully commit to releasing the whole dataset upon acceptance.

### 1. Start GROBID

In a separate terminal:

docker run --rm --init \
  -p 8070:8070 -p 8071:8071 \
  lfoppiano/grobid:latest-crf

---

### 2. Run GROBID Parsing

bash run_grobid.sh

---

### 3. Run Main Pipeline

bash run.sh

The pipeline will:
- Verify Python 3.10
- Create a virtual environment
- Install dependencies
- Ask whether GROBID parsing has already been completed
- Run processing and LLM steps with strict checks

### 4. Run run_normal_retrieval.sh

### 5. Run run_llm_retrieval.sh

---

## vLLM Requirement

Example launch:

vllm serve gpt-oss-120b --port 6531 --api-key dummy

The pipeline checks:

GET http://localhost:6531/v1/models

If the model is missing, execution stops.