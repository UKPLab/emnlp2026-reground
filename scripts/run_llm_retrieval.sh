#!/usr/bin/env bash
set -euo pipefail

VLLM_URL="http://localhost:6531/v1/models"
MODEL="gpt-oss-120b"

if ! curl -s "$VLLM_URL" | grep -q "$MODEL"; then
  echo "ERROR: vLLM server not running with model $MODEL"
  exit 1
fi

echo "[INFO] vLLM server OK"

echo "[INFO] Running LLM retrieval"
python run_llm_retrieval.py --model_name qwen3-30b-thinking

echo "[OK] LLM retrieval finished"
