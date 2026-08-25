#!/usr/bin/env bash
set -euo pipefail

echo "[INFO] Running retrieval metrics"

python compute_metrics.py

echo "[OK] Retrieval metrics completed"
