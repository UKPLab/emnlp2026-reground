#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"
PYTHON_BIN="python3.10"
VLLM_URL="http://localhost:6531/v1/models"
REQUIRED_MODEL="gpt-oss-120b"

#######################################
# 0. Check Python 3.10
#######################################
echo "[0/??] Checking Python version"

if ! command -v ${PYTHON_BIN} >/dev/null 2>&1; then
  echo "ERROR: python3.10 not found. Install Python 3.10."
  exit 1
fi

PY_VERSION="$(${PYTHON_BIN} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${PY_VERSION}" != "3.10" ]]; then
  echo "ERROR: Python 3.10 required, found ${PY_VERSION}"
  exit 1
fi

#######################################
# 1. Create venv & install deps
#######################################
if [[ ! -d "${VENV_DIR}" ]]; then
  ${PYTHON_BIN} -m venv ${VENV_DIR}
fi

source ${VENV_DIR}/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

#######################################
# 2. Step 1 – Clean PDFs
#######################################
echo "[1/9] Cleaning PDFs"
python cleanpdf.py

#######################################
# 3. Ask user about GROBID
#######################################
echo
echo "=================================================="
echo "Is GROBID already running and parse_papers completed?"
echo "If not, start run grobid_script.sh first."
echo "Answer 'yes' or 'no':"
echo "=================================================="
read -r GROBID_READY

case "${GROBID_READY}" in
  yes|YES|y|Y)
    echo "[2/9] Parsing papers into SDR"
    python parse_papers.py
    ;;
  no|NO|n|N)
    echo
    echo "WARNING:"
    echo "Skipping parse_papers.py."
    echo "Run the GROBID parsing script first."
    echo
    ;;
  *)
    echo "ERROR: Invalid response. Use yes or no."
    exit 1
    ;;
esac

#######################################
# 4. Step 3 – Build threads
#######################################
echo "[3/9] Building comment threads"
python threads_builder.py

echo
echo "[INFO] Data is ready. Now time for processing."
echo

#######################################
# 5. Steps 4 & 5
#######################################
echo "[4/9] Detecting regex references"
python regex_detector.py

echo "[5/9] Removing quoted references"
python regex_no_quotes.py

#######################################
# 6. Check vLLM server
#######################################
check_vllm() {
  curl -s "${VLLM_URL}" | grep -q "${REQUIRED_MODEL}"
}

if ! check_vllm; then
  echo "ERROR: vLLM not running with model ${REQUIRED_MODEL}"
  exit 1
fi

#######################################
# 7. Step 6
#######################################
echo "[6/9] Aligning rebuttal sentences to reviews"
python review_alignment.py

#######################################
# 8. Step 7
#######################################
echo "[7/9] Removing duplicate / trivial references"
python unique_refs.py

#######################################
# 9. Re-check vLLM
#######################################
if ! check_vllm; then
  echo "ERROR: vLLM server stopped before LLM filtering"
  exit 1
fi

#######################################
# 10. Step 8
#######################################
echo "[8/9] LLM-based filtering of grounded references"
python llm_filter.py

echo "[OK] Pipeline completed successfully"
