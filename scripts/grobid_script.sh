#!/usr/bin/env bash
set -euo pipefail

GROBID_URL="http://localhost:8070/api/isalive"

#######################################
# 0. Check Docker
#######################################
if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: Docker not found. Install Docker before running GROBID."
  exit 1
fi

#######################################
# 1. Prompt user to run GROBID
#######################################
echo
echo "=================================================="
echo "You need to start GROBID in another terminal."
echo
echo "Run the following command:"
echo
echo "docker run --rm --init \\"
echo "  -p 8070:8070 -p 8071:8071 \\"
echo "  lfoppiano/grobid:latest-crf"
echo
echo "Once GROBID is running, type 'yes' and press ENTER."
echo "=================================================="
read -r USER_CONFIRM

case "${USER_CONFIRM}" in
  yes|YES|y|Y)
    ;;
  *)
    echo "Aborted. Start GROBID first, then re-run this script."
    exit 1
    ;;
esac

#######################################
# 2. Check GROBID health
#######################################
echo "[INFO] Checking GROBID availability at localhost:8070"

if ! curl -sf "${GROBID_URL}" >/dev/null; then
  echo "ERROR: GROBID is not responding at localhost:8070"
  echo "Make sure the Docker container is running."
  exit 1
fi

echo "✓ GROBID is up"

#######################################
# 3. Run parsing script
#######################################
echo "[INFO] Running run_grob.py"
python run_grob.py

echo "[OK] GROBID parsing completed successfully"
