#!/usr/bin/env bash
set -euo pipefail

DRIVE_URL="${1:-}"
if [[ -z "${DRIVE_URL}" ]]; then
  echo "Usage: bash scripts/download_models.sh <GOOGLE_DRIVE_URL>"
  exit 1
fi

# Where the app expects weights to be
DEST_DIR="working/nnUNet_results"
mkdir -p "${DEST_DIR}"

echo "Installing downloader (gdown) if needed..."
python -m pip install --quiet --upgrade pip
python -m pip install --quiet gdown

TMP_ZIP="working/tmp/nnunet_models.zip"
mkdir -p "$(dirname "${TMP_ZIP}")"

echo "Downloading model archive from Google Drive..."
python -m gdown --fuzzy --output "${TMP_ZIP}" "${DRIVE_URL}"

echo "Unzipping into ${DEST_DIR}..."
ZIP_PATH="${TMP_ZIP}" DEST_DIR="${DEST_DIR}" python - <<'PY'
import os, zipfile, sys

zip_path = os.environ.get("ZIP_PATH")
dest_dir = os.environ.get("DEST_DIR")
if not zip_path or not dest_dir:
    print("Missing ZIP_PATH/DEST_DIR env vars", file=sys.stderr)
    sys.exit(2)

with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(dest_dir)
print("Done.")
PY

echo "Model download complete."
