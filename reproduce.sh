#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-repro}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ $# -eq 0 ]]; then
  cat <<'USAGE'
Usage:
  bash reproduce.sh --mode checkpoint --run-name run_qwen_1 --checkpoint 1869 \
    --data-repo <HF_DATASET_REPO> --model-repo <HF_MODEL_REPO>

  bash reproduce.sh --mode frozen --run-name run_qwen_1 --checkpoint 1869 \
    --data-repo <HF_DATASET_REPO> --artifact-repo <HF_ARTIFACT_REPO>

Optional:
  export HF_TOKEN=hf_xxx   # needed for private repos
USAGE
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "$ROOT_DIR/requirements.txt"

python "$ROOT_DIR/scripts/reproduce.py" \
  --project-root "$ROOT_DIR" \
  "$@"
