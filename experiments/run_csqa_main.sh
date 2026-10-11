#!/usr/bin/env bash
# Run the released one-seed CSQA protocol on a single configured TPU allocation.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash experiments/run_csqa_main.sh SOURCE_JSONL MODEL_ROOT OUTPUT_ROOT [TP_SIZE]

Runs the released Qwen3-0.6B CSQA comparison:
  seed=0, anchors={8,16}, N=512, methods={SFT,Replay,CPL,CPLUS}.

SOURCE_JSONL  Prepared four-option CSQA protocol file.
MODEL_ROOT    Root passed to scripts/download_models.py (typically weights).
OUTPUT_ROOT   Fresh directory for commands and experiment artifacts.
TP_SIZE       Tensor-parallel degree; default: 4.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if [[ $# -lt 3 || $# -gt 4 ]]; then
  usage >&2
  exit 2
fi

SOURCE_JSONL="$1"
MODEL_ROOT="$2"
OUTPUT_ROOT="$3"
TP_SIZE="${4:-4}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMMANDS_PATH="${OUTPUT_ROOT}/csqa-main-commands.sh"

if [[ ! -f "${SOURCE_JSONL}" ]]; then
  echo "missing SOURCE_JSONL: ${SOURCE_JSONL}" >&2
  exit 1
fi
if [[ ! -d "${MODEL_ROOT}" ]]; then
  echo "missing MODEL_ROOT: ${MODEL_ROOT}" >&2
  exit 1
fi
if [[ ! "${TP_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TP_SIZE must be a positive integer: ${TP_SIZE}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}"
python "${ROOT}/scripts/validate_csqa_input.py" "${SOURCE_JSONL}"
python "${ROOT}/scripts/print_csqa_main_grid.py" \
  --source-jsonl "${SOURCE_JSONL}" \
  --model-root "${MODEL_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --tp-size "${TP_SIZE}" \
  > "${COMMANDS_PATH}"

echo "Executing released CSQA plan: ${COMMANDS_PATH}"
bash "${COMMANDS_PATH}"
