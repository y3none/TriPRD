#!/usr/bin/env bash
# Two fresh C/D jobs; reuse completed A/B, teacher and joint-MSE probe evidence.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "${SCRIPT_DIR}/.."
extra=()
case "${DRY_RUN:-0}" in
  0) extra+=(--execute) ;;
  1) ;;
  *) echo 'DRY_RUN must be 0 or 1.' >&2; exit 2 ;;
esac
case "${RESUME:-0}" in
  0) ;;
  1) extra+=(--resume) ;;
  *) echo 'RESUME must be 0 or 1.' >&2; exit 2 ;;
esac
active_pid=""
stop_run() {
  local code="$1"
  trap '' INT TERM
  if [[ -n "${active_pid}" ]]; then
    kill -TERM "${active_pid}" 2>/dev/null || true
    wait "${active_pid}" || true
  fi
  exit "${code}"
}
trap 'stop_run 130' INT
trap 'stop_run 143' TERM
"${PYTHON:-python}" -u tools/run_triprd_mse_screen.py \
  --run-root "${RUN_ROOT:-results/optimization/correction_4gpu_ampfix/seed-20261012}" \
  --diagnostics-root "${DIAGNOSTICS_ROOT:-results/optimization/correction_probe_diagnostic_v1}" \
  --output-root "${OUTPUT_ROOT:-results/optimization/correction_mse_screen_v1}" \
  --gpus "${GPUS:-${CUDA_VISIBLE_DEVICES:-${GPU:-0}}}" "${extra[@]}" "$@" &
active_pid=$!
result=0
wait "${active_pid}" || result=$?
active_pid=""
exit "${result}"
