#!/usr/bin/env bash
# Evaluate completed best checkpoints sequentially on one visible GPU.
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
export CUDA_VISIBLE_DEVICES="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
exec "${PYTHON:-python}" -u tools/diagnose_triprd_correction_path.py \
  --run-root "${RUN_ROOT:-results/optimization/correction_4gpu_ampfix/seed-20261012}" \
  --diagnostics-root "${DIAGNOSTICS_ROOT:-results/optimization/correction_probe_diagnostic_v1}" \
  --screen-root "${SCREEN_ROOT:-results/optimization/correction_mse_screen_v1}" \
  --output-root "${OUTPUT_ROOT:-results/optimization/correction_path_diagnostic_v1}" \
  --groups "${GROUPS_TO_CHECK:-A,B,C,D}" --device "${DEVICE:-cuda}" \
  --num-workers "${NUM_WORKERS:-0}" "${extra[@]}" "$@"
