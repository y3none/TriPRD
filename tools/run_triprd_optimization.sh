#!/usr/bin/env bash
# Run correction-v3 phases; independent experiments occupy separate GPUs.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "${SCRIPT_DIR}/.."

phase="${1:-}"
case "${phase}" in
    baseline|probe|screen|final|all) shift ;;
    *) echo 'Usage: bash tools/run_triprd_optimization.sh baseline|probe|screen|final|all [Python CLI options]' >&2; exit 2 ;;
esac
: "${DINO_REPO:?Set DINO_REPO to the local official DINOv3 repository}"
: "${DINO_WEIGHTS:?Set DINO_WEIGHTS to the local generic DINOv3 ViT-B/16 checkpoint}"
# GPUS selects physical device identifiers. An inherited visibility restriction
# remains in force; the Python launcher validates the selected subset.
gpus="${GPUS:-${CUDA_VISIBLE_DEVICES:-${GPU:-0}}}"

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
if [[ -n "${SPLIT_MANIFEST:-}" ]]; then extra+=(--split-manifest "${SPLIT_MANIFEST}"); fi

active_pid=""
stop_run() {
    local code="$1"
    # Keep the shell alive until the Python launcher has reaped its workers.
    trap '' INT TERM
    if [[ -n "${active_pid}" ]]; then
        kill -TERM "${active_pid}" 2>/dev/null || true
        wait "${active_pid}" || true
    fi
    exit "${code}"
}
trap 'stop_run 130' INT
trap 'stop_run 143' TERM

run_phase() {
    local selected_phase="$1"
    shift
    "${PYTHON:-python}" -u tools/run_triprd_optimization_phases.py \
    --phase "${selected_phase}" --output-root "${OUTPUT_ROOT:-results/optimization/correction_phases}" \
    --seed "${SEED:-20261012}" --split-seed "${SPLIT_SEED:-20261012}" \
    --vdt-root "${VDT_ROOT:-datasets/VDT-2048}" --cod-root "${COD_ROOT:-datasets/COD_benchmark}" \
    --model-variant dinov3_vitb16 --dino-repo "${DINO_REPO}" --dino-weights "${DINO_WEIGHTS}" --pretrained \
    --device cuda --gpus "${gpus}" --amp --image-size "${IMAGE_SIZE:-384}" --base-channels "${BASE_CHANNELS:-128}" \
    --batch-size "${BATCH_SIZE:-8}" --num-workers "${NUM_WORKERS:-4}" \
    --epochs "${EPOCHS:-100}" --teacher-epochs "${TEACHER_EPOCHS:-100}" --probe-epochs "${PROBE_EPOCHS:-10}" \
    --warmup-epochs "${WARMUP_EPOCHS:-5}" --lr "${LR:-0.0001}" --weight-decay 0.0001 \
    --source-weight 0.5 --kd-weight 1 --relation-weight 0.2 --val-metrics-every 5 \
    "${extra[@]}" "$@" &
    active_pid=$!
    local result=0
    wait "${active_pid}" || result=$?
    active_pid=""
    return "${result}"
}

if [[ "${phase}" == all ]]; then
    blocked=0
    for selected_phase in baseline probe screen final; do
        echo "Phase ${selected_phase}: GPUs ${gpus}"
        result=0
        run_phase "${selected_phase}" "$@" || result=$?
        # A dry run can preview later phases before their evidence exists.
        # Actual execution always stops on any failed or blocked phase.
        if [[ "${DRY_RUN:-0}" == 1 && "${result}" == 2 ]]; then
            blocked=2
        elif [[ "${result}" != 0 ]]; then
            exit "${result}"
        fi
    done
    exit "${blocked}"
else
    run_phase "${phase}" "$@"
fi
