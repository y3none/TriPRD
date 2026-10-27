#!/usr/bin/env bash
# Complete correction_v3 development protocol; run under nohup/tmux if desired.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "${SCRIPT_DIR}/.."

: "${DINO_REPO:?Set DINO_REPO to the local official DINOv3 repository}"
: "${DINO_WEIGHTS:?Set DINO_WEIGHTS to the local DINOv3 ViT-B/16 checkpoint}"
PYTHON="${PYTHON:-python}"
GPUS="${GPUS:-0,1,2,3,6,7}"
SEED="${SEED:-20261012}"
SPLIT_SEED="${SPLIT_SEED:-20261012}"
COD_TRAIN_MODE="${COD_TRAIN_MODE:-split}"
SKIP_TEST="${SKIP_TEST:-1}"
if [[ "${COD_TRAIN_MODE}" == "full" && -z "${OUTPUT_ROOT:-}" ]]; then
    echo 'Full training requires an explicit separate OUTPUT_ROOT.' >&2
    exit 2
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-results/optimization}"
case "${SKIP_TEST}" in
    1) test_flag="--skip-test" ;;
    0) test_flag="--no-skip-test" ;;
    *) echo 'SKIP_TEST must be 0 or 1.' >&2; exit 2 ;;
esac
extra=()
if [[ -n "${SPLIT_MANIFEST:-}" ]]; then
    extra+=(--split-manifest "${SPLIT_MANIFEST}")
fi
if [[ "${RESUME:-0}" == "1" ]]; then extra+=(--resume); fi
if [[ "${ALLOW_NONPOSITIVE_PROBE:-0}" == "1" ]]; then extra+=(--allow-nonpositive-probe); fi

exec "${PYTHON}" -u tools/run_triprd_correction_queue.py \
    --gpus "${GPUS}" --seed "${SEED}" --split-seed "${SPLIT_SEED}" \
    --output-root "${OUTPUT_ROOT}" --groups A,B,C,D,E,F \
    --vdt-root "${VDT_ROOT:-datasets/VDT-2048}" --cod-root "${COD_ROOT:-datasets/COD_benchmark}" \
    --model-variant dinov3_vitb16 --dino-repo "${DINO_REPO}" --dino-weights "${DINO_WEIGHTS}" --pretrained \
    --epochs 100 --teacher-epochs 100 --image-size 384 --batch-size 8 --base-channels 128 \
    --num-workers "${NUM_WORKERS:-4}" --device cuda --amp --max-samples 0 \
    --val-metrics-every 5 --cod-train-mode "${COD_TRAIN_MODE}" "${test_flag}" \
    --lr 1e-4 --weight-decay 1e-4 --source-weight 0.5 --kd-weight 1 --relation-weight 0.2 --warmup-epochs 5 \
    "${extra[@]}" --execute
