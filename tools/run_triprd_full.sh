#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
VDT_ROOT="${VDT_ROOT:-${REPO_ROOT}/datasets/VDT-2048}"
COD_ROOT="${COD_ROOT:-${REPO_ROOT}/datasets/COD_benchmark}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/triprd_full}"
TRIPRD_GROUPS="${TRIPRD_GROUPS:-A,B,C,D}"
VDT_EPOCHS="${VDT_EPOCHS:-100}"
COD_EPOCHS="${COD_EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-8}"
IMAGE_SIZE="${IMAGE_SIZE:-352}"
BASE_CHANNELS="${BASE_CHANNELS:-256}"
SEED="${SEED:-20261012}"
DEVICE="${DEVICE:-cuda}"
TEST_DATASETS="${TEST_DATASETS:-NC4K,COD10K,CAMO,CHAMELEON}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}"
LOG_PATH="${OUTPUT_ROOT}/run.log"
echo "TriPRD full run started: $(date -Iseconds)" | tee -a "${LOG_PATH}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" | tee -a "${LOG_PATH}"
set +e
"${PYTHON_BIN}" -u tools/run_triprd_ab.py \
  --vdt-root "${VDT_ROOT}" \
  --cod-root "${COD_ROOT}" \
  --out "${OUTPUT_ROOT}" \
  --groups "${TRIPRD_GROUPS}" \
  --model-variant resnet50 \
  --pretrained \
  --vdt-epochs "${VDT_EPOCHS}" \
  --epochs "${COD_EPOCHS}" \
  --max-samples 0 \
  --max-test-samples 0 \
  --batch-size "${BATCH_SIZE}" \
  --image-size "${IMAGE_SIZE}" \
  --base-channels "${BASE_CHANNELS}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --test-datasets "${TEST_DATASETS}" 2>&1 | tee -a "${LOG_PATH}"
pipeline_status=("${PIPESTATUS[@]}")
set -e
status=${pipeline_status[0]}
if [[ ${status} -eq 0 ]]; then status=${pipeline_status[1]}; fi
echo "TriPRD full run finished with status ${status}: $(date -Iseconds)" | tee -a "${LOG_PATH}"
exit "${status}"
