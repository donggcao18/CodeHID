#!/bin/bash
set -euo pipefail

# Available: GPU4-7. Default 3-GPU run uses GPU4,5,6.
# Override by exporting CUDA_VISIBLE_DEVICES before running.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export OMP_NUM_THREADS=1
export PYTHONWARNINGS="ignore"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING=1

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
CFG="${CFG:-${ROOT_DIR}/train_configs/codehid_finetune.yaml}"
OUTPUT_BASE="${OUTPUT_BASE:-${ROOT_DIR}/output/codehid_codebert_dsi}"
MASTER_PORT="${MASTER_PORT:-29531}"
PORT_SCAN_LIMIT="${PORT_SCAN_LIMIT:-100}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/output/torchrun_logs}"
DATASET_FINAL_DIR="${DATASET_FINAL_DIR:-${ROOT_DIR}/stage1_docid_build/data/cosqa_docid_build/final}"
TRAIN_JSONL_NAME="${TRAIN_JSONL_NAME:-train.jsonl}"
VALID_JSONL_NAME="${VALID_JSONL_NAME:-valid.jsonl}"
# Use dedicated overrides to avoid being polluted by unrelated shell vars (e.g., TRAIN_JSONL from other projects).
CGR_TRAIN_JSONL="${CGR_TRAIN_JSONL:-}"
CGR_VALID_JSONL="${CGR_VALID_JSONL:-}"
BATCH_SIZE_TRAIN="${BATCH_SIZE_TRAIN:-4}"
BATCH_SIZE_EVAL="${BATCH_SIZE_EVAL:-2}"
GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-2}"
NUM_WORKERS="${NUM_WORKERS:-12}"
ITERS_PER_EPOCH="${ITERS_PER_EPOCH:-2428}"
MAX_EPOCH="${MAX_EPOCH:-12}"
ENABLE_GRADIENT_CHECKPOINTING="${ENABLE_GRADIENT_CHECKPOINTING:-false}"
PARALLEL_STRATEGY="${PARALLEL_STRATEGY:-ddp}"
DUAL_PHASE_GUIDANCE_TRAIN_ENABLE="${DUAL_PHASE_GUIDANCE_TRAIN_ENABLE:-true}"
DUAL_PHASE_GUIDANCE_TRAIN_LAMBDA_LEX="${DUAL_PHASE_GUIDANCE_TRAIN_LAMBDA_LEX:-0.06}"
DUAL_PHASE_GUIDANCE_TRAIN_LAMBDA_RANK_DISTILL="${DUAL_PHASE_GUIDANCE_TRAIN_LAMBDA_RANK_DISTILL:-0.03}"
DUAL_PHASE_GUIDANCE_TRAIN_HARD_NEG_K="${DUAL_PHASE_GUIDANCE_TRAIN_HARD_NEG_K:-4}"
DUAL_PHASE_GUIDANCE_TRAIN_MINE_TOPK="${DUAL_PHASE_GUIDANCE_TRAIN_MINE_TOPK:-1000}"
DUAL_PHASE_GUIDANCE_TRAIN_MINE_EVERY_N_EPOCH="${DUAL_PHASE_GUIDANCE_TRAIN_MINE_EVERY_N_EPOCH:-1}"
DUAL_PHASE_GUIDANCE_TRAIN_MINE_BATCH_SIZE="${DUAL_PHASE_GUIDANCE_TRAIN_MINE_BATCH_SIZE:-64}"
DUAL_PHASE_GUIDANCE_TRAIN_START_EPOCH="${DUAL_PHASE_GUIDANCE_TRAIN_START_EPOCH:-5}"

resolve_first_existing_path() {
  local path
  for path in "$@"; do
    if [[ -n "${path}" && -f "${path}" ]]; then
      echo "${path}"
      return 0
    fi
  done
  return 1
}

is_port_in_use() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -ltnH "( sport = :${port} )" 2>/dev/null | grep -q .
    return $?
  fi
  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1
    return $?
  fi
  if command -v netstat >/dev/null 2>&1; then
    netstat -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]${port}$"
    return $?
  fi
  return 1
}

find_free_port() {
  local start_port="$1"
  local max_tries="$2"
  local candidate
  local i
  for ((i = 0; i < max_tries; i++)); do
    candidate=$((start_port + i))
    if ((candidate > 65535)); then
      break
    fi
    if ! is_port_in_use "${candidate}"; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

if [[ -z "${CGR_TRAIN_JSONL}" ]]; then
  CGR_TRAIN_JSONL="$(resolve_first_existing_path \
    "${DATASET_FINAL_DIR}/${TRAIN_JSONL_NAME}" \
    "${DATASET_FINAL_DIR}/train.jsonl" \
    "${DATASET_FINAL_DIR}/cosqa_train.jsonl")" || {
      echo "[ERROR] Unable to locate training jsonl under ${DATASET_FINAL_DIR}."
      echo "[ERROR] Tried: ${TRAIN_JSONL_NAME}, train.jsonl, cosqa_train.jsonl"
      exit 1
    }
fi

if [[ -z "${CGR_VALID_JSONL}" ]]; then
  CGR_VALID_JSONL="$(resolve_first_existing_path \
    "${DATASET_FINAL_DIR}/${VALID_JSONL_NAME}" \
    "${DATASET_FINAL_DIR}/valid.jsonl" \
    "${DATASET_FINAL_DIR}/cosqa_valid.jsonl")" || {
      echo "[ERROR] Unable to locate validation jsonl under ${DATASET_FINAL_DIR}."
      echo "[ERROR] Tried: ${VALID_JSONL_NAME}, valid.jsonl, cosqa_valid.jsonl"
      exit 1
    }
fi

SELECTED_PORT="$(find_free_port "${MASTER_PORT}" "${PORT_SCAN_LIMIT}")" || {
  echo "[ERROR] No free port found from ${MASTER_PORT} (scan limit: ${PORT_SCAN_LIMIT})."
  exit 1
}

mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_BASE}"
echo "[DATA] DATASET_FINAL_DIR=${DATASET_FINAL_DIR}"
echo "[DATA] TRAIN_JSONL=${CGR_TRAIN_JSONL}"
echo "[DATA] VALID_JSONL=${CGR_VALID_JSONL}"
echo "[RUN] OUTPUT_BASE=${OUTPUT_BASE}"
echo "[RUN] MASTER_PORT requested=${MASTER_PORT}, selected=${SELECTED_PORT}"
if [[ -n "${MAX_EPOCH}" ]]; then
  echo "[RUN] MAX_EPOCH override=${MAX_EPOCH}"
fi

torchrun \
  --nproc_per_node=3 \
  --master_port="${SELECTED_PORT}" \
  train.py \
  --cfg-path "${CFG}" \
  --options \
    run.distributed=true \
    run.world_size=3 \
    run.dist_url=env:// \
    run.parallel_strategy="${PARALLEL_STRATEGY}" \
    datasets.cosqa_dsi.build_info.ann_path="${CGR_TRAIN_JSONL}" \
    datasets.cosqa_dsi.build_info.valid_ann_path="${CGR_VALID_JSONL}" \
    run.batch_size_train="${BATCH_SIZE_TRAIN}" \
    run.batch_size_eval="${BATCH_SIZE_EVAL}" \
    run.gradient_accumulation_steps="${GRAD_ACC_STEPS}" \
    run.num_workers="${NUM_WORKERS}" \
    run.iters_per_epoch="${ITERS_PER_EPOCH}" \
    run.max_epoch="${MAX_EPOCH:-12}" \
    run.output_dir="${OUTPUT_BASE}" \
    run.dual_phase_guidance_train_enable="${DUAL_PHASE_GUIDANCE_TRAIN_ENABLE}" \
    run.dual_phase_guidance_train_lambda_lex="${DUAL_PHASE_GUIDANCE_TRAIN_LAMBDA_LEX}" \
    run.dual_phase_guidance_train_lambda_rank_distill="${DUAL_PHASE_GUIDANCE_TRAIN_LAMBDA_RANK_DISTILL}" \
    run.dual_phase_guidance_train_hard_neg_k="${DUAL_PHASE_GUIDANCE_TRAIN_HARD_NEG_K}" \
    run.dual_phase_guidance_train_mine_topk="${DUAL_PHASE_GUIDANCE_TRAIN_MINE_TOPK}" \
    run.dual_phase_guidance_train_mine_every_n_epoch="${DUAL_PHASE_GUIDANCE_TRAIN_MINE_EVERY_N_EPOCH}" \
    run.dual_phase_guidance_train_mine_batch_size="${DUAL_PHASE_GUIDANCE_TRAIN_MINE_BATCH_SIZE}" \
    run.dual_phase_guidance_train_start_epoch="${DUAL_PHASE_GUIDANCE_TRAIN_START_EPOCH}" \
    model.dual_phase_guidance_train_enable="${DUAL_PHASE_GUIDANCE_TRAIN_ENABLE}" \
    model.dual_phase_guidance_train_lambda_lex="${DUAL_PHASE_GUIDANCE_TRAIN_LAMBDA_LEX}" \
    model.dual_phase_guidance_train_lambda_rank_distill="${DUAL_PHASE_GUIDANCE_TRAIN_LAMBDA_RANK_DISTILL}" \
    model.enable_gradient_checkpointing="${ENABLE_GRADIENT_CHECKPOINTING}" \
  2>&1 | tee "${LOG_DIR}/run_3gpu_$(date +%Y%m%d_%H%M%S).log"
