#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash stage1_docid_build/run_stage1_pipeline.sh [device] [work_dir] [encoder_model] [train_json] [dev_json]
#
# Example:
#   WORK_DIR=<your_work_dir> \
#   USE_KNN_PREFIX_LOSS=1 \
#   KNN_K=8 \
#   KNN_BATCH_SIZE=2048 \
#   KNN_LOSS_WEIGHT=0.03 \
#   KNN_LOSS_SCHEDULE=linear_decay_to_zero \
#   KNN_DECAY_START_RATIO=0.3 \
#   KNN_DECAY_END_RATIO=0.7 \
#   KNN_TEMPERATURE=0.35 \
#   KNN_PREFIX_WEIGHTS="0.8 0.35 0.12 0.03" \
#   KNN_AUTO_THRESHOLDS=1 \
#   KNN_THRESHOLD_QUANTILES="0.85 0.95 0.99 0.999" \
#   KNN_NUM_NEGATIVES=4 \
#   KNN_NEGATIVE_WEIGHT=0.6 \
#   bash stage1_docid_build/run_stage1_pipeline.sh cuda:0 "$WORK_DIR" <encoder_path> <train_json> <dev_json>

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEVICE="${1:-cuda:0}"
WORK_DIR="${2:-${WORK_DIR:-${ROOT_DIR}/data/cosqa_docid_build}}"
ENCODER_MODEL="${3:-${ENCODER_MODEL:-}}"
TRAIN_JSON="${4:-${TRAIN_JSON:-}}"
DEV_JSON="${5:-${DEV_JSON:-}}"

if [ -z "${ENCODER_MODEL}" ] || [ -z "${TRAIN_JSON}" ] || [ -z "${DEV_JSON}" ]; then
  echo "Usage: bash stage1_docid_build/run_stage1_pipeline.sh [device] [work_dir] [encoder_model] [train_json] [dev_json]" >&2
  echo "Missing required input(s): encoder_model/train_json/dev_json." >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

USE_KNN_PREFIX_LOSS="${USE_KNN_PREFIX_LOSS:-1}"
KNN_K="${KNN_K:-8}"
KNN_BATCH_SIZE="${KNN_BATCH_SIZE:-2048}"
KNN_LOSS_WEIGHT="${KNN_LOSS_WEIGHT:-0.03}"
KNN_LOSS_SCHEDULE="${KNN_LOSS_SCHEDULE:-linear_decay_to_zero}"
KNN_DECAY_START_RATIO="${KNN_DECAY_START_RATIO:-0.3}"
KNN_DECAY_END_RATIO="${KNN_DECAY_END_RATIO:-0.7}"
KNN_TEMPERATURE="${KNN_TEMPERATURE:-0.35}"
KNN_PREFIX_WEIGHTS="${KNN_PREFIX_WEIGHTS:-0.8 0.35 0.12 0.03}"
KNN_AUTO_THRESHOLDS="${KNN_AUTO_THRESHOLDS:-1}"
KNN_THRESHOLD_QUANTILES="${KNN_THRESHOLD_QUANTILES:-0.85 0.95 0.99 0.999}"
KNN_THRESHOLDS="${KNN_THRESHOLDS:-0.55 0.65 0.75 0.82}"
KNN_NUM_NEGATIVES="${KNN_NUM_NEGATIVES:-4}"
KNN_NEGATIVE_WEIGHT="${KNN_NEGATIVE_WEIGHT:-0.6}"

CKPT_ROOT="${WORK_DIR}/ckpt"
mkdir -p "${WORK_DIR}" "${CKPT_ROOT}"

echo "[RUN] DEVICE=${DEVICE}"
echo "[RUN] WORK_DIR=${WORK_DIR}"
echo "[RUN] ENCODER_MODEL=${ENCODER_MODEL}"
echo "[RUN] TRAIN_JSON=${TRAIN_JSON}"
echo "[RUN] DEV_JSON=${DEV_JSON}"

echo "[1/6] Prepare CoSQA positives ..."
"${PYTHON_BIN}" "${ROOT_DIR}/prepare_cosqa_docid_data.py" \
  --train_json "${TRAIN_JSON}" \
  --dev_json "${DEV_JSON}" \
  --output_dir "${WORK_DIR}" \
  --dedup_key code

echo "[2/6] Encode docs ..."
"${PYTHON_BIN}" "${ROOT_DIR}/encode_code_docs.py" \
  --docs_jsonl "${WORK_DIR}/docs.jsonl" \
  --output_npy "${WORK_DIR}/docs.emb-codebert.npy" \
  --model_name "${ENCODER_MODEL}" \
  --text_mode code \
  --batch_size 32 \
  --max_length 512 \
  --device "${DEVICE}"

if [ "${USE_KNN_PREFIX_LOSS}" = "1" ]; then
  echo "[3/6] Build KNN graph ..."
  "${PYTHON_BIN}" "${ROOT_DIR}/build_pseudo_neighbor_graph.py" \
    --data_path "${WORK_DIR}/docs.emb-codebert.npy" \
    --output_indices "${WORK_DIR}/knn_indices.npy" \
    --output_sims "${WORK_DIR}/knn_sims.npy" \
    --k "${KNN_K}" \
    --batch_size "${KNN_BATCH_SIZE}" \
    --device "${DEVICE}"
fi

echo "[4/6] Train 4-level RQ-VAE ..."
if [ "${USE_KNN_PREFIX_LOSS}" = "1" ]; then
  KNN_THRESHOLD_ARGS=()
  if [ "${KNN_AUTO_THRESHOLDS}" = "1" ]; then
    KNN_THRESHOLD_ARGS+=(--knn_auto_thresholds --knn_threshold_quantiles ${KNN_THRESHOLD_QUANTILES})
  else
    KNN_THRESHOLD_ARGS+=(--no-knn_auto_thresholds --knn_thresholds ${KNN_THRESHOLDS})
  fi
  "${PYTHON_BIN}" "${ROOT_DIR}/rqvae/train_rqvae.py" \
    --data_path "${WORK_DIR}/docs.emb-codebert.npy" \
    --ckpt_dir "${CKPT_ROOT}" \
    --device "${DEVICE}" \
    --epochs 300 \
    --eval_step 10 \
    --batch_size "${KNN_BATCH_SIZE}" \
    --num_workers 4 \
    --num_emb_list 256 256 256 256 \
    --sk_epsilons 0.0 0.0 0.0 0.003 \
    --e_dim 32 \
    --beta 0.25 \
    --use_knn_prefix_loss \
    --knn_indices_path "${WORK_DIR}/knn_indices.npy" \
    --knn_sims_path "${WORK_DIR}/knn_sims.npy" \
    --knn_k "${KNN_K}" \
    --knn_loss_weight "${KNN_LOSS_WEIGHT}" \
    --knn_loss_schedule "${KNN_LOSS_SCHEDULE}" \
    --knn_decay_start_ratio "${KNN_DECAY_START_RATIO}" \
    --knn_decay_end_ratio "${KNN_DECAY_END_RATIO}" \
    --knn_temperature "${KNN_TEMPERATURE}" \
    --knn_num_negatives "${KNN_NUM_NEGATIVES}" \
    --knn_negative_weight "${KNN_NEGATIVE_WEIGHT}" \
    --knn_prefix_weights ${KNN_PREFIX_WEIGHTS} \
    "${KNN_THRESHOLD_ARGS[@]}"
else
  "${PYTHON_BIN}" "${ROOT_DIR}/rqvae/train_rqvae.py" \
    --data_path "${WORK_DIR}/docs.emb-codebert.npy" \
    --ckpt_dir "${CKPT_ROOT}" \
    --device "${DEVICE}" \
    --epochs 300 \
    --eval_step 10 \
    --batch_size "${KNN_BATCH_SIZE}" \
    --num_workers 4 \
    --num_emb_list 256 256 256 256 \
    --sk_epsilons 0.0 0.0 0.0 0.003 \
    --e_dim 32 \
    --beta 0.25 \
    --no-use_knn_prefix_loss
fi

LATEST_RUN="$(ls -1dt "${CKPT_ROOT}"/* 2>/dev/null | head -n 1 || true)"
BEST_CKPT="${LATEST_RUN}/best_collision_model.pth"
if [ -z "${LATEST_RUN}" ] || [ ! -f "${BEST_CKPT}" ]; then
  echo "[ERROR] best_collision_model.pth not found under ${CKPT_ROOT}" >&2
  exit 1
fi

echo "[5/6] Generate RQ-VAE codes ..."
"${PYTHON_BIN}" "${ROOT_DIR}/rqvae/generate_docid_codes.py" \
  --checkpoint "${BEST_CKPT}" \
  --output_codes "${WORK_DIR}/rqvae_codes.npy" \
  --device "${DEVICE}" \
  --batch_size 64 \
  --num_workers 4 \
  --max_collision_passes 30

echo "[6/6] Rewrite to hierarchical docids ..."
"${PYTHON_BIN}" "${ROOT_DIR}/rewrite_with_docids.py" \
  --docs_jsonl "${WORK_DIR}/docs.jsonl" \
  --train_input "${WORK_DIR}/train_pos_with_doc.jsonl" \
  --dev_input "${WORK_DIR}/dev_pos_with_doc.jsonl" \
  --codes_path "${WORK_DIR}/rqvae_codes.npy" \
  --output_dir "${WORK_DIR}/final" \
  --level_tags "abcd"

echo "[DONE] Final outputs:"
ls -la "${WORK_DIR}/final"
