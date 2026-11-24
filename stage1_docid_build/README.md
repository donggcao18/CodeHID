# Stage I: DocID Construction

This directory implements Stage I of CodeHID: Pseudo-Neighbor Guided DocID Learning.

The goal of this stage is to build a globally static hierarchical DocID space for code retrieval. The pipeline first encodes code snippets into dense vectors, then applies multi-level residual quantization, and finally regularizes prefix sharing with pseudo-neighbor supervision derived from a k-nearest-neighbor graph.

## Components

- `prepare_cosqa_docid_data.py`: prepares CoSQA train/dev positives and builds a unified document pool.
- `encode_code_docs.py`: encodes code documents into dense embeddings.
- `build_pseudo_neighbor_graph.py`: constructs `knn_indices.npy` and `knn_sims.npy` from document embeddings.
- `rqvae/train_rqvae.py`: trains the residual quantizer with optional pseudo-neighbor prefix loss.
- `rqvae/generate_docid_codes.py`: generates discrete multi-level codes from a trained checkpoint.
- `rewrite_with_docids.py`: rewrites the prepared train/dev splits with hierarchical DocIDs.
- `run_stage1_pipeline.sh`: runs the full Stage I pipeline end to end.

## Default KNN Settings

- `USE_KNN_PREFIX_LOSS=1`
- `KNN_K=8`
- `KNN_BATCH_SIZE=2048`
- `KNN_LOSS_WEIGHT=0.03`
- `KNN_LOSS_SCHEDULE=linear_decay_to_zero`
- `KNN_DECAY_START_RATIO=0.3`
- `KNN_DECAY_END_RATIO=0.7`
- `KNN_TEMPERATURE=0.35`
- `KNN_PREFIX_WEIGHTS="0.8 0.35 0.12 0.03"`
- `KNN_AUTO_THRESHOLDS=1`
- `KNN_THRESHOLD_QUANTILES="0.85 0.95 0.99 0.999"`
- `KNN_NUM_NEGATIVES=4`
- `KNN_NEGATIVE_WEIGHT=0.6`

## Quick Start

```bash
WORK_DIR=<your_work_dir> \
USE_KNN_PREFIX_LOSS=1 \
KNN_K=8 \
KNN_BATCH_SIZE=2048 \
KNN_LOSS_WEIGHT=0.03 \
KNN_LOSS_SCHEDULE=linear_decay_to_zero \
KNN_DECAY_START_RATIO=0.3 \
KNN_DECAY_END_RATIO=0.7 \
KNN_TEMPERATURE=0.35 \
KNN_PREFIX_WEIGHTS="0.8 0.35 0.12 0.03" \
KNN_AUTO_THRESHOLDS=1 \
KNN_THRESHOLD_QUANTILES="0.85 0.95 0.99 0.999" \
KNN_NUM_NEGATIVES=4 \
KNN_NEGATIVE_WEIGHT=0.6 \
bash stage1_docid_build/run_stage1_pipeline.sh cuda:0 "$WORK_DIR" <encoder_path> <train_json> <dev_json>
```

## Outputs

The pipeline writes intermediate and final files under `WORK_DIR`, including:

- `docs.jsonl`
- `docs.emb-codebert.npy`
- `knn_indices.npy`
- `knn_sims.npy`
- `rqvae_codes.npy`
- `final/train.jsonl`
- `final/valid.jsonl`
- `final/doc_index_to_docid.json`

See [DESIGN_pseudo_neighbor_prefix_loss.md](./DESIGN_pseudo_neighbor_prefix_loss.md) for the design notes on pseudo-neighbor prefix supervision.
