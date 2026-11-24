# RQ-VAE Module

This subdirectory contains the residual quantization model used in Stage I DocID construction.

The base training objective remains:

- `L_rqvae = L_recon + lambda_q * L_quant`

When pseudo-neighbor supervision is enabled, training uses:

- `L_total = L_rqvae + lambda_knn * L_knn_prefix`

## Main Files

- `train_rqvae.py`: training entrypoint.
- `rqvae_trainer.py`: training loop and KNN prefix loss integration.
- `embedding_dataset.py`: embedding dataset loader.
- `generate_docid_codes.py`: discrete code generation from a trained checkpoint.
- `models/`: residual quantization model implementation.

## Example

```bash
python stage1_docid_build/rqvae/train_rqvae.py \
  --data_path <work_dir>/docs.emb-codebert.npy \
  --ckpt_dir <work_dir>/ckpt \
  --knn_indices_path <work_dir>/knn_indices.npy \
  --knn_sims_path <work_dir>/knn_sims.npy \
  --device cuda:0
```
