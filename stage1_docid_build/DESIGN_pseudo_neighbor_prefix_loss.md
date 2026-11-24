# RQ-VAE + KNN Pseudo-Hierarchical Prefix Supervision

## 1. Objective

When manual hierarchical labels are unavailable (for example, how many prefix levels two documents should share), we use local neighborhood structure in embedding space as weak supervision for multi-level RQ-VAE codes `c0,c1,c2,c3`.

The training objective combines:

- `L_rqvae`: reconstruction and quantization quality from the base RQ-VAE objective.
- `L_knn_prefix`: encourages semantically close neighbors to share longer code prefixes.

Total loss:

`L = L_rqvae + lambda_knn * L_knn_prefix`

where `L_rqvae = L_recon + lambda_q * L_quant`.

## 2. Building Pseudo Hierarchical Targets

Given document embeddings `e_i`:

1. Compute pairwise similarity (usually cosine).
2. Build `K`-NN graph and keep neighbor set `N_K(i)`.
3. Map similarity to soft supervision `y_ij^(l) in [0,1]` for each prefix depth `l`.

Interpretation: higher similarity implies stronger preference for deeper shared prefixes.

## 3. Differentiable Prefix-Sharing Proxy

Let `p_i^(t)(k)` be the soft assignment probability of document `i` to code `k` at level `t`.

- Same-code probability at level `t`:
  `P_t(i,j) = sum_k p_i^(t)(k) * p_j^(t)(k)`
- Prefix-length-at-least-`l` proxy:
  `y_hat_ij^(l) = product_{t=1..l} P_t(i,j)`

This gives a differentiable approximation for shared-prefix constraints.

## 4. KNN Prefix Loss

For each KNN edge `(i,j)` and each level `l`, apply binary supervision (for example BCE):

`L_knn_prefix = sum_{(i,j),l} w_l * BCE(y_hat_ij^(l), y_ij^(l))`

Practical notes:

- Use larger weights on shallow levels first, e.g. `w=[1.0, 0.7, 0.4, 0.2]`.
- Optional far-neighbor negatives help prevent collapse.

## 5. Training Pipeline

1. **Offline graph building**: precompute `knn_indices.npy` (and optional similarities).
2. **Data loading**: each sample exposes `(embedding, doc_id)` and its KNN neighbors.
3. **Forward**:
   - compute `L_rqvae`;
   - export per-level soft assignments.
4. **Neighborhood loss**: compute `L_knn_prefix` using visible neighbors.
5. **Backprop** with total objective.

## 6. Mapping to This Project

- Keep existing RQ-VAE loss path unchanged.
- Add KNN prefix term in trainer-level loss composition.
- If current implementation only exposes hard `argmin` indices, add soft assignment outputs from quantizer for differentiability.

## 7. Suggested Hyperparameters (Initial)

- `K=16`
- `lambda_knn in {0.05, 0.1, 0.2}`
- assignment temperature `0.1 ~ 0.5`
- threshold candidates (data-dependent), e.g. `tau=[0.55, 0.65, 0.75, 0.82]`

## 8. Intuition

RQ-VAE discretizes continuous semantics into hierarchical DocIDs.  
The KNN prefix loss injects the bias: semantically closer documents should share longer prefixes.  
As a result, `a{c0}b{c1}c{c2}d{c3}` preserves both reconstructability and local semantic structure.
