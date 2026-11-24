import argparse
from pathlib import Path

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Build exact KNN graph from embedding vectors.")
    parser.add_argument("--data_path", type=str, required=True, help="Input embedding file (.npy or .parquet).")
    parser.add_argument("--output_indices", type=str, required=True, help="Output path for knn indices .npy.")
    parser.add_argument("--output_sims", type=str, default=None, help="Optional output path for knn cosine sims.")
    parser.add_argument("--k", type=int, default=16, help="Top-K neighbors for each sample.")
    parser.add_argument("--batch_size", type=int, default=2048, help="Chunk size for query rows.")
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda:0 or cpu.")
    return parser.parse_args()


def load_embeddings(path):
    path = str(path)
    if path.endswith(".npy"):
        return np.load(path)
    if path.endswith(".parquet"):
        import pandas as pd

        values = pd.read_parquet(path)["embedding"].values
        return np.stack(values, axis=0)
    raise ValueError(f"Unsupported embedding format: {path}")


def main():
    args = parse_args()
    emb = load_embeddings(args.data_path).astype(np.float32)
    num_docs = emb.shape[0]
    if num_docs <= 1:
        raise ValueError(f"Need at least 2 docs to build KNN, got {num_docs}.")
    k = min(args.k, num_docs - 1)
    if k <= 0:
        raise ValueError(f"Invalid k={args.k} for num_docs={num_docs}.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    emb_t = torch.from_numpy(emb).to(device)
    emb_t = torch.nn.functional.normalize(emb_t, p=2, dim=1)

    knn_indices = np.zeros((num_docs, k), dtype=np.int64)
    knn_sims = np.zeros((num_docs, k), dtype=np.float32)

    print(f"Building KNN: num_docs={num_docs}, dim={emb.shape[1]}, k={k}, device={device}")
    with torch.no_grad():
        for start in range(0, num_docs, args.batch_size):
            end = min(start + args.batch_size, num_docs)
            q = emb_t[start:end]
            sims = q @ emb_t.T
            rows = torch.arange(end - start, device=device)
            cols = torch.arange(start, end, device=device)
            sims[rows, cols] = -1e9
            top_vals, top_idx = torch.topk(sims, k=k, dim=1, largest=True, sorted=True)
            knn_indices[start:end] = top_idx.cpu().numpy().astype(np.int64)
            knn_sims[start:end] = top_vals.cpu().numpy().astype(np.float32)
            if (start // args.batch_size) % 10 == 0:
                print(f"processed: {end}/{num_docs}")

    Path(args.output_indices).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_indices, knn_indices)
    print(f"Saved indices: {args.output_indices} shape={knn_indices.shape}")

    if args.output_sims:
        Path(args.output_sims).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output_sims, knn_sims)
        print(f"Saved sims: {args.output_sims} shape={knn_sims.shape}")


if __name__ == "__main__":
    main()
