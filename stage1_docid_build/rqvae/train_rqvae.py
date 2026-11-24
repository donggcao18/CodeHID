import argparse
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from embedding_dataset import EmbDataset
from models.rqvae import RQVAE
from rqvae_trainer import Trainer


def _env_int(name, default):
    return int(os.getenv(name, str(default)))


def _env_float(name, default):
    return float(os.getenv(name, str(default)))


def _env_bool(name, default):
    return os.getenv(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "on"}


def _env_float_list(name, default):
    raw = os.getenv(name)
    if not raw:
        return default
    return [float(x) for x in raw.replace(",", " ").split()]


def parse_args():
    parser = argparse.ArgumentParser(description="Index")

    work_dir = os.getenv("WORK_DIR")
    data_default = os.path.join(work_dir, "docs.emb-codebert.npy") if work_dir else "data/cosqa_docid_build/docs.emb-codebert.npy"
    ckpt_default = os.path.join(work_dir, "ckpt") if work_dir else "ckpt/cosqa_docid_build"
    knn_indices_default = os.path.join(work_dir, "knn_indices.npy") if work_dir else "data/cosqa_docid_build/knn_indices.npy"
    knn_sims_default = os.path.join(work_dir, "knn_sims.npy") if work_dir else "data/cosqa_docid_build/knn_sims.npy"

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=_env_int("KNN_BATCH_SIZE", 2048), help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, )
    parser.add_argument('--eval_step', type=int, default=10, help='eval step')
    parser.add_argument('--learner', type=str, default="AdamW", help='optimizer')
    parser.add_argument('--lr_scheduler_type', type=str, default="linear", help='scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=20, help='warmup epochs')
    parser.add_argument(
        "--data_path",
        type=str,
        default=data_default,
        help="Input embedding path (.npy or .parquet).",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4, help='l2 regularization weight')
    parser.add_argument("--dropout_prob", type=float, default=0.0, help="dropout ratio")
    parser.add_argument("--bn", type=bool, default=False, help="use bn or not")
    parser.add_argument("--loss_type", type=str, default="mse", help="loss_type")
    parser.add_argument("--kmeans_init", type=bool, default=True, help="use kmeans_init or not")
    parser.add_argument("--kmeans_iters", type=int, default=100, help="max kmeans iters")
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.0, 0.0, 0.0, 0.003], help="sinkhorn epsilons")
    parser.add_argument("--sk_iters", type=int, default=50, help="max sinkhorn iters")

    parser.add_argument("--device", type=str, default="cuda:0", help="gpu or cpu")

    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[256,256,256,256], help='emb num of every vq')
    parser.add_argument('--e_dim', type=int, default=32, help='vq codebook embedding size')
    parser.add_argument('--quant_loss_weight', type=float, default=1.0, help='vq quantion loss weight')
    parser.add_argument("--beta", type=float, default=0.25, help="Beta for commitment loss")
    parser.add_argument('--layers', type=int, nargs='+', default=[512,256,128,64], help='hidden sizes of every layer')
    parser.add_argument('--save_limit', type=int, default=5, help='save limit for ckpt')
    
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=ckpt_default,
        help="please specify output directory for model",
    )
    parser.add_argument(
        "--use_knn_prefix_loss",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("USE_KNN_PREFIX_LOSS", True),
        help="Enable KNN pseudo-hierarchical prefix loss.",
    )
    parser.add_argument(
        "--knn_indices_path",
        type=str,
        default=knn_indices_default,
        help="Path to KNN neighbor indices (N x K) built from embeddings.",
    )
    parser.add_argument(
        "--knn_sims_path",
        type=str,
        default=knn_sims_default,
        help="Optional KNN cosine similarity matrix aligned with knn_indices.",
    )
    parser.add_argument("--knn_k", type=int, default=_env_int("KNN_K", 8), help="Number of KNN neighbors to use.")
    parser.add_argument("--knn_loss_weight", type=float, default=_env_float("KNN_LOSS_WEIGHT", 0.03), help="Weight for KNN prefix loss.")
    parser.add_argument(
        "--knn_loss_schedule",
        type=str,
        default=os.getenv("KNN_LOSS_SCHEDULE", "linear_decay_to_zero"),
        choices=["constant", "linear_decay_to_zero"],
        help="Schedule for KNN loss weight across epochs.",
    )
    parser.add_argument(
        "--knn_decay_start_ratio",
        type=float,
        default=_env_float("KNN_DECAY_START_RATIO", 0.3),
        help="Epoch ratio to start decaying KNN weight when using linear_decay_to_zero.",
    )
    parser.add_argument(
        "--knn_decay_end_ratio",
        type=float,
        default=_env_float("KNN_DECAY_END_RATIO", 0.7),
        help="Epoch ratio to reach zero KNN weight when using linear_decay_to_zero.",
    )
    parser.add_argument(
        "--knn_temperature",
        type=float,
        default=_env_float("KNN_TEMPERATURE", 0.35),
        help="Softmax temperature for per-level assignment probability.",
    )
    parser.add_argument(
        "--knn_prefix_weights",
        type=float,
        nargs="+",
        default=_env_float_list("KNN_PREFIX_WEIGHTS", [0.8, 0.35, 0.12, 0.03]),
        help="Per-level loss weights for prefix supervision.",
    )
    parser.add_argument(
        "--knn_thresholds",
        type=float,
        nargs="+",
        default=[0.55, 0.65, 0.75, 0.82],
        help="Manual cosine similarity thresholds for target prefix depth supervision.",
    )
    parser.add_argument(
        "--knn_auto_thresholds",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("KNN_AUTO_THRESHOLDS", True),
        help="Auto-estimate thresholds from knn_sims quantiles.",
    )
    parser.add_argument(
        "--knn_threshold_quantiles",
        type=float,
        nargs="+",
        default=_env_float_list("KNN_THRESHOLD_QUANTILES", [0.85, 0.95, 0.99, 0.999]),
        help="Quantiles used when --knn_auto_thresholds is enabled.",
    )
    parser.add_argument(
        "--knn_num_negatives",
        type=int,
        default=_env_int("KNN_NUM_NEGATIVES", 4),
        help="Number of in-batch negatives sampled per anchor for KNN prefix loss.",
    )
    parser.add_argument(
        "--knn_negative_weight",
        type=float,
        default=_env_float("KNN_NEGATIVE_WEIGHT", 0.6),
        help="Weight of negative-pair BCE term in KNN prefix loss.",
    )
    return parser.parse_args()


if __name__ == '__main__':
    """fix the random seed"""
    seed = 2024
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()
    print("=================================================")
    print(args)
    print("=================================================")

    logging.basicConfig(level=logging.DEBUG)

    """build dataset"""
    data = EmbDataset(args.data_path, return_index=args.use_knn_prefix_loss)
    model = RQVAE(in_dim=data.dim,
                  num_emb_list=args.num_emb_list,
                  e_dim=args.e_dim,
                  layers=args.layers,
                  dropout_prob=args.dropout_prob,
                  bn=args.bn,
                  loss_type=args.loss_type,
                  quant_loss_weight=args.quant_loss_weight,
                  beta=args.beta,
                  kmeans_init=args.kmeans_init,
                  kmeans_iters=args.kmeans_iters,
                  sk_epsilons=args.sk_epsilons,
                  sk_iters=args.sk_iters,
                  )
    print(model)
    data_loader = DataLoader(
        data,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
    )
    trainer = Trainer(args, model, len(data_loader))
    best_loss, best_collision_rate = trainer.fit(data_loader)

    print("Best Loss",best_loss)
    print("Best Collision Rate", best_collision_rate)
