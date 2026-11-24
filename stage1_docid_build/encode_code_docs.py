#!/usr/bin/env python3
"""
Encode unified docs into dense vectors for vanilla RQ-VAE.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def build_text(row: Dict, mode: str) -> str:
    code = row.get("code", "").strip()
    if mode == "code":
        return code
    query = row.get("first_query", "").strip()
    return f"Query: {query}\nCode:\n{code}"


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    x = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1e-6)
    return x / denom


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--docs_jsonl",
        type=str,
        default="data/cosqa_docid_build/docs.jsonl",
    )
    parser.add_argument(
        "--output_npy",
        type=str,
        default="data/cosqa_docid_build/docs.emb-codebert.npy",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/codebert-base",
    )
    parser.add_argument("--text_mode", type=str, choices=["code", "query_code"], default="code")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--l2_normalize", action="store_true", default=False)
    args = parser.parse_args()

    docs_path = Path(args.docs_jsonl)
    out_path = Path(args.output_npy)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(docs_path)
    rows = sorted(rows, key=lambda x: int(x["doc_index"]))
    texts = [build_text(row, args.text_mode) for row in rows]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)
    model.eval()

    embs: List[np.ndarray] = []
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), args.batch_size), desc="Encoding"):
            batch_text = texts[start : start + args.batch_size]
            tok = tokenizer(
                batch_text,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            tok = {k: v.to(device) for k, v in tok.items()}
            out = model(**tok)
            pooled = mean_pool(out.last_hidden_state, tok["attention_mask"])
            if args.l2_normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
            embs.append(pooled.cpu().numpy().astype(np.float32))

    emb = np.concatenate(embs, axis=0)
    np.save(out_path, emb)

    meta = {
        "docs_jsonl": str(docs_path),
        "output_npy": str(out_path),
        "model_name": args.model_name,
        "text_mode": args.text_mode,
        "num_docs": int(emb.shape[0]),
        "dim": int(emb.shape[1]),
        "max_length": args.max_length,
        "l2_normalize": bool(args.l2_normalize),
    }
    (out_path.with_suffix(".meta.json")).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("========== Encode Docs for Vanilla RQ-VAE ==========")
    print(f"Num docs : {emb.shape[0]}")
    print(f"Dim      : {emb.shape[1]}")
    print(f"Saved npy: {out_path}")


if __name__ == "__main__":
    main()
