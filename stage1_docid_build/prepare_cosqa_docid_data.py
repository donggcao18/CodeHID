#!/usr/bin/env python3
"""
Prepare CoSQA positives for vanilla RQ-VAE docid learning.

This script:
1) Loads CoSQA train/dev JSON arrays.
2) Keeps only label == 1 samples.
3) Builds a unified document pool from train+dev (default dedup by code).
4) Writes train/dev intermediate jsonl with stable doc_index for each sample.
"""

import argparse
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple


def normalize_query(text: str) -> str:
    return str(text).strip()


def normalize_code(text: str) -> str:
    # Keep line structure, strip trailing spaces per line for stable dedup.
    lines = [line.rstrip() for line in str(text).splitlines()]
    return "\n".join(lines).strip()


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def load_positive_samples(path: Path, split: str) -> Tuple[List[Dict], Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    kept = []
    stats = {
        "total": len(data),
        "kept_label1": 0,
        "skipped_non_pos": 0,
        "skipped_empty": 0,
    }

    for row in data:
        if row.get("label", 0) != 1:
            stats["skipped_non_pos"] += 1
            continue

        query = normalize_query(row.get("doc", ""))
        code = normalize_code(row.get("code", ""))
        source_idx = str(row.get("idx", ""))

        if not query or not code or not source_idx:
            stats["skipped_empty"] += 1
            continue

        kept.append(
            {
                "split": split,
                "query": query,
                "code": code,
                "source_idx": source_idx,
            }
        )
        stats["kept_label1"] += 1

    return kept, stats


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_json",
        type=str,
        default="train.json",
    )
    parser.add_argument(
        "--dev_json",
        type=str,
        default="dev.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/cosqa_docid_build",
    )
    parser.add_argument(
        "--dedup_key",
        type=str,
        choices=["code", "idx"],
        default="code",
        help="How to build unified doc pool. 'code' is recommended.",
    )
    args = parser.parse_args()

    train_path = Path(args.train_json)
    dev_path = Path(args.dev_json)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_pos, train_stats = load_positive_samples(train_path, "train")
    dev_pos, dev_stats = load_positive_samples(dev_path, "dev")
    all_pos = train_pos + dev_pos

    docs: "OrderedDict[str, Dict]" = OrderedDict()
    train_rows: List[Dict] = []
    dev_rows: List[Dict] = []

    for row in all_pos:
        if args.dedup_key == "code":
            doc_key = sha1_text(row["code"])
        else:
            doc_key = row["source_idx"]

        if doc_key not in docs:
            docs[doc_key] = {
                "doc_index": len(docs),
                "doc_key": doc_key,
                "code": row["code"],
                "first_query": row["query"],
                "first_seen_split": row["split"],
                "first_source_idx": row["source_idx"],
                "positive_count": 0,
            }

        docs[doc_key]["positive_count"] += 1
        doc_index = docs[doc_key]["doc_index"]

        one = {
            "query": row["query"],
            "code": row["code"],
            "source_idx": row["source_idx"],
            "doc_key": doc_key,
            "doc_index": doc_index,
        }
        if row["split"] == "train":
            train_rows.append(one)
        else:
            dev_rows.append(one)

    docs_rows = [v for _, v in docs.items()]
    doc_key_to_index = {k: v["doc_index"] for k, v in docs.items()}

    write_jsonl(out_dir / "docs.jsonl", docs_rows)
    write_jsonl(out_dir / "train_pos_with_doc.jsonl", train_rows)
    write_jsonl(out_dir / "dev_pos_with_doc.jsonl", dev_rows)

    (out_dir / "doc_key_to_index.json").write_text(
        json.dumps(doc_key_to_index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    stats = {
        "input": {
            "train_json": str(train_path),
            "dev_json": str(dev_path),
        },
        "filter_rule": "keep label == 1 only",
        "dedup_key": args.dedup_key,
        "train": train_stats,
        "dev": dev_stats,
        "output": {
            "num_docs_unified": len(docs_rows),
            "num_train_rows": len(train_rows),
            "num_dev_rows": len(dev_rows),
        },
        "files": {
            "docs": str(out_dir / "docs.jsonl"),
            "train": str(out_dir / "train_pos_with_doc.jsonl"),
            "dev": str(out_dir / "dev_pos_with_doc.jsonl"),
            "doc_key_to_index": str(out_dir / "doc_key_to_index.json"),
        },
    }
    (out_dir / "stats_prepare.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("========== Prepare CoSQA for Vanilla RQ-VAE ==========")
    print(f"Train positives: {len(train_rows)}")
    print(f"Dev positives  : {len(dev_rows)}")
    print(f"Unified docs   : {len(docs_rows)}")
    print(f"Saved to       : {out_dir}")


if __name__ == "__main__":
    main()
