#!/usr/bin/env python3
"""
Rewrite CoSQA train/dev jsonl with unified docids from vanilla RQ-VAE codes.

Default docid style:
  a1b2c3d4
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_code_item(x) -> int:
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, str):
        m = re.search(r"-?\d+", x)
        if m:
            return int(m.group(0))
    raise ValueError(f"Unsupported code item: {x!r}")


def load_codes(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise ValueError(f"Codes must be 2D, got shape {arr.shape}")
        return arr.astype(np.int64)

    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            rows = []
            for i in range(len(data)):
                one = data[str(i)]
                rows.append([parse_code_item(v) for v in one])
            return np.array(rows, dtype=np.int64)
        if isinstance(data, list):
            rows = [[parse_code_item(v) for v in one] for one in data]
            return np.array(rows, dtype=np.int64)

    raise ValueError(f"Unsupported codes file: {path}")


def _format_docid_from_code_tuple(code_tuple: Tuple[int, ...], prefix: str, level_tags: str) -> str:
    if len(code_tuple) == 0:
        raise ValueError("Empty code tuple is not allowed.")
    if len(code_tuple) > len(level_tags):
        raise ValueError(
            f"Code levels={len(code_tuple)} exceed available level tags={len(level_tags)} "
            f"('{level_tags}')."
        )

    parts = []
    for i, value in enumerate(code_tuple):
        v = int(value)
        if v < 0:
            raise ValueError(f"Negative code value is not supported in docid format: {code_tuple}")
        parts.append(f"{level_tags[i]}{v}")
    return f"{prefix}{''.join(parts)}"


def build_docid_map_by_code_tuple(
    codes: np.ndarray,
    prefix: str,
    level_tags: str,
    allow_shared_docid: bool = False,
) -> Tuple[Dict[int, str], int, int, Dict[int, Tuple[int, ...]]]:
    mapping: Dict[int, str] = {}
    assigned_codes: Dict[int, Tuple[int, ...]] = {}
    used_docids: Dict[str, int] = {}
    collisions_resolved = 0
    collisions_shared = 0

    for doc_index in range(int(codes.shape[0])):
        base_tuple = tuple(int(v) for v in codes[doc_index].tolist())
        code_tuple = base_tuple
        docid = _format_docid_from_code_tuple(
            code_tuple=code_tuple,
            prefix=prefix,
            level_tags=level_tags,
        )
        if docid in used_docids:
            if allow_shared_docid:
                collisions_shared += 1
            else:
                # Keep pure hierarchical format (a/b/c/d only), no extra suffix.
                # Resolve collision by shifting the last level digit deterministically.
                last_digit = int(base_tuple[-1])
                offset = 1
                while True:
                    candidate_tuple = list(base_tuple)
                    candidate_tuple[-1] = last_digit + offset
                    candidate_tuple = tuple(candidate_tuple)
                    candidate_docid = _format_docid_from_code_tuple(
                        code_tuple=candidate_tuple,
                        prefix=prefix,
                        level_tags=level_tags,
                    )
                    if candidate_docid not in used_docids:
                        code_tuple = candidate_tuple
                        docid = candidate_docid
                        collisions_resolved += 1
                        break
                    offset += 1
        used_docids[docid] = doc_index
        mapping[doc_index] = docid
        assigned_codes[doc_index] = code_tuple
    return mapping, collisions_resolved, collisions_shared, assigned_codes


def rewrite_split(rows: List[Dict], doc_index_to_docid: Dict[int, str]) -> List[Dict]:
    out = []
    for row in rows:
        doc_index = int(row["doc_index"])
        docid = doc_index_to_docid[doc_index]
        out.append(
            {
                "query": row["query"],
                "code": row["code"],
                "docid": docid,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--docs_jsonl",
        type=str,
        default="data/cosqa_docid_build/docs.jsonl",
    )
    parser.add_argument(
        "--train_input",
        type=str,
        default="data/cosqa_docid_build/train_pos_with_doc.jsonl",
    )
    parser.add_argument(
        "--dev_input",
        type=str,
        default="data/cosqa_docid_build/dev_pos_with_doc.jsonl",
    )
    parser.add_argument(
        "--codes_path",
        type=str,
        required=True,
        help="RQ-VAE output codes file (.npy or .json). Order must match docs.jsonl doc_index.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/cosqa_docid_build/final",
    )
    parser.add_argument("--docid_prefix", type=str, default="")
    parser.add_argument(
        "--level_tags",
        type=str,
        default="abcd",
        help="Level labels for hierarchical docids, e.g. 'abcd' -> a1b2c3d4.",
    )
    parser.add_argument(
        "--docid_width",
        type=int,
        default=8,
        help="Deprecated for transitional mode; kept for backward-compatible CLI.",
    )
    parser.add_argument(
        "--allow-shared-docid",
        action="store_true",
        help=(
            "Allow multiple doc_index entries to share the same hierarchical docid "
            "when code tuples collide. Default OFF keeps collision splitting."
        ),
    )
    args = parser.parse_args()

    docs_path = Path(args.docs_jsonl)
    train_path = Path(args.train_input)
    dev_path = Path(args.dev_input)
    codes_path = Path(args.codes_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    docs = load_jsonl(docs_path)
    docs = sorted(docs, key=lambda x: int(x["doc_index"]))
    train_rows = load_jsonl(train_path)
    dev_rows = load_jsonl(dev_path)
    codes = load_codes(codes_path)

    if len(docs) != int(codes.shape[0]):
        raise ValueError(
            f"docs size ({len(docs)}) != codes rows ({codes.shape[0]})."
        )

    doc_index_to_docid, collisions_resolved, collisions_shared, assigned_codes = build_docid_map_by_code_tuple(
        codes=codes,
        prefix=args.docid_prefix,
        level_tags=args.level_tags,
        allow_shared_docid=bool(args.allow_shared_docid),
    )

    train_out = rewrite_split(train_rows, doc_index_to_docid)
    dev_out = rewrite_split(dev_rows, doc_index_to_docid)

    # Clean up legacy prefixed output names if they exist from older runs.
    removed_legacy_files = []
    for legacy_name in ("cosqa_train.jsonl", "cosqa_valid.jsonl"):
        legacy_path = out_dir / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()
            removed_legacy_files.append(legacy_name)

    write_jsonl(out_dir / "train.jsonl", train_out)
    write_jsonl(out_dir / "valid.jsonl", dev_out)

    doc_key_to_docid = {}
    for doc in docs:
        doc_index = int(doc["doc_index"])
        doc_key_to_docid[doc["doc_key"]] = doc_index_to_docid[doc_index]

    docid_to_code = {}
    for i in range(int(codes.shape[0])):
        docid = doc_index_to_docid[i]
        code_tuple = tuple(int(v) for v in assigned_codes[i])
        docid_to_code[docid] = list(code_tuple)

    (out_dir / "doc_index_to_docid.json").write_text(
        json.dumps({str(k): v for k, v in doc_index_to_docid.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "doc_key_to_docid.json").write_text(
        json.dumps(doc_key_to_docid, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "docid_to_code.json").write_text(
        json.dumps(docid_to_code, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    unique_codes = len({tuple(int(v) for v in row.tolist()) for row in codes})
    stats = {
        "docs": len(docs),
        "code_shape": list(codes.shape),
        "unique_code_tuples": unique_codes,
        "unique_docids": len(set(doc_index_to_docid.values())),
        "docid_collisions_resolved": int(collisions_resolved),
        "docid_collisions_shared": int(collisions_shared),
        "allow_shared_docid": bool(args.allow_shared_docid),
        "train_rows": len(train_out),
        "dev_rows": len(dev_out),
        "docid_prefix": args.docid_prefix,
        "docid_style": "hierarchical_levels",
        "level_tags": args.level_tags,
        "docid_width_deprecated": args.docid_width,
        "removed_legacy_files": removed_legacy_files,
        "output_dir": str(out_dir),
    }
    (out_dir / "stats_rewrite.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("========== Rewrite CoSQA with RQ-VAE DocIDs ==========")
    print(f"Docs        : {len(docs)}")
    print(f"Codes shape : {codes.shape}")
    if args.allow_shared_docid:
        print(f"Collisions  : {collisions_shared} (shared docid mode)")
    else:
        print(f"Collisions  : {collisions_resolved} (resolved by shifting d-level)")
    print(f"Train rows  : {len(train_out)}")
    print(f"Dev rows    : {len(dev_out)}")
    if len(removed_legacy_files) > 0:
        print(f"Cleaned old : {', '.join(removed_legacy_files)}")
    print(f"Saved to    : {out_dir}")


if __name__ == "__main__":
    main()
