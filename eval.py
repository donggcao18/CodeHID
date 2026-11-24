import os
import argparse
import datetime
import glob
import json
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from codehid.models.codehid_codebert import CodeHIDCodeBERTDSI


def _first_existing_path(*paths: Optional[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return None


def parse_args():
    codegr_root = os.environ.get("CODEHID_ROOT", os.environ.get("CODEGR_ROOT", os.getcwd()))
    default_dataset_final_dir = os.environ.get(
        "DATASET_FINAL_DIR",
        os.path.join(codegr_root, "stage1_docid_build", "data", "cosqa_docid_build", "final"),
    )
    default_valid_jsonl = (
        os.environ.get("EVAL_VALID_JSONL_PATH")
        or os.environ.get("VALID_JSONL_PATH")
        or _first_existing_path(
            os.path.join(default_dataset_final_dir, "valid.jsonl"),
            os.path.join(default_dataset_final_dir, "cosqa_valid.jsonl"),
        )
        or os.path.join(default_dataset_final_dir, "valid.jsonl")
    )
    default_docid_vocab = (
        os.environ.get("EVAL_DOCID_VOCAB_PATH")
        or os.environ.get("DOCID_VOCAB_PATH")
        or os.path.join(default_dataset_final_dir, "doc_index_to_docid.json")
    )
    default_dual_phase_guidance_infer_doc_emb = (
        os.environ.get("DUAL_PHASE_GUIDANCE_INFER_DOC_EMB_PATH")
        or _first_existing_path(
            os.path.join(os.path.dirname(default_dataset_final_dir.rstrip("/")), "docs.emb-graphcodebert.npy"),
            os.path.join(os.path.dirname(default_dataset_final_dir.rstrip("/")), "docs.emb-codebert.npy"),
        )
        or os.path.join(os.path.dirname(default_dataset_final_dir.rstrip("/")), "docs.emb-graphcodebert.npy")
    )
    default_dual_phase_guidance_infer_doc_index = os.environ.get("DUAL_PHASE_GUIDANCE_INFER_DOC_INDEX_TO_DOCID_PATH", None)

    parser = argparse.ArgumentParser(description="Manual eval for CoSQA checkpoint")
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=None,
        help="Evaluate a single checkpoint path (highest priority).",
    )
    parser.add_argument(
        "--ckpt-mode",
        type=str,
        default="best",
        choices=["best", "last4", "best_and_last4"],
        help="When --ckpt-path is not set, select checkpoints from a run dir.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run directory like output/codehid_codebert_dsi/<run_id>. If omitted, auto-pick latest run dir.",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="Path to save per-sample predictions as JSON. Default: <ckpt_dir>/eval_valid_predictions.json",
    )
    parser.add_argument(
        "--save-topk",
        type=int,
        default=20,
        help="How many generated docids to save per sample.",
    )
    parser.add_argument(
        "--valid-jsonl-path",
        "--dataset-path",
        dest="valid_jsonl_path",
        type=str,
        default=default_valid_jsonl,
        help="Validation jsonl path used for manual evaluation. Alias: --dataset-path",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=20,
        help="Beam size used in generation (recommend 20~30 for stable top-k).",
    )
    parser.add_argument(
        "--dual-phase-guidance-infer-enable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable dual-phase-guidance infer query-conditioned retrieval + lexical temporary rescoring.",
    )
    parser.add_argument(
        "--dual-phase-guidance-infer-lexical-topk",
        type=int,
        default=200,
        help="Top-L lexical candidate docs per query for query-conditioned trie.",
    )
    parser.add_argument(
        "--dual-phase-guidance-infer-lexical-source",
        type=str,
        default="code",
        choices=["code", "query", "query_code"],
        help="Text source for lexical retrieval embedding.",
    )
    parser.add_argument(
        "--dual-phase-guidance-infer-lexical-bonus-weight",
        type=float,
        default=0.10,
        help="Bonus weight for temporary lexical rescoring during beam decoding.",
    )
    parser.add_argument(
        "--dual-phase-guidance-infer-lexical-pooling",
        type=str,
        default="max",
        choices=["max", "mean"],
        help="Pooling strategy when multiple docs share same next token under a prefix.",
    )
    parser.add_argument(
        "--dual-phase-guidance-infer-uniform-bonus",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ignore lexical score magnitudes and assign equal bonus to all lexical candidates.",
    )
    parser.add_argument(
        "--dual-phase-guidance-infer-doc-emb-path",
        type=str,
        default=default_dual_phase_guidance_infer_doc_emb,
        help="Dense lexical doc embeddings for candidate retrieval.",
    )
    parser.add_argument(
        "--dual-phase-guidance-infer-doc-index-to-docid-path",
        type=str,
        default=default_dual_phase_guidance_infer_doc_index,
        help="Mapping path from doc_index to rewritten docid.",
    )
    parser.add_argument(
        "--dual-phase-guidance-infer-restrict-to-legal-vocab",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter lexical candidates by constrained legal docid vocabulary.",
    )
    parser.add_argument(
        "--dual-phase-guidance-infer-use-query-trie",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use query-conditioned candidate trie instead of one global trie.",
    )
    parser.add_argument(
        "--dual-phase-guidance-infer-debug-topn",
        type=int,
        default=0,
        help="How many lexical candidates to print in debug logs per sample.",
    )
    parser.add_argument(
        "--constrained-decode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use legal-docid trie constraints during generation.",
    )
    parser.add_argument(
        "--docid-vocab-path",
        type=str,
        default=default_docid_vocab,
        help="Path to legal docid vocabulary (.json mapping or .jsonl with docid field).",
    )
    parser.add_argument(
        "--use-patch-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Override checkpoint model.use_patch_tokens for evaluation (default: use checkpoint value).",
    )
    parser.add_argument(
        "--patch-token-count",
        type=int,
        default=4,
        help="Override checkpoint model.patch_token_count for evaluation.",
    )
    parser.add_argument(
        "--patch-pooling",
        type=str,
        default="mean",
        choices=["mean", "stride"],
        help="Override checkpoint model.patch_pooling for evaluation.",
    )
    parser.add_argument(
        "--debug-log-path",
        type=str,
        default=os.environ.get(
            "DEBUG_LOG_PATH",
            os.path.join(codegr_root, "output", "debug_constrained_decode.log"),
        ),
        help="Write constrained decoding debug logs to this file.",
    )
    args = parser.parse_args()
    if not args.dual_phase_guidance_infer_doc_index_to_docid_path:
        args.dual_phase_guidance_infer_doc_index_to_docid_path = args.docid_vocab_path
    return args


def _append_debug_log(log_path: str, message: str):
    if not log_path:
        return
    log_path = str(log_path)
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(str(message).rstrip("\n") + "\n")


def _init_debug_log(log_path: str, args):
    if not log_path:
        return
    log_path = str(log_path)
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"[DBG][session] start={ts}\n")
        f.write(f"[DBG][session] argv={vars(args)}\n")


def _latest_run_dir(run_dir: str = None) -> str:
    if run_dir:
        return run_dir

    codegr_root = os.environ.get("CODEHID_ROOT", os.environ.get("CODEGR_ROOT", os.getcwd()))
    output_base = os.environ.get("OUTPUT_BASE", os.path.join(codegr_root, "output", "codehid_codebert_dsi"))
    pattern = os.path.join(output_base, "*")
    cands = [p for p in glob.glob(pattern) if os.path.isdir(p)]
    if not cands:
        raise FileNotFoundError(f"No run dir matched: {pattern}")
    cands = sorted(cands, key=os.path.getmtime)
    return cands[-1]


def _epoch_from_name(path: str) -> int:
    base = os.path.basename(path)
    # checkpoint_epoch_18.pth -> 18
    return int(base.split("_")[-1].split(".")[0])


def resolve_ckpt_paths(ckpt_path: str = None, ckpt_mode: str = "best", run_dir: str = None):
    if ckpt_path:
        return [ckpt_path]

    picked_run_dir = _latest_run_dir(run_dir)

    ckpts = []
    if ckpt_mode in ("best", "best_and_last4"):
        best_path = os.path.join(picked_run_dir, "checkpoint_best.pth")
        if os.path.exists(best_path):
            ckpts.append(best_path)
        elif ckpt_mode == "best":
            raise FileNotFoundError(f"checkpoint_best.pth not found in {picked_run_dir}")

    if ckpt_mode in ("last4", "best_and_last4"):
        epoch_cands = glob.glob(os.path.join(picked_run_dir, "checkpoint_epoch_*.pth"))
        if not epoch_cands:
            if ckpt_mode == "last4":
                raise FileNotFoundError(f"No checkpoint_epoch_*.pth found in {picked_run_dir}")
        else:
            epoch_cands = sorted(epoch_cands, key=_epoch_from_name)
            ckpts.extend(epoch_cands[-4:])

    # Keep order, drop duplicates (best may overlap with epoch ckpt copy path in some workflows)
    dedup = []
    seen = set()
    for p in ckpts:
        if p in seen:
            continue
        seen.add(p)
        dedup.append(p)

    if not dedup:
        raise FileNotFoundError(f"No checkpoints resolved under {picked_run_dir} with mode={ckpt_mode}")
    return dedup


def _extract_model_cfg_from_ckpt(ckpt_obj):
    if not isinstance(ckpt_obj, dict):
        return {}
    cfg = ckpt_obj.get("config")
    if not isinstance(cfg, dict):
        return {}
    model_cfg = cfg.get("model")
    if not isinstance(model_cfg, dict):
        return {}
    return model_cfg


def build_model(model_cfg=None):
    cfg = dict(model_cfg or {})
    default_encoder_name = os.environ.get("CODEBERT_ENCODER_NAME", "microsoft/graphcodebert-base")
    default_decoder_name = os.environ.get("DECODER_NAME", "Qwen/Qwen2.5-Coder-7B")
    return CodeHIDCodeBERTDSI(
        encoder_name=cfg.get("encoder_name", default_encoder_name),
        decoder_name=cfg.get("decoder_name", default_decoder_name),
        use_llm=bool(cfg.get("use_llm", True)),
        hidden_size=int(cfg.get("hidden_size", 3584)),
        freeze_encoder=bool(cfg.get("freeze_encoder", True)),
        freeze_decoder=bool(cfg.get("freeze_decoder", True)),
        encoder_max_length=int(cfg.get("encoder_max_length", 512)),
        decoder_max_length=int(cfg.get("decoder_max_length", 64)),
        decoder_dtype=str(cfg.get("decoder_dtype", "float16")),
        enable_gradient_checkpointing=bool(cfg.get("enable_gradient_checkpointing", False)),
        use_lora=bool(cfg.get("use_lora", True)),
        lora_r=int(cfg.get("lora_r", 8)),
        lora_alpha=int(cfg.get("lora_alpha", 16)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        lora_target_modules=cfg.get("lora_target_modules", None),
        use_patch_tokens=bool(cfg.get("use_patch_tokens", True)),
        patch_token_count=int(cfg.get("patch_token_count", 4)),
        patch_pooling=str(cfg.get("patch_pooling", "mean")),
        enable_alignment_proj=bool(cfg.get("enable_alignment_proj", True)),
        enable_patch_linear=bool(cfg.get("enable_patch_linear", True)),
        prefix_length=int(cfg.get("prefix_length", 16)),
        docid_prefix=str(cfg.get("docid_prefix", "")),
        gradual_unfreeze=bool(cfg.get("gradual_unfreeze", False)),
        unfreeze_prefix_epoch=int(cfg.get("unfreeze_prefix_epoch", 0)),
        unfreeze_alignment_epoch=int(cfg.get("unfreeze_alignment_epoch", 1)),
        unfreeze_embed_epoch=int(cfg.get("unfreeze_embed_epoch", 2)),
        unfreeze_lm_head_epoch=int(cfg.get("unfreeze_lm_head_epoch", 3)),
        unfreeze_decoder_embed_and_lm_head=bool(cfg.get("unfreeze_decoder_embed_and_lm_head", True)),
        label_smoothing=float(cfg.get("label_smoothing", 0.0)),
    )


def apply_eval_model_overrides(model_cfg, args):
    cfg = dict(model_cfg or {})
    if args.use_patch_tokens is not None:
        cfg["use_patch_tokens"] = bool(args.use_patch_tokens)
    if args.patch_token_count is not None:
        cfg["patch_token_count"] = int(args.patch_token_count)
    if args.patch_pooling is not None:
        cfg["patch_pooling"] = str(args.patch_pooling)
    return cfg


# =========================
# 0. Inference dtype alignment
# =========================
def fix_infer_dtypes(model):
    """
    Normalize dtypes at inference to avoid:
      - alignment_proj mat1/mat2 dtype mismatch
      - lm_head expected Half but found Float

    Recommended inference setup:
      - alignment_proj: FP32 (CodeBERT outputs are typically FP32)
      - decoder/emb/lm_head: FP16 (match decoder hidden_states)
    """
    # Keep alignment layers in FP32.
    model.alignment_proj.to(dtype=torch.float32)
    if hasattr(model, "patch_linear") and model.patch_linear is not None:
        model.patch_linear.to(dtype=torch.float32)

    # Keep decoder-side modules in FP16.
    model.decoder.to(dtype=torch.float16)
    model.decoder.get_input_embeddings().to(dtype=torch.float16)
    model.decoder.get_output_embeddings().to(dtype=torch.float16)
    if hasattr(model.decoder, "lm_head") and model.decoder.lm_head is not None:
        model.decoder.lm_head.to(dtype=torch.float16)

    # Print a quick dtype summary for debugging.
    try:
        print(
            "[DTYPE CHECK]",
            "alignment_proj:", model.alignment_proj.weight.dtype,
            "| patch_linear:", model.patch_linear.weight.dtype if hasattr(model, "patch_linear") else "N/A",
            "| decoder(any):", next(model.decoder.parameters()).dtype,
            "| emb:", next(model.decoder.get_input_embeddings().parameters()).dtype,
            "| lm_head:", next(model.decoder.get_output_embeddings().parameters()).dtype,
        )
    except Exception as e:
        print("Could not inspect dtypes:", repr(e))


# =========================
# 1. Load CoSQA jsonl
# =========================
def load_cosqa_jsonl(path, max_samples=None):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
            if max_samples is not None and len(data) >= max_samples:
                break
    return data


def load_docid_vocab(path: str):
    p = str(path)
    if p.endswith(".json"):
        data = json.load(open(p, "r", encoding="utf-8"))
        if isinstance(data, dict):
            vals = []
            for v in data.values():
                if isinstance(v, str) and v.strip():
                    vals.append(v.strip())
            if len(vals) == 0:
                raise ValueError(f"No docid strings found in JSON dict: {p}")
            return sorted(list(set(vals)))
        if isinstance(data, list):
            vals = [str(x).strip() for x in data if isinstance(x, str) and str(x).strip()]
            if len(vals) == 0:
                raise ValueError(f"No docid strings found in JSON list: {p}")
            return sorted(list(set(vals)))
        raise ValueError(f"Unsupported JSON type for docid vocab: {type(data)}")

    vals = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            d = item.get("docid")
            if isinstance(d, str) and d.strip():
                vals.append(d.strip())
    if len(vals) == 0:
        raise ValueError(f"No docid strings found in JSONL: {p}")
    return sorted(list(set(vals)))


def build_encoder_text(sample: Dict, text_mode: str) -> str:
    code = str(sample.get("code", "")).strip()
    if text_mode == "code":
        return code
    query = str(sample.get("query", "")).strip()
    if text_mode == "query":
        return query
    return f"Query: {query}\nCode:\n{code}"


def build_eval_samples(item: Dict) -> Dict[str, List[str]]:
    return {
        "query": [str(item.get("query", ""))],
        "code": [str(item.get("code", ""))],
    }


def _normalize_query_for_grouping(query: str) -> str:
    q = str(query or "").strip().lower()
    q = re.sub(r"\s+", " ", q)
    return q


def compute_query_group_consistency_metrics(details: List[Dict], data: List[Dict]) -> Dict[str, float]:
    """
    Group by normalized query text and report:
      - QueryGroupConsistency@1: all predictions in group share the same top1
      - QueryGroupAllHit@1: all predictions in group hit gold at top1
      - QueryGroupPairAgree@1: pairwise top1 agreement ratio within groups
      - QueryGroupPairJointHit@1: pairwise joint-hit ratio within groups
    """
    if not details or not data:
        return {
            "QueryGroupTotal": 0,
            "QueryGroupMultiSize": 0,
            "QueryGroupConsistency@1": 0.0,
            "QueryGroupAllHit@1": 0.0,
            "QueryGroupPairAgree@1": 0.0,
            "QueryGroupPairJointHit@1": 0.0,
        }

    groups: Dict[str, List[int]] = {}
    for rec in details:
        idx = int(rec.get("index", -1))
        if idx < 0 or idx >= len(data):
            continue
        key = _normalize_query_for_grouping(data[idx].get("query", ""))
        groups.setdefault(key, []).append(idx)

    details_by_idx = {int(r.get("index", -1)): r for r in details}
    total_groups = 0
    multi_groups = 0
    consistent_groups = 0
    all_hit_groups = 0
    pair_total = 0
    pair_agree = 0
    pair_joint_hit = 0

    for _, idxs in groups.items():
        if len(idxs) == 0:
            continue
        total_groups += 1
        if len(idxs) < 2:
            continue
        multi_groups += 1

        top1_list = []
        hit1_list = []
        for idx in idxs:
            rec = details_by_idx.get(int(idx))
            if rec is None:
                continue
            preds = rec.get("generated_docids") or []
            top1 = preds[0] if len(preds) > 0 else None
            hit1 = bool((rec.get("hit_flags") or {}).get("hit@1", False))
            top1_list.append(top1)
            hit1_list.append(hit1)

        if len(top1_list) < 2:
            continue

        first = top1_list[0]
        if all(x == first for x in top1_list):
            consistent_groups += 1
        if all(hit1_list):
            all_hit_groups += 1

        n = len(top1_list)
        for i in range(n):
            for j in range(i + 1, n):
                pair_total += 1
                if top1_list[i] == top1_list[j]:
                    pair_agree += 1
                if hit1_list[i] and hit1_list[j]:
                    pair_joint_hit += 1

    denom_group = float(max(1, multi_groups))
    denom_pair = float(max(1, pair_total))
    return {
        "QueryGroupTotal": int(total_groups),
        "QueryGroupMultiSize": int(multi_groups),
        "QueryGroupConsistency@1": float(consistent_groups) / denom_group,
        "QueryGroupAllHit@1": float(all_hit_groups) / denom_group,
        "QueryGroupPairAgree@1": float(pair_agree) / denom_pair,
        "QueryGroupPairJointHit@1": float(pair_joint_hit) / denom_pair,
    }


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    x = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1e-6)
    return x / denom


class DenseLexicalCandidateRetriever:
    def __init__(
        self,
        model,
        emb_path: str,
        doc_index_to_docid_path: str,
        device: str,
        text_mode: str = "code",
        l2_normalize: bool = False,
    ):
        self.model = model
        self.device = device
        self.text_mode = text_mode
        self.l2_normalize = bool(l2_normalize)

        with open(doc_index_to_docid_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.docids = [raw[str(i)] for i in range(len(raw))]

        embs = np.load(emb_path).astype(np.float32)
        if len(embs) != len(self.docids):
            raise ValueError(f"doc embedding size mismatch: {len(embs)} vs {len(self.docids)}")
        self.doc_embs = torch.from_numpy(embs).to(device=self.device, dtype=torch.float32)
        if self.l2_normalize:
            self.doc_embs = torch.nn.functional.normalize(self.doc_embs, p=2, dim=-1)

    @torch.no_grad()
    def retrieve(self, sample: Dict, topk: int) -> List[Tuple[str, float]]:
        topk = max(1, min(int(topk), len(self.docids)))
        text = build_encoder_text(sample, self.text_mode)
        tok = self.model.encoder_tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=self.model.encoder_max_length,
            return_tensors="pt",
        ).to(self.device)
        last_hidden = self.model._encode_with_roberta(
            input_ids=tok.input_ids,
            attention_mask=tok.attention_mask,
        )
        q_emb = mean_pool(last_hidden, tok.attention_mask).to(dtype=torch.float32)
        if self.l2_normalize:
            q_emb = torch.nn.functional.normalize(q_emb, p=2, dim=-1)
        scores = torch.matmul(self.doc_embs, q_emb[0])
        top_scores, top_idx = torch.topk(scores, k=topk, largest=True, sorted=True)

        out = []
        for s, idx in zip(top_scores.tolist(), top_idx.tolist()):
            out.append((self.docids[int(idx)], float(s)))
        return out


def normalize_doc_scores(docid_score_pairs: List[Tuple[str, float]]) -> Dict[str, float]:
    if not docid_score_pairs:
        return {}

    values = np.asarray([float(s) for _, s in docid_score_pairs], dtype=np.float32)
    lo = float(values.min())
    hi = float(values.max())
    span = hi - lo
    out: Dict[str, float] = {}
    if span > 1e-8:
        for docid, score in docid_score_pairs:
            out[str(docid)] = float((float(score) - lo) / span)
    else:
        n = max(1, len(docid_score_pairs))
        for rank, (docid, _) in enumerate(docid_score_pairs):
            out[str(docid)] = float((n - rank) / n)
    return out


def build_dual_phase_guidance_infer_prefix_bonus_map(model, docid_to_score: Dict[str, float], pooling: str = "max") -> Dict[tuple, Dict[int, float]]:
    if pooling not in {"max", "mean"}:
        raise ValueError(f"Unsupported pooling: {pooling}")

    tok = model.decoder_tokenizer
    prefix_bonus: Dict[tuple, Dict[int, float]] = {}
    for raw_docid, score in docid_to_score.items():
        suffix = model._extract_docid_suffix(raw_docid)
        token_ids = tok(
            f"{suffix}</DOCID>",
            add_special_tokens=False,
            return_tensors=None,
        )["input_ids"]
        if len(token_ids) == 0:
            continue

        prefix = tuple()
        for tid in token_ids:
            tid = int(tid)
            bucket = prefix_bonus.setdefault(prefix, {})
            if pooling == "max":
                prev = bucket.get(tid, None)
                bucket[tid] = float(score) if prev is None else max(float(score), float(prev))
            else:
                # temporary sum; mean is finalized below.
                bucket[tid] = float(bucket.get(tid, 0.0)) + float(score)
            prefix = prefix + (tid,)

    if pooling == "mean":
        # Use count by replaying paths once, so mean is per (prefix, token) pair.
        counts: Dict[tuple, Dict[int, int]] = {}
        for raw_docid, _ in docid_to_score.items():
            suffix = model._extract_docid_suffix(raw_docid)
            token_ids = tok(
                f"{suffix}</DOCID>",
                add_special_tokens=False,
                return_tensors=None,
            )["input_ids"]
            prefix = tuple()
            for tid in token_ids:
                tid = int(tid)
                c = counts.setdefault(prefix, {})
                c[tid] = int(c.get(tid, 0)) + 1
                prefix = prefix + (tid,)
        for prefix, token_to_score in prefix_bonus.items():
            cnt_map = counts.get(prefix, {})
            for tid in list(token_to_score.keys()):
                cnt = max(1, int(cnt_map.get(tid, 1)))
                token_to_score[tid] = float(token_to_score[tid]) / float(cnt)

    return prefix_bonus


# =========================
# 2. DocID normalization (prefix-agnostic)
#   - XXXXXXXX
#   - a1b2c3d4
# =========================
def normalize_docid(pred: str) -> str:
    if pred is None:
        return pred
    pred = str(pred).strip().lower()
    compact = re.sub(r"\s+", "", pred)
    compact = compact.replace("<docid>", "").replace("</docid>", "")

    # 1) Hierarchical format (preferred): a1b2c3d4 (any prefix allowed)
    m = re.search(r"((?:[a-z]\d+){2,})", compact)
    if m:
        groups = re.findall(r"([a-z])(\d+)", m.group(1))
        if len(groups) > 0:
            return "".join(f"{k}{int(v)}" for k, v in groups)

    # 2) Legacy 8-digit numeric format
    m2 = re.search(r"(\d{8})", compact)
    if m2:
        return m2.group(1)

    return compact


def is_docid_structurally_valid(docid: str) -> bool:
    if docid is None:
        return False
    return re.fullmatch(r"a\d{1,3}b\d{1,3}c\d{1,3}d\d{1,3}", str(docid)) is not None


def extract_docid_hints(docid: str):
    s = str(docid or "").strip().lower()
    hints = []

    m = re.search(r"a(\d+)b(\d+)c(\d+)d(\d+)", s)
    if m:
        a, b, c, d = m.groups()
        a = str(int(a))
        b = str(int(b))
        c = str(int(c))
        d = str(int(d))
        if len(d) > 0:
            hints.append(f"a{a}b{b}c{c}d{d[:3]}")
        hints.append(f"a{a}b{b}c{c}")
        hints.append(f"a{a}b{b}")
        hints.append(f"a{a}")
        return hints

    m = re.search(r"a(\d+)b(\d+)c(\d+)", s)
    if m:
        a, b, c = m.groups()
        a = str(int(a))
        b = str(int(b))
        c = str(int(c))
        hints.append(f"a{a}b{b}c{c}")
        hints.append(f"a{a}b{b}")
        hints.append(f"a{a}")
        return hints

    m = re.search(r"a(\d+)b(\d+)", s)
    if m:
        a, b = m.groups()
        hints.append(f"a{int(a)}b{int(b)}")
        hints.append(f"a{int(a)}")
        return hints

    m = re.search(r"a(\d+)", s)
    if m:
        hints.append(f"a{int(m.group(1))}")
    return hints


def recover_legal_docids_from_hints(hints, legal_docid_list, topk):
    if not hints or not legal_docid_list:
        return []
    out = []
    seen = set()
    for h in hints:
        for d in legal_docid_list:
            if d.startswith(h) and d not in seen:
                out.append(d)
                seen.add(d)
                if len(out) >= int(topk):
                    return out
    return out


def build_docid_bucket_by_a(legal_docid_list):
    buckets = {}
    for d in legal_docid_list or []:
        m = re.match(r"a(\d+)b", str(d))
        if not m:
            continue
        key = f"a{int(m.group(1))}"
        buckets.setdefault(key, []).append(d)
    return buckets


def common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    if len(a) > len(b):
        a, b = b, a

    prev = list(range(len(a) + 1))
    for j in range(1, len(b) + 1):
        bj = b[j - 1]
        cur = [j]
        for i in range(1, len(a) + 1):
            cost = 0 if a[i - 1] == bj else 1
            cur.append(min(
                prev[i] + 1,
                cur[i - 1] + 1,
                prev[i - 1] + cost,
            ))
        prev = cur
    return prev[-1]


def recover_legal_docids_by_nearest(
    bad_docids,
    legal_docid_list,
    topk,
    legal_docid_bucket_by_a=None,
):
    bad = []
    seen_bad = set()
    for x in bad_docids or []:
        s = str(x or "").strip()
        if not s or s in seen_bad:
            continue
        seen_bad.add(s)
        bad.append(s)
    if len(bad) == 0 or not legal_docid_list:
        return []

    # Narrow to the same "a" bucket first to avoid scanning the full vocabulary.
    pool = []
    pool_seen = set()
    if legal_docid_bucket_by_a:
        for s in bad:
            m = re.search(r"a(\d+)", s)
            if not m:
                continue
            key = f"a{int(m.group(1))}"
            for d in legal_docid_bucket_by_a.get(key, []):
                if d in pool_seen:
                    continue
                pool_seen.add(d)
                pool.append(d)
    if len(pool) == 0:
        pool = legal_docid_list

    refs = bad[:12]
    scored = []
    for d in pool:
        best_dist = None
        best_pref = 0
        for r in refs:
            dist = levenshtein_distance(r, d)
            if best_dist is None or dist < best_dist:
                best_dist = dist
            pref = common_prefix_len(r, d)
            if pref > best_pref:
                best_pref = pref
        scored.append((int(best_dist), -int(best_pref), len(d), d))

    scored.sort()
    out = []
    seen = set()
    for _, _, _, d in scored:
        if d in seen:
            continue
        seen.add(d)
        out.append(d)
        if len(out) >= int(topk):
            break
    return out


# =========================
# 3. DocID generation (minimal: eos + max_new_tokens)
# =========================
@torch.no_grad()
def generate_docids(
    model,
    samples,
    num_beams=20,
    num_return_sequences=10,
    oversample_factor=5,
    constrained_trie=None,
    legal_docid_set=None,
    legal_docid_list=None,
    legal_docid_bucket_by_a=None,
    dual_phase_guidance_infer_candidates: Optional[List[Tuple[str, float]]] = None,
    dual_phase_guidance_infer_use_query_trie: bool = True,
    dual_phase_guidance_infer_lexical_bonus_weight: float = 0.0,
    dual_phase_guidance_infer_lexical_pooling: str = "max",
    dual_phase_guidance_infer_uniform_bonus: bool = False,
    dual_phase_guidance_infer_debug_topn: int = 3,
    debug_log_path=None,
    sample_index=None,
):
    """
    samples = {"query":[str], "code":[str]}
    return: List[str]  # normalized docids
    """
    end_id = model.decoder_tokenizer.convert_tokens_to_ids("</DOCID>")
    if end_id is None or end_id < 0:
        end_id = model.decoder_tokenizer.eos_token_id

    target_k = int(num_return_sequences)
    dropped_norms = []
    active_constrained_trie = constrained_trie
    active_constrained_docids = None
    lexical_prefix_bonus = None

    if dual_phase_guidance_infer_candidates:
        dedup_pairs: List[Tuple[str, float]] = []
        seen_doc = set()
        for docid, score in dual_phase_guidance_infer_candidates:
            d = normalize_docid(docid)
            if not d or d in seen_doc:
                continue
            seen_doc.add(d)
            dedup_pairs.append((d, float(score)))
        if bool(dual_phase_guidance_infer_uniform_bonus):
            docid_to_norm_score = {str(docid): 1.0 for docid, _ in dedup_pairs}
        else:
            docid_to_norm_score = normalize_doc_scores(dedup_pairs)
        if len(docid_to_norm_score) > 0:
            lexical_prefix_bonus = build_dual_phase_guidance_infer_prefix_bonus_map(
                model,
                docid_to_score=docid_to_norm_score,
                pooling=str(dual_phase_guidance_infer_lexical_pooling),
            )
            if dual_phase_guidance_infer_use_query_trie:
                active_constrained_trie = None
                active_constrained_docids = list(docid_to_norm_score.keys())
            if int(dual_phase_guidance_infer_debug_topn) > 0:
                head = dedup_pairs[: int(dual_phase_guidance_infer_debug_topn)]
                _append_debug_log(
                    debug_log_path,
                    f"[DBG][dual_phase_guidance_infer_candidates] index={sample_index} top={head}",
                )

    def _decode_unique(outputs):
        preds = []
        seen = set()
        for seq in outputs:
            text = model.decoder_tokenizer.decode(
                seq,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            p = normalize_docid(text)
            if not is_docid_structurally_valid(p):
                _append_debug_log(
                    debug_log_path,
                    f"[DBG][DROP_PATTERN] index={sample_index} raw={repr(text)} norm={repr(p)}",
                )
                dropped_norms.append(p)
                continue
            if legal_docid_set is not None and p not in legal_docid_set:
                _append_debug_log(
                    debug_log_path,
                    f"[DBG][DROP_OOV] index={sample_index} raw={repr(text)} norm={repr(p)}",
                )
                dropped_norms.append(p)
                continue
            if p in seen:
                continue
            seen.add(p)
            preds.append(p)
        return preds

    # A) Preserve original beam-head order first (stabilizes Hit@1/5).
    head_beams = max(int(num_beams), target_k)
    head_outputs = model.generate(
        samples,
        max_new_tokens=24,
        num_beams=head_beams,
        num_return_sequences=target_k,
        early_stopping=True,
        eos_token_id=int(end_id),
        pad_token_id=int(model.decoder_tokenizer.pad_token_id),
        use_cache=True,
        min_new_tokens=4,
        constrained_trie=active_constrained_trie,
        constrained_docids=active_constrained_docids,
        lexical_prefix_bonus=lexical_prefix_bonus,
        lexical_bonus_weight=float(dual_phase_guidance_infer_lexical_bonus_weight),
    )
    head_preds = _decode_unique(head_outputs)

    # B) Oversample only for tail fill after dedup (mainly improves Hit@10).
    raw_k = max(target_k, target_k * int(oversample_factor))
    raw_beams = max(int(num_beams), raw_k)
    if raw_k > target_k or raw_beams > head_beams:
        tail_outputs = model.generate(
            samples,
            max_new_tokens=24,
            num_beams=raw_beams,
            num_return_sequences=raw_k,
            early_stopping=True,
            eos_token_id=int(end_id),
            pad_token_id=int(model.decoder_tokenizer.pad_token_id),
            use_cache=True,
            min_new_tokens=4,
            constrained_trie=active_constrained_trie,
            constrained_docids=active_constrained_docids,
            lexical_prefix_bonus=lexical_prefix_bonus,
            lexical_bonus_weight=float(dual_phase_guidance_infer_lexical_bonus_weight),
        )
        tail_preds = _decode_unique(tail_outputs)
        seen = set(head_preds)
        for p in tail_preds:
            if p in seen:
                continue
            head_preds.append(p)
            seen.add(p)
            if len(head_preds) >= target_k:
                break

    if legal_docid_list is not None and len(legal_docid_list) > 0 and len(head_preds) < target_k:
        seen = set(head_preds)
        hints = []
        hint_seen = set()
        for bad in dropped_norms:
            for h in extract_docid_hints(bad):
                if h not in hint_seen:
                    hint_seen.add(h)
                    hints.append(h)

        recovered = recover_legal_docids_from_hints(
            hints=hints,
            legal_docid_list=legal_docid_list,
            topk=target_k,
        )
        added = 0
        for p in recovered:
            if p in seen:
                continue
            head_preds.append(p)
            seen.add(p)
            added += 1
            if len(head_preds) >= target_k:
                break
        if added > 0:
            _append_debug_log(
                debug_log_path,
                f"[DBG][FILL_HINT] index={sample_index} added={added} hints={hints[:6]}",
            )

        if len(head_preds) < target_k:
            nearest = recover_legal_docids_by_nearest(
                bad_docids=dropped_norms,
                legal_docid_list=legal_docid_list,
                topk=target_k,
                legal_docid_bucket_by_a=legal_docid_bucket_by_a,
            )
            nearest_added = 0
            for p in nearest:
                if p in seen:
                    continue
                head_preds.append(p)
                seen.add(p)
                nearest_added += 1
                if len(head_preds) >= target_k:
                    break
            if nearest_added > 0:
                _append_debug_log(
                    debug_log_path,
                    f"[DBG][FILL_NEAREST] index={sample_index} added={nearest_added} refs={dropped_norms[:4]}",
                )

        if len(head_preds) < target_k:
            fallback_added = 0
            for p in legal_docid_list:
                if p in seen:
                    continue
                head_preds.append(p)
                seen.add(p)
                fallback_added += 1
                if len(head_preds) >= target_k:
                    break
            if fallback_added > 0:
                _append_debug_log(
                    debug_log_path,
                    f"[DBG][FILL_GLOBAL] index={sample_index} added={fallback_added}",
                )

    return head_preds[:target_k]


# =========================
# 4. Hit@K + MRR@20
# =========================
def evaluate_cosqa(
    model,
    data,
    device="cuda",
    ks=(1, 3, 5, 10),
    mrr_k=20,
    save_topk=10,
    num_beams=20,
    return_details=False,
    constrained_trie=None,
    legal_docid_set=None,
    legal_docid_list=None,
    legal_docid_bucket_by_a=None,
    dual_phase_guidance_infer_enable=False,
    dual_phase_guidance_infer_candidate_retriever: Optional[DenseLexicalCandidateRetriever] = None,
    dual_phase_guidance_infer_lexical_topk=1000,
    dual_phase_guidance_infer_restrict_to_legal_vocab=True,
    dual_phase_guidance_infer_use_query_trie=True,
    dual_phase_guidance_infer_lexical_bonus_weight=0.0,
    dual_phase_guidance_infer_lexical_pooling="max",
    dual_phase_guidance_infer_uniform_bonus=False,
    dual_phase_guidance_infer_debug_topn=3,
    debug_log_path=None,
):
    model.to(device)
    model.eval()

    hit_count = {k: 0 for k in ks}
    reciprocal_rank_sum = 0.0
    total = len(data)
    gen_k = max(int(max(ks)), int(save_topk), int(num_beams), int(mrr_k))
    details = [] if return_details else None

    for idx, item in enumerate(tqdm(data, desc="Evaluating")):
        samples = build_eval_samples(item)
        gold = normalize_docid(item["docid"])
        dual_phase_guidance_infer_candidates = None
        if bool(dual_phase_guidance_infer_enable) and dual_phase_guidance_infer_candidate_retriever is not None:
            raw_cands = dual_phase_guidance_infer_candidate_retriever.retrieve(item, topk=int(dual_phase_guidance_infer_lexical_topk))
            if bool(dual_phase_guidance_infer_restrict_to_legal_vocab) and legal_docid_set is not None:
                filtered = []
                for d, s in raw_cands:
                    nd = normalize_docid(d)
                    if nd in legal_docid_set:
                        filtered.append((nd, s))
                dual_phase_guidance_infer_candidates = filtered
            else:
                dual_phase_guidance_infer_candidates = [(normalize_docid(d), s) for d, s in raw_cands]

        preds = generate_docids(
            model,
            samples,
            num_beams=gen_k,
            num_return_sequences=gen_k,
            oversample_factor=5,
            constrained_trie=constrained_trie,
            legal_docid_set=legal_docid_set,
            legal_docid_list=legal_docid_list,
            legal_docid_bucket_by_a=legal_docid_bucket_by_a,
            dual_phase_guidance_infer_candidates=dual_phase_guidance_infer_candidates,
            dual_phase_guidance_infer_use_query_trie=bool(dual_phase_guidance_infer_use_query_trie),
            dual_phase_guidance_infer_lexical_bonus_weight=float(dual_phase_guidance_infer_lexical_bonus_weight),
            dual_phase_guidance_infer_lexical_pooling=str(dual_phase_guidance_infer_lexical_pooling),
            dual_phase_guidance_infer_uniform_bonus=bool(dual_phase_guidance_infer_uniform_bonus),
            dual_phase_guidance_infer_debug_topn=int(dual_phase_guidance_infer_debug_topn),
            debug_log_path=debug_log_path,
            sample_index=idx,
        )
        if len(preds) == 0:
            _append_debug_log(
                debug_log_path,
                f"[DBG][sample_empty] index={idx} gold={repr(gold)}",
            )
        if legal_docid_set is not None:
            oov_preds = [p for p in preds if p not in legal_docid_set]
            if len(oov_preds) > 0:
                _append_debug_log(
                    debug_log_path,
                    f"[DBG][sample_oov] index={idx} gold={repr(gold)} oov_preds={oov_preds}",
                )

        hit_flags = {}
        for k in ks:
            hit = gold in preds[:k]
            hit_flags[k] = bool(hit)
            if hit:
                hit_count[k] += 1

        rank_at_mrr_k = -1
        for rank_i, pred_docid in enumerate(preds[: int(mrr_k)], start=1):
            if pred_docid == gold:
                rank_at_mrr_k = rank_i
                break
        rr_at_mrr_k = 0.0 if rank_at_mrr_k < 1 else (1.0 / float(rank_at_mrr_k))
        reciprocal_rank_sum += rr_at_mrr_k

        if return_details:
            details.append(
                {
                    "index": idx,
                    "gold_docid": gold,
                    "generated_docids": preds[: int(save_topk)],
                    "dual_phase_guidance_infer_candidate_size": 0 if dual_phase_guidance_infer_candidates is None else len(dual_phase_guidance_infer_candidates),
                    "hit_flags": {f"hit@{k}": hit_flags[k] for k in ks},
                    f"rank@{int(mrr_k)}": None if rank_at_mrr_k < 1 else rank_at_mrr_k,
                    f"rr@{int(mrr_k)}": rr_at_mrr_k,
                }
            )

    results = {}
    for k in ks:
        results[f"Hit@{k}"] = (hit_count[k] / total) if total > 0 else 0.0
    results[f"MRR@{int(mrr_k)}"] = (reciprocal_rank_sum / total) if total > 0 else 0.0

    results["Total"] = total
    if return_details:
        group_metrics = compute_query_group_consistency_metrics(details, data)
        results.update(group_metrics)
    if return_details:
        return results, details
    return results


# =========================
# 5. Main (manual flow)
# =========================
def main():
    args = parse_args()
    if int(args.save_topk) <= 0:
        raise ValueError("--save-topk must be > 0")
    if int(args.num_beams) <= 0:
        raise ValueError("--num-beams must be > 0")
    _init_debug_log(args.debug_log_path, args)
    _append_debug_log(args.debug_log_path, f"[DBG][env] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch sees device_count =", torch.cuda.device_count())
    print("device name =", torch.cuda.get_device_name(0))

    ckpt_paths = resolve_ckpt_paths(args.ckpt_path, args.ckpt_mode, args.run_dir)
    print("\nResolved checkpoints:")
    for p in ckpt_paths:
        print(" -", p)

    # ---------
    # 5.3 Load VALID split (aligned with current training data format)
    # ---------
    valid_path = str(args.valid_jsonl_path)
    data = load_cosqa_jsonl(
        valid_path,
        max_samples=None,   # Use full set by default; use 200/1000 for quick checks.
    )

    export_payload = {
        "dataset_path": valid_path,
        "total": len(data),
        "save_topk": int(args.save_topk),
        "num_beams": int(args.num_beams),
        "constrained_decode": bool(args.constrained_decode),
        "docid_vocab_path": args.docid_vocab_path,
        "debug_log_path": args.debug_log_path,
        "checkpoints": [],
    }
    export_payload["dual_phase_guidance_infer"] = {
        "enabled": bool(args.dual_phase_guidance_infer_enable),
        "lexical_topk": int(args.dual_phase_guidance_infer_lexical_topk),
        "lexical_source": args.dual_phase_guidance_infer_lexical_source,
        "lexical_bonus_weight": float(args.dual_phase_guidance_infer_lexical_bonus_weight),
        "lexical_pooling": args.dual_phase_guidance_infer_lexical_pooling,
        "uniform_bonus": bool(args.dual_phase_guidance_infer_uniform_bonus),
        "doc_emb_path": args.dual_phase_guidance_infer_doc_emb_path,
        "doc_index_to_docid_path": args.dual_phase_guidance_infer_doc_index_to_docid_path,
        "restrict_to_legal_vocab": bool(args.dual_phase_guidance_infer_restrict_to_legal_vocab),
        "use_query_trie": bool(args.dual_phase_guidance_infer_use_query_trie),
    }
    export_payload["eval_model_overrides"] = {
        "use_patch_tokens": args.use_patch_tokens,
        "patch_token_count": args.patch_token_count,
        "patch_pooling": args.patch_pooling,
    }
    legal_docids = None
    legal_docid_set = None
    legal_docid_list = None
    legal_docid_bucket_by_a = None
    if args.constrained_decode:
        legal_docids = load_docid_vocab(args.docid_vocab_path)
        legal_docid_set = set(normalize_docid(x) for x in legal_docids)
        legal_docid_list = sorted(list(legal_docid_set))
        legal_docid_bucket_by_a = build_docid_bucket_by_a(legal_docid_list)
        export_payload["docid_vocab_size"] = len(legal_docids)
        print(f"Loaded legal docid vocab: {len(legal_docids)} from {args.docid_vocab_path}")
        _append_debug_log(
            args.debug_log_path,
            f"[DBG][vocab] path={args.docid_vocab_path} size={len(legal_docids)} normalized_size={len(legal_docid_set)}",
        )
        _append_debug_log(args.debug_log_path, f"[DBG][vocab] head={legal_docids[:5]}")
        _append_debug_log(args.debug_log_path, f"[DBG][vocab] tail={legal_docids[-5:]}")

    summary = []
    for ckpt_path in ckpt_paths:
        _append_debug_log(args.debug_log_path, f"[DBG][ckpt] ===== {ckpt_path} =====")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        ckpt_model_cfg = _extract_model_cfg_from_ckpt(ckpt)
        model_cfg = apply_eval_model_overrides(ckpt_model_cfg, args)
        model = build_model(model_cfg=model_cfg)
        if hasattr(model, "set_debug_log_path"):
            model.set_debug_log_path(args.debug_log_path)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        print("\nLoaded checkpoint:", ckpt_path)
        if len(ckpt_model_cfg) > 0:
            print(
                "Model cfg from ckpt:",
                {
                    "encoder_name": ckpt_model_cfg.get("encoder_name"),
                    "decoder_name": ckpt_model_cfg.get("decoder_name"),
                    "use_patch_tokens": ckpt_model_cfg.get("use_patch_tokens"),
                    "patch_token_count": ckpt_model_cfg.get("patch_token_count"),
                    "patch_pooling": ckpt_model_cfg.get("patch_pooling"),
                    "enable_alignment_proj": ckpt_model_cfg.get("enable_alignment_proj"),
                    "enable_patch_linear": ckpt_model_cfg.get("enable_patch_linear"),
                },
            )
        else:
            print("Model cfg from ckpt: missing -> fallback defaults were used.")
        print(
            "Model cfg for eval:",
            {
                "encoder_name": model_cfg.get("encoder_name"),
                "decoder_name": model_cfg.get("decoder_name"),
                "use_patch_tokens": model_cfg.get("use_patch_tokens"),
                "patch_token_count": model_cfg.get("patch_token_count"),
                "patch_pooling": model_cfg.get("patch_pooling"),
                "enable_alignment_proj": model_cfg.get("enable_alignment_proj"),
                "enable_patch_linear": model_cfg.get("enable_patch_linear"),
            },
        )
        print("Missing keys count:", len(missing))
        print("Unexpected keys count:", len(unexpected))

        model.to(device)
        model.eval()
        fix_infer_dtypes(model)
        dual_phase_guidance_infer_retriever = None
        if bool(args.dual_phase_guidance_infer_enable):
            dual_phase_guidance_infer_retriever = DenseLexicalCandidateRetriever(
                model=model,
                emb_path=args.dual_phase_guidance_infer_doc_emb_path,
                doc_index_to_docid_path=args.dual_phase_guidance_infer_doc_index_to_docid_path,
                device=device,
                text_mode=args.dual_phase_guidance_infer_lexical_source,
                l2_normalize=False,
            )
            print(
                f"dual-phase guidance infer lexical retriever enabled: topk={int(args.dual_phase_guidance_infer_lexical_topk)} "
                f"source={args.dual_phase_guidance_infer_lexical_source} bonus_w={float(args.dual_phase_guidance_infer_lexical_bonus_weight)}"
            )
        constrained_trie = None
        if args.constrained_decode:
            constrained_trie = model.build_docid_token_trie(legal_docids)
            print("Constrained decode trie enabled for this checkpoint.")

        item = data[0]
        samples = build_eval_samples(item)
        sanity_k = max(10, int(args.save_topk), int(args.num_beams))
        sanity_dual_phase_guidance_infer_candidates = None
        if dual_phase_guidance_infer_retriever is not None:
            sanity_dual_phase_guidance_infer_candidates = dual_phase_guidance_infer_retriever.retrieve(item, topk=int(args.dual_phase_guidance_infer_lexical_topk))
            if bool(args.dual_phase_guidance_infer_restrict_to_legal_vocab) and legal_docid_set is not None:
                sanity_dual_phase_guidance_infer_candidates = [
                    (normalize_docid(d), s)
                    for d, s in sanity_dual_phase_guidance_infer_candidates
                    if normalize_docid(d) in legal_docid_set
                ]
        preds = generate_docids(
            model,
            samples,
            num_beams=sanity_k,
            num_return_sequences=sanity_k,
            oversample_factor=5,
            constrained_trie=constrained_trie,
            legal_docid_set=legal_docid_set,
            legal_docid_list=legal_docid_list,
            legal_docid_bucket_by_a=legal_docid_bucket_by_a,
            dual_phase_guidance_infer_candidates=sanity_dual_phase_guidance_infer_candidates,
            dual_phase_guidance_infer_use_query_trie=bool(args.dual_phase_guidance_infer_use_query_trie),
            dual_phase_guidance_infer_lexical_bonus_weight=float(args.dual_phase_guidance_infer_lexical_bonus_weight),
            dual_phase_guidance_infer_lexical_pooling=args.dual_phase_guidance_infer_lexical_pooling,
            dual_phase_guidance_infer_uniform_bonus=bool(args.dual_phase_guidance_infer_uniform_bonus),
            dual_phase_guidance_infer_debug_topn=int(args.dual_phase_guidance_infer_debug_topn),
            debug_log_path=args.debug_log_path,
            sample_index="sanity",
        )

        print("\n===== SANITY CHECK (VALID) =====")
        print("Gold docid:", item["docid"])
        print("Generated docids:")
        for p in preds:
            print(f"  [{p}]")
        print("================================\n")

        eval_ks = (1, 3, 5, 10)
        eval_mrr_k = 20
        results, sample_details = evaluate_cosqa(
            model,
            data,
            device=device,
            ks=eval_ks,
            mrr_k=eval_mrr_k,
            save_topk=int(args.save_topk),
            num_beams=int(args.num_beams),
            return_details=True,
            constrained_trie=constrained_trie,
            legal_docid_set=legal_docid_set,
            legal_docid_list=legal_docid_list,
            legal_docid_bucket_by_a=legal_docid_bucket_by_a,
            dual_phase_guidance_infer_enable=bool(args.dual_phase_guidance_infer_enable),
            dual_phase_guidance_infer_candidate_retriever=dual_phase_guidance_infer_retriever,
            dual_phase_guidance_infer_lexical_topk=int(args.dual_phase_guidance_infer_lexical_topk),
            dual_phase_guidance_infer_restrict_to_legal_vocab=bool(args.dual_phase_guidance_infer_restrict_to_legal_vocab),
            dual_phase_guidance_infer_use_query_trie=bool(args.dual_phase_guidance_infer_use_query_trie),
            dual_phase_guidance_infer_lexical_bonus_weight=float(args.dual_phase_guidance_infer_lexical_bonus_weight),
            dual_phase_guidance_infer_lexical_pooling=args.dual_phase_guidance_infer_lexical_pooling,
            dual_phase_guidance_infer_uniform_bonus=bool(args.dual_phase_guidance_infer_uniform_bonus),
            dual_phase_guidance_infer_debug_topn=int(args.dual_phase_guidance_infer_debug_topn),
            debug_log_path=args.debug_log_path,
        )
        summary.append((ckpt_path, results))
        export_payload["checkpoints"].append(
            {
                "checkpoint": ckpt_path,
                "metrics": results,
                "sanity_check": {
                    "gold_docid": item["docid"],
                    "generated_docids": preds[: int(args.save_topk)],
                },
                "samples": sample_details,
            }
        )

        print("\n===== Evaluation Results (VALID) =====")
        print("Checkpoint:", ckpt_path)
        print(f"Total: {results['Total']}")
        for k in eval_ks:
            print(f"Hit@{k}: {results[f'Hit@{k}']:.4f}")
        print(f"MRR@{eval_mrr_k}: {results[f'MRR@{eval_mrr_k}']:.4f}")
        if int(results.get("QueryGroupMultiSize", 0)) > 0:
            print(
                "Query-group consistency:"
                f" groups={int(results['QueryGroupMultiSize'])}"
                f" consistency@1={results['QueryGroupConsistency@1']:.4f}"
                f" all_hit@1={results['QueryGroupAllHit@1']:.4f}"
                f" pair_agree@1={results['QueryGroupPairAgree@1']:.4f}"
                f" pair_joint_hit@1={results['QueryGroupPairJointHit@1']:.4f}"
            )

    if len(summary) > 1:
        print("\n===== Summary (sorted by MRR@20 desc) =====")
        ranked = sorted(summary, key=lambda x: x[1]["MRR@20"], reverse=True)
        for path, r in ranked:
            print(
                f"{path} | Hit@1={r['Hit@1']:.4f} "
                f"Hit@3={r['Hit@3']:.4f} Hit@5={r['Hit@5']:.4f} Hit@10={r['Hit@10']:.4f} "
                f"MRR@20={r['MRR@20']:.4f}"
            )

    default_json_path = os.path.join(
        os.path.dirname(ckpt_paths[0]),
        "eval_valid_predictions.json",
    )
    save_json_path = args.save_json if args.save_json else default_json_path
    save_dir = os.path.dirname(save_json_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    with open(save_json_path, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, ensure_ascii=False, indent=2)
    print(f"\nSaved prediction details to: {save_json_path}")


if __name__ == "__main__":
    main()
