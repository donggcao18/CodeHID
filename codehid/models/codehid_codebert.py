import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import types
from typing import Dict, Iterable, List, Optional
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, RobertaTokenizer
try:
    from transformers import LogitsProcessor, LogitsProcessorList
except Exception:
    from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

from codehid.common.registry import registry
from codehid.models.base_model import BaseModel
from codehid.models.codebert_encoder import CodeBERTEncoder


def _as_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(v)


class BatchPrefixLexicalLogitsProcessor(LogitsProcessor):
    """
    Stage-II inference-time temporary lexical rescoring:
    add lexical bonus to next-token logits conditioned on the currently generated prefix.
    """

    def __init__(
        self,
        batch_prefix_bonus: List[Dict[tuple, Dict[int, float]]],
        prefix_total_len: int,
        num_beams: int,
        bonus_weight: float = 1.0,
    ):
        self.batch_prefix_bonus = batch_prefix_bonus
        self.prefix_total_len = int(prefix_total_len)
        self.num_beams = max(1, int(num_beams))
        self.bonus_weight = float(bonus_weight)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.bonus_weight == 0.0:
            return scores
        if not self.batch_prefix_bonus:
            return scores

        n_rows = scores.size(0)
        out = scores
        for row_id in range(n_rows):
            batch_id = min(row_id // self.num_beams, len(self.batch_prefix_bonus) - 1)
            prefix_bonus = self.batch_prefix_bonus[batch_id]
            if not prefix_bonus:
                continue

            gen_prefix = tuple(input_ids[row_id, self.prefix_total_len :].tolist())
            token_to_bonus = prefix_bonus.get(gen_prefix)
            if not token_to_bonus:
                continue

            token_ids = [int(tid) for tid in token_to_bonus.keys()]
            bonus_vals = torch.tensor(
                [float(token_to_bonus[tid]) for tid in token_ids],
                device=out.device,
                dtype=out.dtype,
            )
            out[row_id, token_ids] += bonus_vals * self.bonus_weight

        return out


@registry.register_model("codehid_codebert_dsi")
class CodeHIDCodeBERTDSI(BaseModel):
    """
    CodeBERT -> token-level alignment -> CausalLM decoder -> Generative Retrieval (DSI)

    Structured DOCID supervision:
        <DOCID> cosqa_a1b2c3d4 </DOCID>  (transitional)

    Generation-side fixes:
      (1) use </DOCID> only for eos_token_id (single int)
      (2) allowed token set covers tokenizer suffix pieces
      (3) set min_new_tokens by hierarchical suffix length

    Key fix for collapsed/unchanged outputs:
      - do not call CodeBERTEncoder.forward directly (may return prefix embedding only)
      - call the internal RoBERTa backbone and use input-dependent last_hidden_state
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "codehid_codebert_dsi": "codehid/configs/models/codehid_codebert.yaml",
    }

    def __init__(
        self,
        encoder_name: str,
        decoder_name: str,
        hidden_size: int,
        use_llm: bool = True,
        freeze_encoder: bool = True,
        freeze_decoder: bool = True,
        encoder_max_length: int = 512,
        decoder_max_length: int = 64,
        decoder_dtype: str = "float16",
        enable_gradient_checkpointing: bool = True,
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        use_patch_tokens: bool = False,
        patch_token_count: int = 4,
        patch_pooling: str = "mean",
        prefix_length: int = 10,
        docid_prefix: str = "",
        enable_alignment_proj: bool = True,
        enable_patch_linear: bool = True,
        # Option A: train only embedding and lm_head.
        unfreeze_decoder_embed_and_lm_head: bool = False,
        # Gradual unfreezing: release trainable params epoch by epoch.
        gradual_unfreeze: bool = False,
        unfreeze_prefix_epoch: int = 0,
        unfreeze_alignment_epoch: int = 1,
        unfreeze_embed_epoch: int = 3,
        unfreeze_lm_head_epoch: int = 4,
        label_smoothing: float = 0.0,
        dual_phase_guidance_train_enable: bool = False,
        dual_phase_guidance_train_lambda_lex: float = 0.0,
        dual_phase_guidance_train_lambda_rank_distill: float = 0.0,
        dual_phase_guidance_train_lex_temperature: float = 0.07,
        dual_phase_guidance_train_rank_temperature: float = 1.0,
        dual_phase_guidance_train_encode_batch_size: int = 1,
    ):
        super().__init__()

        # Fixed prefix length.
        self.prefix_length = int(prefix_length)

        # =============================== Encoder (CodeBERT) ===============================
        self.encoder = CodeBERTEncoder(
            model_name=encoder_name,
            freeze=freeze_encoder,
            prefix_length=prefix_length,
        )
        self.encoder_tokenizer = RobertaTokenizer.from_pretrained(encoder_name)
        self.encoder_max_length = int(encoder_max_length)

        # =============================== Decoder (CausalLM) ===============================
        self.decoder_max_length = int(decoder_max_length)

        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        torch_dtype = dtype_map.get(str(decoder_dtype).lower(), torch.float16)

        self.use_llm = bool(use_llm)
        if self.use_llm:
            self.decoder = AutoModelForCausalLM.from_pretrained(
                decoder_name,
                torch_dtype=torch_dtype,
            )
        else:
            # Build decoder from architecture config only (random init),
            # so "use_llm" can be controlled via YAML without code changes.
            decoder_cfg = AutoConfig.from_pretrained(decoder_name)
            self.decoder = AutoModelForCausalLM.from_config(decoder_cfg)
            if torch_dtype != torch.float32:
                self.decoder = self.decoder.to(dtype=torch_dtype)

        self.decoder_tokenizer = AutoTokenizer.from_pretrained(
            decoder_name, use_fast=False
        )
        if self.decoder_tokenizer.pad_token is None:
            self.decoder_tokenizer.pad_token = self.decoder_tokenizer.eos_token

        # ===== Register DOCID special tokens (CRITICAL) =====
        special_tokens = {"additional_special_tokens": ["<DOCID>", "</DOCID>"]}
        num_added = self.decoder_tokenizer.add_special_tokens(special_tokens)
        if num_added > 0:
            self.decoder.resize_token_embeddings(len(self.decoder_tokenizer))

        # Training-time memory controls
        try:
            self.decoder.config.use_cache = False
        except Exception:
            pass

        if enable_gradient_checkpointing:
            try:
                self.decoder.gradient_checkpointing_enable()
            except Exception:
                try:
                    self.decoder.model.gradient_checkpointing_enable()
                except Exception:
                    pass

        # =============================== Freeze decoder by default ===============================
        if freeze_decoder:
            for p in self.decoder.parameters():
                p.requires_grad = False

        # =============================== LoRA on decoder ===============================
        # Default strategy for faster training:
        # freeze decoder backbone, train only LoRA adapters (+ lightweight heads).
        self.use_lora = bool(use_lora)
        if lora_target_modules is None:
            lora_target_modules = ["q_proj", "v_proj"]

        if self.use_lora:
            lora_cfg = LoraConfig(
                r=int(lora_r),
                lora_alpha=int(lora_alpha),
                target_modules=list(lora_target_modules),
                lora_dropout=float(lora_dropout),
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.decoder = get_peft_model(self.decoder, lora_cfg)

        # =============================== Option A: unfreeze embed + lm_head and cast to FP32 ===============================
        # Prevents GradScaler.step() from raising "Attempting to unscale FP16 gradients."
        if unfreeze_decoder_embed_and_lm_head:
            for n, p in self.decoder.named_parameters():
                if "lora_" in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False

            emb = self.decoder.get_input_embeddings()
            emb.to(dtype=torch.float32)
            for p in emb.parameters():
                p.requires_grad = True

            head = self.decoder.get_output_embeddings()
            head.to(dtype=torch.float32)
            for p in head.parameters():
                p.requires_grad = True

        # =============================== Alignment head ===============================
        self.hidden_size = int(hidden_size)
        # Backward compatibility:
        #   - enable_alignment_proj now controls the prefix branch only.
        #   - patch branch can stay active via its own linear layer.
        self.enable_alignment_proj = bool(enable_alignment_proj)
        self.enable_patch_linear = bool(enable_patch_linear)
        self.alignment_proj = nn.Linear(
            self.encoder.hidden_size,  # CodeBERT hidden_size (usually 768)
            self.hidden_size,          # decoder hidden size (e.g. 3584/4096)
        )
        self.patch_linear = nn.Linear(
            self.encoder.hidden_size,  # patch hidden from CodeBERT space
            self.encoder.hidden_size,  # keep token-space transform independent
        )
        self.use_patch_tokens = bool(use_patch_tokens)
        self.patch_token_count = int(max(0, patch_token_count))
        self.patch_pooling = str(patch_pooling).lower()
        if self.patch_pooling not in ("mean", "stride"):
            raise ValueError(f"Unsupported patch_pooling={patch_pooling}. Use 'mean' or 'stride'.")
        self._prefix_branch_enabled = bool(self.enable_alignment_proj and self.prefix_length > 0)
        self._patch_branch_enabled = bool(
            self.enable_patch_linear and self.use_patch_tokens and self.patch_token_count > 0
        )
        self._alignment_path_enabled = bool(self._prefix_branch_enabled or self._patch_branch_enabled)

        if not self._prefix_branch_enabled:
            self.encoder.prefix_embedding.requires_grad = False
        if not self._patch_branch_enabled:
            for p in self.patch_linear.parameters():
                p.requires_grad = False
        if not self._alignment_path_enabled:
            for p in self.alignment_proj.parameters():
                p.requires_grad = False

        # =============================== DOCID prefix utils ===============================
        self.docid_prefix = str(docid_prefix or "")

        self._fixed_prefix_text = "<DOCID>"
        if len(self.docid_prefix) > 0:
            self._fixed_prefix_text = f"{self._fixed_prefix_text} {self.docid_prefix}"
        self._fixed_prefix_ids = self.decoder_tokenizer(
            self._fixed_prefix_text,
            add_special_tokens=False,
            return_tensors=None,
        )["input_ids"]  # List[int]

        self._allowed_suffix_ids = self._build_allowed_suffix_ids()
        self._strict_suffix_ids = self._build_strict_suffix_ids()
        self._docid_pattern_token_ids = self._build_docid_pattern_token_ids()
        self.label_smoothing = float(label_smoothing)
        self.dual_phase_guidance_train_enable = _as_bool(dual_phase_guidance_train_enable)
        self.dual_phase_guidance_train_lambda_lex = float(dual_phase_guidance_train_lambda_lex)
        self.dual_phase_guidance_train_lambda_rank_distill = float(dual_phase_guidance_train_lambda_rank_distill)
        self.dual_phase_guidance_train_lex_temperature = max(1e-6, float(dual_phase_guidance_train_lex_temperature))
        self.dual_phase_guidance_train_rank_temperature = max(1e-6, float(dual_phase_guidance_train_rank_temperature))
        # Gradient-tracked Stage-II guidance encoding can be a peak-memory hotspot.
        # Use small chunks by default for stability.
        self.dual_phase_guidance_train_encode_batch_size = max(1, int(dual_phase_guidance_train_encode_batch_size))

        # =============================== Gradual unfreeze schedule ===============================
        self.gradual_unfreeze = bool(gradual_unfreeze)
        self.unfreeze_prefix_epoch = int(unfreeze_prefix_epoch)
        self.unfreeze_alignment_epoch = int(unfreeze_alignment_epoch)
        self.unfreeze_embed_epoch = int(unfreeze_embed_epoch)
        self.unfreeze_lm_head_epoch = int(unfreeze_lm_head_epoch)

        self._last_unfreeze_stage = None
        if self.gradual_unfreeze:
            # Start fully frozen and release by epoch schedule.
            self.encoder.prefix_embedding.requires_grad = False
            for p in self.alignment_proj.parameters():
                p.requires_grad = False
            for p in self.patch_linear.parameters():
                p.requires_grad = False

            emb = self.decoder.get_input_embeddings()
            head = self.decoder.get_output_embeddings()
            for p in emb.parameters():
                p.requires_grad = False
            for p in head.parameters():
                p.requires_grad = False

            # Keep embed/lm_head in FP32 during gradual unfreezing to avoid AMP unscale errors.
            emb.to(dtype=torch.float32)
            head.to(dtype=torch.float32)

        self._bf16_triu_supported_cache: Dict[str, bool] = {}
        self.debug_log_path: Optional[str] = None
        self._install_bf16_causal_mask_patch_if_needed()

    def set_debug_log_path(self, log_path: Optional[str]):
        self.debug_log_path = str(log_path) if log_path else None
        if self.debug_log_path:
            log_dir = os.path.dirname(self.debug_log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            self._debug_log(f"[DBG][model] debug_log_path={self.debug_log_path}")

    def _debug_log(self, message: str):
        if not self.debug_log_path:
            return
        log_dir = os.path.dirname(self.debug_log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(self.debug_log_path, "a", encoding="utf-8") as f:
            f.write(str(message).rstrip("\n") + "\n")

    @classmethod
    def from_config(cls, cfg):
        return cls(
            encoder_name=cfg.encoder_name,
            decoder_name=cfg.decoder_name,
            use_llm=cfg.get("use_llm", True),
            hidden_size=cfg.hidden_size,
            freeze_encoder=cfg.get("freeze_encoder", True),
            freeze_decoder=cfg.get("freeze_decoder", True),
            encoder_max_length=cfg.get("encoder_max_length", 512),
            decoder_max_length=cfg.get("decoder_max_length", 64),
            decoder_dtype=cfg.get("decoder_dtype", "float16"),
            enable_gradient_checkpointing=cfg.get("enable_gradient_checkpointing", True),
            use_lora=cfg.get("use_lora", True),
            lora_r=cfg.get("lora_r", 8),
            lora_alpha=cfg.get("lora_alpha", 16),
            lora_dropout=cfg.get("lora_dropout", 0.05),
            lora_target_modules=cfg.get("lora_target_modules", None),
            use_patch_tokens=cfg.get("use_patch_tokens", False),
            patch_token_count=cfg.get("patch_token_count", 4),
            patch_pooling=cfg.get("patch_pooling", "mean"),
            prefix_length=cfg.get("prefix_length", 10),
            docid_prefix=cfg.get("docid_prefix", ""),
            enable_alignment_proj=cfg.get(
                "enable_alignment_proj",
                cfg.get("enable_prefix_proj", True),
            ),
            enable_patch_linear=cfg.get("enable_patch_linear", True),
            unfreeze_decoder_embed_and_lm_head=cfg.get("unfreeze_decoder_embed_and_lm_head", False),
            gradual_unfreeze=cfg.get("gradual_unfreeze", False),
            unfreeze_prefix_epoch=cfg.get("unfreeze_prefix_epoch", 0),
            unfreeze_alignment_epoch=cfg.get("unfreeze_alignment_epoch", 1),
            unfreeze_embed_epoch=cfg.get("unfreeze_embed_epoch", 3),
            unfreeze_lm_head_epoch=cfg.get("unfreeze_lm_head_epoch", 4),
            label_smoothing=cfg.get("label_smoothing", 0.0),
            dual_phase_guidance_train_enable=cfg.get("dual_phase_guidance_train_enable", False),
            dual_phase_guidance_train_lambda_lex=cfg.get("dual_phase_guidance_train_lambda_lex", 0.0),
            dual_phase_guidance_train_lambda_rank_distill=cfg.get("dual_phase_guidance_train_lambda_rank_distill", 0.0),
            dual_phase_guidance_train_lex_temperature=cfg.get("dual_phase_guidance_train_lex_temperature", 0.07),
            dual_phase_guidance_train_rank_temperature=cfg.get("dual_phase_guidance_train_rank_temperature", 1.0),
            dual_phase_guidance_train_encode_batch_size=cfg.get("dual_phase_guidance_train_encode_batch_size", 1),
        )

    def apply_gradual_unfreeze(self, epoch: int) -> bool:
        """
        Gradual unfreezing by epoch:
          1) prefix_embedding
          2) alignment_proj + patch_linear
          3) decoder embed_tokens
          4) decoder lm_head
        Returns whether the trainable parameter set changed in this call.
        """
        if not self.gradual_unfreeze:
            return False

        stage = 0
        if epoch >= self.unfreeze_prefix_epoch:
            stage = 1
        if epoch >= self.unfreeze_alignment_epoch:
            stage = 2
        if epoch >= self.unfreeze_embed_epoch:
            stage = 3
        if epoch >= self.unfreeze_lm_head_epoch:
            stage = 4

        if self._last_unfreeze_stage == stage:
            return False

        self._last_unfreeze_stage = stage

        self.encoder.prefix_embedding.requires_grad = (stage >= 1) and self._prefix_branch_enabled
        for p in self.alignment_proj.parameters():
            p.requires_grad = (stage >= 2) and self._alignment_path_enabled
        for p in self.patch_linear.parameters():
            p.requires_grad = (stage >= 2) and self._patch_branch_enabled

        emb = self.decoder.get_input_embeddings()
        head = self.decoder.get_output_embeddings()
        for p in emb.parameters():
            p.requires_grad = stage >= 3
        for p in head.parameters():
            p.requires_grad = stage >= 4

        return True

    def _get_qwen2_backbone_for_mask_patch(self):
        # Plain HF model: Qwen2ForCausalLM.model
        direct_model = getattr(self.decoder, "model", None)
        if direct_model is not None and hasattr(direct_model, "_update_causal_mask"):
            return direct_model

        # PEFT wrapped model: PeftModel.base_model.model.model
        base = getattr(self.decoder, "base_model", None)
        if base is not None:
            maybe_lm = getattr(base, "model", None)
            if maybe_lm is not None:
                inner = getattr(maybe_lm, "model", None)
                if inner is not None and hasattr(inner, "_update_causal_mask"):
                    return inner
                if hasattr(maybe_lm, "_update_causal_mask"):
                    return maybe_lm
        return None

    def _is_bf16_triu_supported(self, device: torch.device) -> bool:
        key = f"{device.type}:{getattr(device, 'index', None)}"
        if key in self._bf16_triu_supported_cache:
            return self._bf16_triu_supported_cache[key]

        supported = True
        if device.type == "cuda":
            try:
                t = torch.zeros((2, 2), dtype=torch.bfloat16, device=device)
                _ = torch.triu(t, diagonal=1)
            except RuntimeError:
                supported = False
        self._bf16_triu_supported_cache[key] = supported
        return supported

    def _install_bf16_causal_mask_patch_if_needed(self) -> None:
        qwen_model = self._get_qwen2_backbone_for_mask_patch()
        if qwen_model is None:
            return
        if getattr(qwen_model, "_codehid_bf16_mask_patch_installed", False):
            return

        try:
            from transformers.cache_utils import StaticCache
            from transformers.modeling_attn_mask_utils import AttentionMaskConverter
            from transformers.models.qwen2 import modeling_qwen2 as qwen2_mod
        except Exception:
            return

        original_update_causal_mask = qwen_model._update_causal_mask

        def _patched_update_causal_mask(
            model_self,
            attention_mask: torch.Tensor,
            input_tensor: torch.Tensor,
            cache_position: torch.Tensor,
            past_key_values,
            output_attentions: bool,
        ):
            if input_tensor.dtype != torch.bfloat16 or input_tensor.device.type != "cuda":
                return original_update_causal_mask(
                    attention_mask, input_tensor, cache_position, past_key_values, output_attentions
                )
            if self._is_bf16_triu_supported(input_tensor.device):
                return original_update_causal_mask(
                    attention_mask, input_tensor, cache_position, past_key_values, output_attentions
                )

            if model_self.config._attn_implementation == "flash_attention_2":
                if attention_mask is not None and 0.0 in attention_mask:
                    return attention_mask
                return None

            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            using_static_cache = isinstance(past_key_values, StaticCache)

            if model_self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
                if AttentionMaskConverter._ignore_causal_mask_sdpa(
                    attention_mask,
                    inputs_embeds=input_tensor,
                    past_key_values_length=past_seen_tokens,
                    is_training=model_self.training,
                ):
                    return None

            mask_dtype = torch.float32
            min_dtype = torch.finfo(mask_dtype).min
            sequence_length = input_tensor.shape[1]
            if using_static_cache:
                target_length = past_key_values.get_max_length()
            else:
                target_length = (
                    attention_mask.shape[-1]
                    if isinstance(attention_mask, torch.Tensor)
                    else past_seen_tokens + sequence_length + 1
                )

            causal_mask = qwen2_mod._prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=target_length,
                dtype=mask_dtype,
                device=input_tensor.device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=input_tensor.shape[0],
            )

            if (
                model_self.config._attn_implementation == "sdpa"
                and attention_mask is not None
                and attention_mask.device.type == "cuda"
                and not output_attentions
            ):
                causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

            return causal_mask.to(dtype=input_tensor.dtype)

        qwen_model._update_causal_mask = types.MethodType(_patched_update_causal_mask, qwen_model)
        qwen_model._codehid_bf16_mask_patch_installed = True

    # ------------------------------------------------ Helpers ------------------------------------------------
    def _ensure_docid_prefix(self, raw_id: str) -> str:
        s = str(raw_id).strip()
        if s.startswith("<DOCID>"):
            inner = s.replace("<DOCID>", "").replace("</DOCID>", "").strip()
            if self.docid_prefix and not inner.startswith(self.docid_prefix):
                inner = self.docid_prefix + inner
            return f"<DOCID> {inner} </DOCID>"

        if self.docid_prefix and not s.startswith(self.docid_prefix):
            s = self.docid_prefix + s
        return f"<DOCID> {s} </DOCID>"

    def _tokenize_docids(self, answer_ids):
        structured = [self._ensure_docid_prefix(x) for x in answer_ids]
        return self.decoder_tokenizer(
            structured,
            padding=True,
            truncation=True,
            max_length=self.decoder_max_length,
            return_tensors="pt",
            add_special_tokens=False,
        )

    def _extract_docid_suffix(self, raw_id: str) -> str:
        """
        Extract trainable DocID suffix from answer_id:
          - new format: a1b2c3d4
          - legacy format: XXXXXXXX (8 digits)
        """
        s = str(raw_id).strip().lower()
        s = s.replace("<docid>", " ").replace("</docid>", " ")
        compact = re.sub(r"\s+", "", s)

        # 1) Prefer configured prefix + hierarchical suffix (when prefix is configured)
        if self.docid_prefix:
            prefix_pat = re.escape(self.docid_prefix.lower())
            m = re.search(prefix_pat + r"((?:[a-z]\d+){2,})", compact)
            if m:
                groups = re.findall(r"([a-z])(\d+)", m.group(1))
                if len(groups) > 0:
                    return "".join(f"{k}{int(v)}" for k, v in groups)

        # 2) Accept a1b2c3d4 with no configured prefix in answer text
        m2 = re.search(r"((?:[a-z]\d+){2,})", compact)
        if m2:
            groups = re.findall(r"([a-z])(\d+)", m2.group(1))
            if len(groups) > 0:
                return "".join(f"{k}{int(v)}" for k, v in groups)

        # 3) Fallback to legacy 8-digit format
        m3 = re.search(r"(\d{8})", compact)
        if m3:
            return m3.group(1)

        digits = "".join(ch for ch in compact if ch.isdigit())
        if len(digits) == 0:
            # Fallback to avoid empty target causing training failure.
            return "a0b0c0d0"
        if len(digits) < 8:
            digits = digits.zfill(8)
        return digits[-8:]

    def _tokenize_docid_targets(self, answer_ids):
        targets = [f"{self._extract_docid_suffix(x)}</DOCID>" for x in answer_ids]
        return self.decoder_tokenizer(
            targets,
            padding=True,
            truncation=True,
            max_length=self.decoder_max_length,
            return_tensors="pt",
            add_special_tokens=False,
        )

    def _build_allowed_suffix_ids(self):
        tok = self.decoder_tokenizer
        allowed = set()

        candidates = []
        # Transitional mode: allow a/b/c/d and numeric token fragments.
        for ch in "0123456789abcd":
            candidates += [ch, " " + ch]
        for ch in "abcd":
            for n in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "15", "31", "63"]:
                candidates += [f"{ch}{n}", f" {ch}{n}"]
        candidates += [
            "00", "000", "0000", "00000000",
            "a0b0c0d0",
        ]

        for s in candidates:
            ids = tok(s, add_special_tokens=False)["input_ids"]
            for tid in ids:
                allowed.add(int(tid))

        end_id = tok.convert_tokens_to_ids("</DOCID>")
        if end_id is None or end_id < 0:
            raise ValueError("Cannot find token id for </DOCID> (did you add_special_tokens?)")
        end_id = int(end_id)

        banned = set(getattr(tok, "all_special_ids", []))
        for attr in ["bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"]:
            tid = getattr(tok, attr, None)
            if tid is not None:
                banned.add(int(tid))

        allowed = allowed - banned
        allowed.add(end_id)
        if len(allowed) == 0:
            raise ValueError("Allowed token set is empty after filtering specials.")

        return sorted(list(allowed))

    def _build_strict_suffix_ids(self):
        """
        Prefer single-token [a-d0-9].
        Fall back to the relaxed allowed set if tokenizer coverage is insufficient.
        """
        tok = self.decoder_tokenizer
        strict_ids = set()
        for ch in "abcd0123456789":
            ids = tok(ch, add_special_tokens=False)["input_ids"]
            if len(ids) == 1:
                strict_ids.add(int(ids[0]))
        return sorted(list(strict_ids))

    def _build_docid_pattern_token_ids(self) -> Optional[Dict[str, List[int]]]:
        """
        Build token ids for structural pattern:
          a<1-3 digits>b<1-3 digits>c<1-3 digits>d<1-3 digits>
        Returns None when tokenizer cannot represent required chars as single token.
        """
        tok = self.decoder_tokenizer

        def _single_token_id(text: str) -> Optional[int]:
            ids = tok(text, add_special_tokens=False)["input_ids"]
            if len(ids) != 1:
                return None
            return int(ids[0])

        letter_ids: List[int] = []
        for ch in "abcd":
            tid = _single_token_id(ch)
            if tid is None:
                return None
            letter_ids.append(tid)

        digit_ids: List[int] = []
        for ch in "0123456789":
            tid = _single_token_id(ch)
            if tid is None:
                return None
            digit_ids.append(tid)

        return {
            "letters": letter_ids,
            "digits": digit_ids,
        }

    def _pattern_allowed_next(self, generated_token_ids: List[int], end_id: int) -> List[int]:
        """
        Structural fallback constraint:
          a<1-3 digits>b<1-3 digits>c<1-3 digits>d<1-3 digits></DOCID>
        """
        pattern = self._docid_pattern_token_ids
        if pattern is None:
            return []

        letters = [int(x) for x in pattern["letters"]]
        digits = [int(x) for x in pattern["digits"]]
        digit_set = set(digits)

        seg_idx = 0
        phase = "letter"  # "letter" or "digits"
        digit_count = 0

        for tid in generated_token_ids:
            tid = int(tid)
            if phase == "letter":
                if tid != letters[seg_idx]:
                    return []
                phase = "digits"
                digit_count = 0
                continue

            if tid in digit_set and digit_count < 3:
                digit_count += 1
                continue

            if digit_count >= 1 and seg_idx < len(letters) - 1 and tid == letters[seg_idx + 1]:
                seg_idx += 1
                digit_count = 0
                continue

            return []

        if phase == "letter":
            return [letters[seg_idx]]

        allowed: List[int] = []
        if digit_count < 3:
            allowed.extend(digits)
        if digit_count >= 1:
            if seg_idx < len(letters) - 1:
                allowed.append(letters[seg_idx + 1])
            else:
                allowed.append(int(end_id))

        uniq: List[int] = []
        seen = set()
        for tid in allowed:
            tid = int(tid)
            if tid in seen:
                continue
            seen.add(tid)
            uniq.append(tid)
        return uniq

    def build_docid_token_trie(self, docids: Iterable[str]) -> Dict:
        """
        Build a token trie for constrained decoding.
        Each path corresponds to: <docid_suffix_tokens> + </DOCID>
        """
        docids = list(docids)
        tok = self.decoder_tokenizer
        end_id = tok.convert_tokens_to_ids("</DOCID>")
        if end_id is None or end_id < 0:
            raise ValueError("Cannot find token id for </DOCID> (did you add_special_tokens?)")
        end_id = int(end_id)

        self._debug_log(f"[DBG][trie] vocab_docids={len(docids)} end_id={end_id}")
        root: Dict = {}
        for raw in docids:
            suffix = self._extract_docid_suffix(raw)
            ids = tok(
                f"{suffix}</DOCID>",
                add_special_tokens=False,
                return_tensors=None,
            )["input_ids"]
            self._debug_log(
                f"[DBG][trie][entry] raw={repr(raw)} suffix={repr(suffix)} token_ids={ids}"
            )
            if len(ids) == 0:
                continue
            node = root
            for tid in ids:
                tid = int(tid)
                node = node.setdefault(tid, {})
            node.setdefault("_end_", True)

        if len(root) == 0:
            raise ValueError("Empty trie built from provided docids.")
        self._debug_log(f"[DBG][trie] root_branch_count={len(root)}")
        return root

    def _trie_allowed_next(self, trie_root: Dict, generated_token_ids: List[int]) -> List[int]:
        node = trie_root
        for tid in generated_token_ids:
            tid = int(tid)
            if tid not in node:
                return []
            node = node[tid]
        return [int(k) for k in node.keys() if k != "_end_"]

    def _get_underlying_roberta(self):
        """
        Try to resolve the actual RoBERTa/CodeBERT backbone from CodeBERTEncoder.
        CodeBERTEncoder.forward may return only prefix embeddings; therefore this path
        bypasses forward and calls the backbone to get input-dependent last_hidden_state.
        """
        # Try common backbone field names.
        for attr in ["model", "encoder", "roberta", "backbone", "bert"]:
            if hasattr(self.encoder, attr):
                m = getattr(self.encoder, attr)
                if m is not None:
                    return m
        return None

    def _encode_with_roberta(self, input_ids, attention_mask):
        """
        Returns contextual hidden states: [B, L, 768]
        """
        roberta = self._get_underlying_roberta()
        if roberta is None:
            raise AttributeError(
                "Cannot access underlying RoBERTa from CodeBERTEncoder. "
                "Please expose it as .model or .encoder etc."
            )
        out = roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return out.last_hidden_state

    def _pool_last_hidden_mean(self, last_hidden, attention_mask):
        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled = (last_hidden * mask).sum(dim=1) / denom
        return pooled

    def _encode_texts_for_dual_phase_guidance_train_impl(self, texts, batch_size=64, normalize=True, with_grad=False):
        if len(texts) == 0:
            dim = int(self.hidden_size) if self._alignment_path_enabled else int(self.encoder.hidden_size)
            return torch.zeros((0, dim), device=self.device, dtype=torch.float32)

        outputs = []
        was_training = self.training
        if not with_grad:
            self.eval()

        ctx = torch.enable_grad() if with_grad else torch.no_grad()
        with ctx:
            for st in range(0, len(texts), max(1, int(batch_size))):
                ed = min(st + int(batch_size), len(texts))
                enc = self.encoder_tokenizer(
                    texts[st:ed],
                    padding=True,
                    truncation=True,
                    max_length=self.encoder_max_length,
                    return_tensors="pt",
                ).to(self.device)

                hidden = self._encode_with_roberta(
                    input_ids=enc.input_ids,
                    attention_mask=enc.attention_mask,
                )
                pooled = self._pool_last_hidden_mean(hidden, enc.attention_mask)
                if self._alignment_path_enabled:
                    proj_dtype = self.alignment_proj.weight.dtype
                    pooled = pooled.to(dtype=proj_dtype)
                    pooled = self.alignment_proj(pooled)
                pooled = pooled.float()
                if normalize:
                    pooled = F.normalize(pooled, p=2, dim=-1)
                outputs.append(pooled)

        if not with_grad:
            self.train(was_training)

        return torch.cat(outputs, dim=0)

    def encode_texts_for_dual_phase_guidance_train(self, texts, batch_size=64, normalize=True):
        return self._encode_texts_for_dual_phase_guidance_train_impl(
            texts=texts,
            batch_size=batch_size,
            normalize=normalize,
            with_grad=False,
        )

    def _extract_encoder_prefix(self, enc_hidden):
        """
        Take the first prefix_length tokens and add learnable prefix embeddings
        so prefix-tuning parameters participate in the main path.
        """
        use_len = min(enc_hidden.size(1), self.prefix_length)
        enc_hidden = enc_hidden[:, :use_len, :]
        learned_prefix = self.encoder.prefix_embedding[:use_len].unsqueeze(0).to(
            device=enc_hidden.device,
            dtype=enc_hidden.dtype,
        )
        return enc_hidden + learned_prefix

    def _project_to_decoder(self, enc_hidden):
        """
        Dtype guard: Linear(mat1/mat2) dtypes must match.
        """
        proj_dtype = self.alignment_proj.weight.dtype
        if enc_hidden.dtype != proj_dtype:
            enc_hidden = enc_hidden.to(dtype=proj_dtype)

        encoder_lm_embeds = self.alignment_proj(enc_hidden)

        # Decoder side is usually fp16/bf16; align prefix embeds with decoder dtype.
        encoder_lm_embeds = encoder_lm_embeds.to(dtype=self.decoder.dtype)
        return encoder_lm_embeds

    def _project_prefix(self, enc_hidden):
        if not self._prefix_branch_enabled:
            # Alignment path disabled: no encoder prefix is injected into decoder inputs.
            return torch.zeros(
                (enc_hidden.size(0), 0, self.hidden_size),
                device=enc_hidden.device,
                dtype=self.decoder.dtype,
            )
        return self._project_to_decoder(enc_hidden)

    def _project_patch(self, patch_hidden):
        patch_dtype = self.patch_linear.weight.dtype
        if patch_hidden.dtype != patch_dtype:
            patch_hidden = patch_hidden.to(dtype=patch_dtype)
        patch_hidden = self.patch_linear(patch_hidden)
        return self._project_to_decoder(patch_hidden)

    def _build_patch_lm_embeds(self, enc_hidden_full, enc_attention_mask):
        """
        Optional "patch token" branch for text-only DSI:
        sample/aggregate extra tokens from encoder sequence and map to decoder hidden.
        """
        if not self._patch_branch_enabled:
            return None, None

        if (not self.use_patch_tokens) or self.patch_token_count <= 0:
            return None, None

        # Avoid duplicating the same prefix positions already consumed by prefix branch.
        start = min(self.prefix_length, enc_hidden_full.size(1))
        token_hidden = enc_hidden_full[:, start:, :]
        token_mask = enc_attention_mask[:, start:] if enc_attention_mask is not None else None
        if token_hidden.size(1) == 0:
            token_hidden = enc_hidden_full
            token_mask = enc_attention_mask

        B, L, H = token_hidden.shape
        if L == 0:
            return None, None
        n = min(self.patch_token_count, L)

        if self.patch_pooling == "stride":
            idx = torch.linspace(0, L - 1, steps=n, device=token_hidden.device)
            idx = torch.round(idx).long()
            patch_hidden = token_hidden.index_select(1, idx)
            patch_mask = torch.ones((B, n), device=token_hidden.device, dtype=torch.long)
            return self._project_patch(patch_hidden), patch_mask

        # mean pooling over n contiguous chunks
        bounds = torch.linspace(0, L, steps=n + 1, device=token_hidden.device)
        bounds = torch.floor(bounds).long()
        pooled = []
        pooled_mask = []
        for i in range(n):
            s = int(bounds[i].item())
            e = int(bounds[i + 1].item())
            if e <= s:
                e = min(s + 1, L)
            chunk = token_hidden[:, s:e, :]
            if token_mask is None:
                chunk_out = chunk.mean(dim=1)
                valid = torch.ones((B,), device=token_hidden.device, dtype=torch.long)
            else:
                m = token_mask[:, s:e].unsqueeze(-1).to(chunk.dtype)
                denom = m.sum(dim=1).clamp_min(1.0)
                chunk_out = (chunk * m).sum(dim=1) / denom
                valid = (m.sum(dim=1).squeeze(-1) > 0).to(dtype=torch.long)
            pooled.append(chunk_out.unsqueeze(1))
            pooled_mask.append(valid.unsqueeze(1))

        patch_hidden = torch.cat(pooled, dim=1)
        patch_mask = torch.cat(pooled_mask, dim=1)
        return self._project_patch(patch_hidden), patch_mask

    # ------------------------------------------------ Training forward ------------------------------------------------
    def forward(self, samples):
        device = self.device

        encoder_texts = [
            f"Query: {q}\nCode:\n{c}"
            for q, c in zip(samples["query"], samples["code"])
        ]

        enc = self.encoder_tokenizer(
            encoder_texts,
            padding=True,
            truncation=True,
            max_length=self.encoder_max_length,
            return_tensors="pt",
        ).to(device)

        # Key change: do not call self.encoder(...); call backbone for last_hidden_state.
        enc_hidden_full = self._encode_with_roberta(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
        )

        enc_hidden = self._extract_encoder_prefix(enc_hidden_full)

        # Alignment projection with dtype alignment.
        encoder_lm_embeds = self._project_prefix(enc_hidden)
        B, _, _ = encoder_lm_embeds.shape

        prefix_attention_mask = torch.ones(
            (B, encoder_lm_embeds.size(1)),
            device=device,
            dtype=torch.long,
        )
        patch_lm_embeds, patch_attention_mask = self._build_patch_lm_embeds(
            enc_hidden_full=enc_hidden_full,
            enc_attention_mask=enc.attention_mask,
        )
        if patch_lm_embeds is not None:
            encoder_lm_embeds = torch.cat([encoder_lm_embeds, patch_lm_embeds], dim=1)
            prefix_attention_mask = torch.cat(
                [prefix_attention_mask, patch_attention_mask.to(device=device, dtype=torch.long)],
                dim=1,
            )
        B, L_enc, _ = encoder_lm_embeds.shape

        prompt_ids = torch.tensor(
            self._fixed_prefix_ids, device=device, dtype=torch.long
        ).unsqueeze(0).expand(B, -1)
        embed_layer = self.decoder.get_input_embeddings()
        prompt_embeds = embed_layer(prompt_ids).to(dtype=self.decoder.dtype)
        prompt_attention_mask = torch.ones(
            (B, prompt_ids.size(1)), device=device, dtype=torch.long
        )

        target = self._tokenize_docid_targets(samples["answer_id"])
        target = {k: v.to(device) for k, v in target.items()}
        target_input_ids = target["input_ids"]
        target_attention_mask = target.get(
            "attention_mask", torch.ones_like(target_input_ids)
        )
        target_embeds = embed_layer(target_input_ids).to(dtype=self.decoder.dtype)

        inputs_embeds = torch.cat([encoder_lm_embeds, prompt_embeds, target_embeds], dim=1)
        attention_mask = torch.cat(
            [prefix_attention_mask, prompt_attention_mask, target_attention_mask], dim=1
        )

        L_prompt = prompt_ids.size(1)
        L_tgt = target_input_ids.size(1)
        labels = torch.full(
            (B, L_enc + L_prompt + L_tgt),
            fill_value=-100,
            dtype=torch.long,
            device=device,
        )
        labels[:, L_enc + L_prompt:] = target_input_ids
        labels[labels == self.decoder_tokenizer.pad_token_id] = -100

        outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )

        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)

        if self.label_smoothing > 0:
            loss_seq = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                label_smoothing=self.label_smoothing,
            )
        else:
            loss_seq = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        dual_phase_guidance_train_enable = _as_bool(
            samples.get("dual_phase_guidance_train_enable", self.dual_phase_guidance_train_enable)
        )
        lambda_lex = float(
            samples.get("dual_phase_guidance_train_lambda_lex", self.dual_phase_guidance_train_lambda_lex)
        )
        lambda_rank = float(
            samples.get("dual_phase_guidance_train_lambda_rank_distill", self.dual_phase_guidance_train_lambda_rank_distill)
        )

        loss_lex = torch.zeros((), device=device, dtype=loss_seq.dtype)
        loss_rank_distill = torch.zeros((), device=device, dtype=loss_seq.dtype)

        if dual_phase_guidance_train_enable and (lambda_lex > 0.0 or lambda_rank > 0.0):
            hard_negative_codes = samples.get("hard_negative_codes", None)
            teacher_scores = samples.get("rank_teacher_scores", None)

            if isinstance(hard_negative_codes, list) and len(hard_negative_codes) == B:
                candidate_texts_flat = []
                spans = []
                for i in range(B):
                    neg_codes = hard_negative_codes[i] if isinstance(hard_negative_codes[i], list) else []
                    cand_texts = [f"Code:\n{samples['code'][i]}"] + [f"Code:\n{x}" for x in neg_codes]
                    st = len(candidate_texts_flat)
                    candidate_texts_flat.extend(cand_texts)
                    spans.append((st, len(candidate_texts_flat)))

                query_texts = [f"Query: {q}\nCode:\n" for q in samples["query"]]
                dp_encode_bs = max(1, int(self.dual_phase_guidance_train_encode_batch_size))
                q_vecs = self._encode_texts_for_dual_phase_guidance_train_impl(
                    query_texts, batch_size=min(dp_encode_bs, max(1, len(query_texts))), normalize=True, with_grad=True
                )
                cand_vecs = self._encode_texts_for_dual_phase_guidance_train_impl(
                    candidate_texts_flat,
                    batch_size=min(dp_encode_bs, max(1, len(candidate_texts_flat))),
                    normalize=True,
                    with_grad=True,
                )

                lex_terms = []
                distill_terms = []
                for i, (st, ed) in enumerate(spans):
                    cand = cand_vecs[st:ed]
                    if cand.size(0) <= 1:
                        continue

                    score = torch.matmul(cand, q_vecs[i])
                    if lambda_lex > 0.0:
                        logits = score / self.dual_phase_guidance_train_lex_temperature
                        lex_terms.append(
                            F.cross_entropy(
                                logits.unsqueeze(0),
                                torch.zeros((1,), device=device, dtype=torch.long),
                            )
                        )

                    if lambda_rank > 0.0 and isinstance(teacher_scores, list) and i < len(teacher_scores):
                        t = teacher_scores[i]
                        if isinstance(t, list) and len(t) == cand.size(0):
                            t_tensor = torch.tensor(t, device=device, dtype=score.dtype)
                            s_log_probs = F.log_softmax(score / self.dual_phase_guidance_train_rank_temperature, dim=-1)
                            t_probs = F.softmax(t_tensor / self.dual_phase_guidance_train_rank_temperature, dim=-1)
                            distill = F.kl_div(s_log_probs, t_probs, reduction="batchmean")
                            distill_terms.append(distill * (self.dual_phase_guidance_train_rank_temperature ** 2))

                if len(lex_terms) > 0:
                    loss_lex = torch.stack(lex_terms).mean().to(dtype=loss_seq.dtype)
                if len(distill_terms) > 0:
                    loss_rank_distill = torch.stack(distill_terms).mean().to(dtype=loss_seq.dtype)

        loss = loss_seq + (lambda_lex * loss_lex) + (lambda_rank * loss_rank_distill)

        return {
            "loss": loss,
            "logits": outputs.logits,
            "loss_seq_docid": loss_seq.detach(),
            "loss_lex": loss_lex.detach(),
            "loss_dual_phase_guidance_train_rank_distill": loss_rank_distill.detach(),
            "loss_rank_distill": loss_rank_distill.detach(),
        }

    # ------------------------------------------------ Generation ------------------------------------------------
    @torch.no_grad()
    def generate(self, samples, **gen_kwargs):
        device = self.device
        tok = self.decoder_tokenizer

        encoder_texts = [
            f"Query: {q}\nCode:\n{c}"
            for q, c in zip(samples["query"], samples["code"])
        ]

        enc = self.encoder_tokenizer(
            encoder_texts,
            padding=True,
            truncation=True,
            max_length=self.encoder_max_length,
            return_tensors="pt",
        ).to(device)

        # Key change: do not call self.encoder(...); call backbone for last_hidden_state.
        enc_hidden_full = self._encode_with_roberta(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
        )

        enc_hidden = self._extract_encoder_prefix(enc_hidden_full)

        # Alignment projection with dtype alignment.
        encoder_lm_embeds = self._project_prefix(enc_hidden)
        B, _, _ = encoder_lm_embeds.shape

        encoder_attention_mask = torch.ones(
            (B, encoder_lm_embeds.size(1)), device=device, dtype=torch.long
        )
        patch_lm_embeds, patch_attention_mask = self._build_patch_lm_embeds(
            enc_hidden_full=enc_hidden_full,
            enc_attention_mask=enc.attention_mask,
        )
        if patch_lm_embeds is not None:
            encoder_lm_embeds = torch.cat([encoder_lm_embeds, patch_lm_embeds], dim=1)
            encoder_attention_mask = torch.cat(
                [encoder_attention_mask, patch_attention_mask.to(device=device, dtype=torch.long)],
                dim=1,
            )
        B, L_enc, _ = encoder_lm_embeds.shape

        prompt_ids = torch.tensor(self._fixed_prefix_ids, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)
        embed_layer = self.decoder.get_input_embeddings()
        prompt_embeds = embed_layer(prompt_ids).to(dtype=self.decoder.dtype)
        prompt_attention_mask = torch.ones((B, prompt_ids.size(1)), device=device, dtype=torch.long)

        inputs_embeds = torch.cat([encoder_lm_embeds, prompt_embeds], dim=1)
        attention_mask = torch.cat([encoder_attention_mask, prompt_attention_mask], dim=1)

        end_id = tok.convert_tokens_to_ids("</DOCID>")
        if end_id is None or end_id < 0:
            raise ValueError("Cannot find token id for </DOCID> (did you add_special_tokens?)")

        constrained_trie = gen_kwargs.pop("constrained_trie", None)
        constrained_docids = gen_kwargs.pop("constrained_docids", None)
        if constrained_trie is None and constrained_docids is not None:
            constrained_trie = self.build_docid_token_trie(constrained_docids)
        self._debug_log(
            f"[DBG][generate] constrained_trie={'yes' if constrained_trie is not None else 'no'} "
            f"num_beams={gen_kwargs.get('num_beams', 5)} max_new_tokens={gen_kwargs.get('max_new_tokens', None)}"
        )

        gen_kwargs.setdefault("eos_token_id", int(end_id))
        gen_kwargs.setdefault("pad_token_id", int(tok.pad_token_id))
        gen_kwargs.setdefault("min_new_tokens", 4)

        gen_kwargs.setdefault("do_sample", False)
        gen_kwargs.setdefault("num_beams", 5)
        gen_kwargs.setdefault("temperature", 0.0)

        allowed_ids = self._allowed_suffix_ids
        strict_step = len(self._strict_suffix_ids) >= 8
        if strict_step:
            suffix_only_ids = self._strict_suffix_ids
        else:
            suffix_only_ids = [tid for tid in allowed_ids if tid != int(end_id)]
        prefix_total_len = L_enc + prompt_ids.size(1)

        lexical_prefix_bonus = gen_kwargs.pop("dual_phase_guidance_infer_lexical_prefix_bonus", None)
        if lexical_prefix_bonus is None:
            lexical_prefix_bonus = gen_kwargs.pop("lexical_prefix_bonus", None)
        lexical_bonus_weight = float(
            gen_kwargs.pop(
                "dual_phase_guidance_infer_lexical_bonus_weight",
                gen_kwargs.pop("lexical_bonus_weight", 0.0),
            )
        )
        if lexical_prefix_bonus is not None:
            if isinstance(lexical_prefix_bonus, dict):
                lexical_prefix_bonus = [lexical_prefix_bonus for _ in range(B)]
            if not isinstance(lexical_prefix_bonus, list):
                raise TypeError("lexical_prefix_bonus must be a dict or list of dicts.")
            if len(lexical_prefix_bonus) != B:
                raise ValueError(
                    f"lexical_prefix_bonus batch mismatch: got {len(lexical_prefix_bonus)} but batch is {B}"
                )

            lp = BatchPrefixLexicalLogitsProcessor(
                batch_prefix_bonus=lexical_prefix_bonus,
                prefix_total_len=prefix_total_len,
                num_beams=int(gen_kwargs.get("num_beams", 5)),
                bonus_weight=lexical_bonus_weight,
            )
            logits_processor = gen_kwargs.pop("logits_processor", None)
            if logits_processor is None:
                logits_processor = LogitsProcessorList()
            elif not isinstance(logits_processor, LogitsProcessorList):
                logits_processor = LogitsProcessorList(logits_processor)
            logits_processor.append(lp)
            gen_kwargs["logits_processor"] = logits_processor
            self._debug_log(
                f"[DBG][dual_phase_guidance_infer_tmp_rescore] enabled batch={B} bonus_weight={lexical_bonus_weight} "
                f"num_beams={gen_kwargs.get('num_beams', 5)}"
            )

        def prefix_allowed_tokens_fn(batch_id, input_ids):
            gen_len = input_ids.shape[0] - prefix_total_len
            if constrained_trie is not None:
                generated = input_ids[prefix_total_len:].tolist()
                trie_allowed = self._trie_allowed_next(constrained_trie, generated)
                if len(trie_allowed) > 0:
                    return trie_allowed
                # If trie misses, use strict structural fallback first.
                pattern_allowed = self._pattern_allowed_next(generated, int(end_id))
                if len(pattern_allowed) > 0:
                    generated_text = tok.decode(
                        generated,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    self._debug_log(
                        f"[DBG][fallback_pattern] batch={batch_id} gen_len={gen_len} "
                        f"generated_ids={generated} generated_text={repr(generated_text)} "
                        f"pattern_allowed={pattern_allowed}"
                    )
                    return pattern_allowed

                # Last-resort fallback: force </DOCID> to stop invalid branch.
                generated_text = tok.decode(
                    generated,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                self._debug_log(
                    f"[DBG][fallback_end_only] batch={batch_id} gen_len={gen_len} "
                    f"generated_ids={generated} generated_text={repr(generated_text)} "
                    f"allowed_end_id={int(end_id)}"
                )
                return [int(end_id)]
            if gen_len < 2:
                return suffix_only_ids
            if strict_step and gen_len >= 24:
                return [int(end_id)]
            return allowed_ids

        gen_kwargs["prefix_allowed_tokens_fn"] = prefix_allowed_tokens_fn

        if "max_new_tokens" not in gen_kwargs and "max_length" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 24 if strict_step else 32

        if "max_length" in gen_kwargs and "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = int(gen_kwargs.pop("max_length"))

        gen_kwargs.setdefault("use_cache", True)

        # Transformers (decoder-only + inputs_embeds) initializes empty input_ids by
        # default. With beam search + prefix_allowed_tokens_fn, that hits a reshape
        # error in PrefixConstrainedLogitsProcessor when seq_len == 0.
        # Provide bookkeeping input_ids aligned with inputs_embeds length.
        generation_input_ids = gen_kwargs.pop("input_ids", None)
        if generation_input_ids is None or generation_input_ids.shape[-1] == 0:
            filler_id = tok.pad_token_id
            if filler_id is None:
                filler_id = tok.eos_token_id
            if filler_id is None:
                raise ValueError("decoder tokenizer has neither pad_token_id nor eos_token_id for generation")
            generation_input_ids = torch.full(
                (B, inputs_embeds.size(1)),
                int(filler_id),
                device=device,
                dtype=torch.long,
            )

        return self.decoder.generate(
            input_ids=generation_input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **gen_kwargs,
        )
