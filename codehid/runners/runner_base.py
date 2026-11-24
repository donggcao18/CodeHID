"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
"""

import json
import logging
import os
import glob
from pathlib import Path
from typing import Any, Dict, List

import torch
from codehid.common.dist_utils import is_main_process, main_process
from codehid.common.registry import registry
from codehid.datasets.data_utils import reorg_datasets_by_split
from codehid.datasets.datasets.dataloader_utils import IterLoader, PrefetchLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader


def _fallback_collate(batch: List[Any]) -> Any:
    """
    Safe collate for string-heavy samples.
    """
    if len(batch) == 0:
        return batch

    elem = batch[0]
    if isinstance(elem, dict):
        out: Dict[str, Any] = {}
        for k in elem.keys():
            out[k] = [b[k] for b in batch]
        return out

    if isinstance(elem, (list, tuple)):
        return [_fallback_collate([b[i] for b in batch]) for i in range(len(elem))]

    return batch


@registry.register_runner("runner_base")
class RunnerBase:
    """
    Modified runner with:
      - list/tuple dataset unwrap
      - safe collate_fn
      - per-epoch checkpoint saving
    """

    def __init__(self, cfg, task, model, datasets, job_id):
        self.config = cfg
        self.job_id = job_id

        self.task = task
        self.datasets = datasets

        self._model = model
        self._wrapped_model = None
        self._device = None
        self._optimizer = None
        self._scaler = None
        self._dataloaders = None
        self._lr_sched = None

        self.start_epoch = 0
        self.best_metric_value = None
        self.setup_output_dir()

    # ------------------------------------------------------------------
    # device / model
    # ------------------------------------------------------------------
    @property
    def device(self):
        if self._device is None:
            self._device = torch.device(self.config.run_cfg.device)
        return self._device

    @property
    def use_distributed(self):
        return self._as_bool(self.config.run_cfg.distributed)

    @property
    def parallel_strategy(self):
        return str(self.config.run_cfg.get("parallel_strategy", "ddp")).lower()

    def _wrap_model(self):
        has_trainable = any(p.requires_grad for p in self._model.parameters())
        if not has_trainable:
            # Gradual-unfreeze stage may temporarily freeze all params.
            # Defer distributed wrapping until at least one parameter becomes trainable.
            return self._model

        return DDP(self._model, device_ids=[self.config.run_cfg.gpu])

    @property
    def model(self):
        if self._model.device != self.device:
            self._model = self._model.to(self.device)

        if self._wrapped_model is None:
            if self.use_distributed:
                self._wrapped_model = self._wrap_model()
            else:
                self._wrapped_model = self._model
        return self._wrapped_model

    # ------------------------------------------------------------------
    def unwrap_dist_model(self, model):
        if hasattr(model, "module"):
            return model.module
        return model

    def _print_trainable_params(self, max_lines: int = 200):
        model_ref = self._wrapped_model if self._wrapped_model is not None else self._model
        model_no_ddp = self.unwrap_dist_model(model_ref)
        total, trainable = 0, 0
        lines = []
        for n, p in model_no_ddp.named_parameters():
            num = p.numel()
            total += num
            if p.requires_grad:
                trainable += num
                lines.append(f"[TRAIN] {n:70s} dtype={str(p.dtype):12s} shape={tuple(p.shape)} numel={num}")
        logging.info(f"Trainable params: {trainable} / {total} ({100.0 * trainable / max(total,1):.6f}%)")
        for s in lines[:max_lines]:
            logging.info(s)
        if len(lines) > max_lines:
            logging.info(f"... ({len(lines) - max_lines} more trainable params not shown)")

    # ------------------------------------------------------------------
    # optimizer / scaler / lr scheduler
    # ------------------------------------------------------------------
    @property
    def optimizer(self):
        if self._optimizer is None:
            # ensure model is moved to device / wrapped before scanning parameters
            _ = self.model

            p_wd, p_non_wd = [], []
            for n, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                    p_non_wd.append(p)
                else:
                    p_wd.append(p)

            optim_params = [
                {"params": p_wd, "weight_decay": float(self.config.run_cfg.weight_decay)},
                {"params": p_non_wd, "weight_decay": 0},
            ]
            beta2 = self.config.run_cfg.get("beta2", 0.999)
            self._optimizer = torch.optim.AdamW(
                optim_params,
                lr=float(self.config.run_cfg.init_lr),
                betas=(0.9, beta2),
            )
        return self._optimizer

    @property
    def scaler(self):
        amp_enabled = self._as_bool(self.config.run_cfg.get("amp", False), default=False)
        amp_dtype = str(
            self.config.run_cfg.get(
                "amp_dtype",
                self.config.model_cfg.get("decoder_dtype", "float16"),
            )
        ).lower()
        # bf16 does not require GradScaler.
        if amp_enabled and amp_dtype not in ("bf16", "bfloat16"):
            if self._scaler is None:
                self._scaler = torch.cuda.amp.GradScaler()
        return self._scaler

    @property
    def lr_scheduler(self):
        if self._lr_sched is None:
            lr_sched_cls = registry.get_lr_scheduler_class(self.config.run_cfg.lr_sched)

            max_epoch = int(self.config.run_cfg.max_epoch)
            min_lr = float(self.config.run_cfg.min_lr)
            init_lr = float(self.config.run_cfg.init_lr)

            warmup_start_lr = self.config.run_cfg.get("warmup_lr", -1)
            warmup_steps = self.config.run_cfg.get("warmup_steps", 0)

            iters_per_epoch = self.config.run_cfg.get("iters_per_epoch", None)
            if iters_per_epoch is None:
                try:
                    iters_per_epoch = len(self.dataloaders["train"])
                except Exception:
                    iters_per_epoch = 10000

            self._lr_sched = lr_sched_cls(
                optimizer=self.optimizer,
                max_epoch=max_epoch,
                iters_per_epoch=iters_per_epoch,
                min_lr=min_lr,
                init_lr=init_lr,
                warmup_start_lr=warmup_start_lr,
                warmup_steps=warmup_steps,
            )
        return self._lr_sched

    # ------------------------------------------------------------------
    # dataloaders
    # ------------------------------------------------------------------
    @property
    def train_splits(self):
        return self.config.run_cfg.get("train_splits", [])

    @property
    def dataloaders(self):
        if self._dataloaders is None:
            logging.info("Building dataloaders")

            datasets = reorg_datasets_by_split(self.datasets)
            self.datasets = datasets

            split_names = sorted(datasets.keys())
            datasets_list = [datasets[s] for s in split_names]
            is_trains = [s in self.train_splits for s in split_names]

            batch_sizes = [
                self.config.run_cfg.batch_size_train if s == "train"
                else self.config.run_cfg.batch_size_eval
                for s in split_names
            ]

            loaders = []
            for ds, bsz, is_train in zip(datasets_list, batch_sizes, is_trains):
                if isinstance(ds, (list, tuple)):
                    if len(ds) == 0:
                        raise ValueError("Empty dataset container (list/tuple) encountered.")
                    base_ds = ds[0]
                else:
                    base_ds = ds

                collate_fn = getattr(base_ds, "collater", None)
                if collate_fn is None:
                    collate_fn = getattr(base_ds, "collate_fn", None)
                if collate_fn is None:
                    collate_fn = _fallback_collate

                loader = DataLoader(
                    base_ds,
                    batch_size=bsz,
                    shuffle=is_train,
                    num_workers=int(self.config.run_cfg.num_workers),
                    pin_memory=True,
                    drop_last=True if is_train else False,
                    collate_fn=collate_fn,
                )

                loader = PrefetchLoader(loader)
                if is_train:
                    loader = IterLoader(loader, use_distributed=False)

                loaders.append(loader)

            self._dataloaders = dict(zip(split_names, loaders))

        return self._dataloaders

    @property
    def train_loader(self):
        return self.dataloaders["train"]

    # ------------------------------------------------------------------
    # output
    # ------------------------------------------------------------------
    def setup_output_dir(self):
        lib_root = Path(registry.get_path("library_root"))
        output_dir = lib_root / self.config.run_cfg.output_dir / self.job_id
        result_dir = output_dir / "result"

        output_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)

        registry.register_path("result_dir", str(result_dir))
        registry.register_path("output_dir", str(output_dir))

        self.output_dir = output_dir

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    def train(self):
        self.log_config()

        # Apply epoch-0 gradual unfreeze before any DDP wrapping/optimizer build.
        self._maybe_apply_gradual_unfreeze(self.start_epoch)

        try:
            self._print_trainable_params()
        except Exception as e:
            logging.warning(f"Failed to print trainable params: {e}")

        for epoch in range(self.start_epoch, int(self.config.run_cfg.max_epoch)):
            logging.info(f"Epoch {epoch} start")

            self._maybe_apply_gradual_unfreeze(epoch)

            # Support YAML key `gradient_accumulation_steps`.
            accum = (
                self.config.run_cfg.get("accum_grad_iters", None)
                or self.config.run_cfg.get("gradient_accumulation_steps", None)
                or 1
            )
            amp_dtype = str(
                self.config.run_cfg.get(
                    "amp_dtype",
                    self.config.model_cfg.get("decoder_dtype", "float16"),
                )
            )

            stats = self.task.train_epoch(
                epoch=epoch,
                model=self.model,
                data_loader=self.train_loader,
                optimizer=self.optimizer,
                scaler=self.scaler,
                amp_enabled=self._as_bool(self.config.run_cfg.get("amp", False), default=False),
                amp_dtype=amp_dtype,
                lr_scheduler=self.lr_scheduler,
                cuda_enabled=self.device.type == "cuda",
                log_freq=self.config.run_cfg.get("log_freq", 50),
                accum_grad_iters=int(accum),
            )

            self.log_stats(stats, "train")

            ckpt_cfg = self.config.run_cfg.get("checkpoint", {})
            save_freq = int(ckpt_cfg.get("save_freq", 1))
            save_last = self._as_bool(ckpt_cfg.get("save_last", True), default=True)
            save_best = self._as_bool(ckpt_cfg.get("save_best", False), default=False)
            best_metric_name = str(ckpt_cfg.get("best_metric", "loss"))
            best_mode = str(ckpt_cfg.get("best_mode", "min")).lower()

            is_last_epoch = epoch == int(self.config.run_cfg.max_epoch) - 1
            should_save = (save_freq > 0 and (epoch % save_freq == 0)) or (save_last and is_last_epoch)

            best_improved = False
            if save_best:
                metric_val = self._safe_float(stats.get(best_metric_name, None))
                if metric_val is not None:
                    if self.best_metric_value is None:
                        best_improved = True
                    elif best_mode == "max":
                        best_improved = metric_val > self.best_metric_value
                    else:
                        best_improved = metric_val < self.best_metric_value

                    if best_improved:
                        self.best_metric_value = metric_val

            if is_main_process():
                if should_save:
                    self._save_checkpoint(epoch)
                if save_best and best_improved:
                    self._save_checkpoint(
                        epoch,
                        ckpt_name="checkpoint_best.pth",
                        keep_history=False,
                    )
                    logging.info(
                        f"[BestCkpt] Updated best {best_metric_name}={self.best_metric_value:.6f} "
                        f"at epoch={epoch}"
                    )

        logging.info("Training finished.")

    @staticmethod
    def _safe_float(v):
        if v is None:
            return None
        if isinstance(v, (float, int)):
            return float(v)
        try:
            return float(str(v))
        except Exception:
            return None

    def _maybe_apply_gradual_unfreeze(self, epoch: int):
        """
        Apply gradual-unfreeze scheduling before each epoch if the model exposes it.
        Rebuild optimizer/scheduler when the set of trainable parameters changes.
        """
        model_no_ddp = self._model
        fn = getattr(model_no_ddp, "apply_gradual_unfreeze", None)
        if fn is None:
            return

        changed = bool(fn(epoch))
        if not changed:
            return

        logging.info(f"[GradualUnfreeze] stage changed at epoch={epoch}; rebuilding optimizer/lr_scheduler.")

        # Distributed wrappers need rebuilding after requires_grad changes so new params are synced.
        if self.use_distributed:
            self._wrapped_model = None
            _ = self.model

        self._optimizer = None
        self._lr_sched = None

        try:
            self._print_trainable_params()
        except Exception as e:
            logging.warning(f"Failed to print trainable params after unfreeze: {e}")

    # ------------------------------------------------------------------
    # checkpoint
    # ------------------------------------------------------------------
    def _save_checkpoint(self, cur_epoch: int, ckpt_name: str = None, keep_history: bool = True):
        """
        Save checkpoint with configurable size control:
          - save_trainable_only: only trainable parameter tensors in model state.
          - max_to_keep: keep only latest N checkpoints in current run directory.
          - atomic write: save to temp path then replace target.
        """
        model_no_ddp = self.unwrap_dist_model(self.model)
        ckpt_cfg = self.config.run_cfg.get("checkpoint", {})

        save_trainable_only = bool(
            self.config.run_cfg.get(
                "save_trainable_only",
                ckpt_cfg.get("save_trainable_only", True),
            )
        )
        save_trainable_only = self._as_bool(save_trainable_only, default=True)
        save_optimizer = bool(
            self.config.run_cfg.get(
                "save_optimizer",
                ckpt_cfg.get("save_optimizer", True),
            )
        )
        save_optimizer = self._as_bool(save_optimizer, default=True)
        save_scaler = bool(
            self.config.run_cfg.get(
                "save_scaler",
                ckpt_cfg.get("save_scaler", True),
            )
        )
        save_scaler = self._as_bool(save_scaler, default=True)
        max_to_keep = int(
            self.config.run_cfg.get(
                "max_to_keep",
                ckpt_cfg.get("max_to_keep", 1),
            )
        )

        full_state = model_no_ddp.state_dict()
        if save_trainable_only:
            trainable_keys = {n for n, p in model_no_ddp.named_parameters() if p.requires_grad}
            model_state = {k: v for k, v in full_state.items() if k in trainable_keys}
            mode = "trainable_only"
        else:
            model_state = full_state
            mode = "full"

        optimizer_state = None
        if save_optimizer:
            optimizer_state = self.optimizer.state_dict()

        save_obj = {
            "model": model_state,
            "optimizer": optimizer_state,
            "scaler": self.scaler.state_dict() if (save_scaler and self.scaler) else None,
            "epoch": cur_epoch,
            "config": self.config.to_dict(),
            "checkpoint_mode": mode,
        }

        if ckpt_name is None:
            save_path = os.path.join(self.output_dir, f"checkpoint_epoch_{cur_epoch}.pth")
        else:
            save_path = os.path.join(self.output_dir, ckpt_name)
        tmp_path = save_path + ".tmp"
        logging.info(
            f"Saving {mode} checkpoint to {save_path} "
            f"(keys={len(save_obj['model'])}, save_optimizer={save_optimizer}, save_scaler={save_scaler})"
        )

        try:
            torch.save(save_obj, tmp_path)
            os.replace(tmp_path, save_path)
        except Exception:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            raise

        if keep_history and max_to_keep > 0:
            ckpts = glob.glob(os.path.join(self.output_dir, "checkpoint_epoch_*.pth"))
            ckpts = sorted(ckpts, key=lambda p: int(os.path.basename(p).split("_")[-1].split(".")[0]))
            stale_ckpts = ckpts[:-max_to_keep]
            for p in stale_ckpts:
                try:
                    os.remove(p)
                    logging.info(f"Removed old checkpoint: {p}")
                except Exception as e:
                    logging.warning(f"Failed to remove old checkpoint {p}: {e}")

    # ------------------------------------------------------------------
    # logging
    # ------------------------------------------------------------------
    @main_process
    def log_stats(self, stats, split_name):
        with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
            f.write(json.dumps({f"{split_name}_{k}": v for k, v in stats.items()}) + "\n")

    @main_process
    def log_config(self):
        with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
            f.write(json.dumps(self.config.to_dict(), indent=4) + "\n")

    @staticmethod
    def _as_bool(value, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "f", "no", "n", "off", "none", "null", ""}:
            return False
        return bool(value)
