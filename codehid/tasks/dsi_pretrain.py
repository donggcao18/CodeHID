import logging
import random

import torch
import torch.distributed as dist

from codehid.common.registry import registry
from codehid.tasks.base_task import BaseTask


def _as_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(v)


@registry.register_task("dsi_pretrain")
class DSIPretrainTask(BaseTask):
    """
    DSI-style generative retrieval pretraining task.

    Dataset (via BaseTask.build_datasets):
        datasets = {
            "wiki": {
                "train": Dataset
            }
        }

    Each dataset sample contains:
        - query
        - code
        - answer_id
    """

    def __init__(
        self,
        dual_phase_guidance_train_enable=False,
        dual_phase_guidance_train_lambda_lex=0.0,
        dual_phase_guidance_train_lambda_rank_distill=0.0,
        dual_phase_guidance_train_hard_neg_k=4,
        dual_phase_guidance_train_mine_topk=50,
        dual_phase_guidance_train_mine_every_n_epoch=1,
        dual_phase_guidance_train_mine_batch_size=64,
        dual_phase_guidance_train_start_epoch=0,
    ):
        super().__init__()
        self.dual_phase_guidance_train_enable = _as_bool(dual_phase_guidance_train_enable)
        self.dual_phase_guidance_train_lambda_lex = float(dual_phase_guidance_train_lambda_lex)
        self.dual_phase_guidance_train_lambda_rank_distill = float(dual_phase_guidance_train_lambda_rank_distill)
        self.dual_phase_guidance_train_hard_neg_k = max(1, int(dual_phase_guidance_train_hard_neg_k))
        self.dual_phase_guidance_train_mine_topk = max(2, int(dual_phase_guidance_train_mine_topk))
        self.dual_phase_guidance_train_mine_every_n_epoch = max(1, int(dual_phase_guidance_train_mine_every_n_epoch))
        self.dual_phase_guidance_train_mine_batch_size = max(1, int(dual_phase_guidance_train_mine_batch_size))
        self.dual_phase_guidance_train_start_epoch = int(dual_phase_guidance_train_start_epoch)
        self._last_mined_epoch = -1

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg
        return cls(
            dual_phase_guidance_train_enable=run_cfg.get("dual_phase_guidance_train_enable", False),
            dual_phase_guidance_train_lambda_lex=run_cfg.get("dual_phase_guidance_train_lambda_lex", 0.0),
            dual_phase_guidance_train_lambda_rank_distill=run_cfg.get("dual_phase_guidance_train_lambda_rank_distill", 0.0),
            dual_phase_guidance_train_hard_neg_k=run_cfg.get("dual_phase_guidance_train_hard_neg_k", 4),
            dual_phase_guidance_train_mine_topk=run_cfg.get("dual_phase_guidance_train_mine_topk", 50),
            dual_phase_guidance_train_mine_every_n_epoch=run_cfg.get("dual_phase_guidance_train_mine_every_n_epoch", 1),
            dual_phase_guidance_train_mine_batch_size=run_cfg.get("dual_phase_guidance_train_mine_batch_size", 64),
            dual_phase_guidance_train_start_epoch=run_cfg.get("dual_phase_guidance_train_start_epoch", 0),
        )

    @staticmethod
    def _to_list(x):
        if x is None:
            return []
        if isinstance(x, (list, tuple)):
            return list(x)
        return [x]

    def _build_encoder_texts(self, samples):
        queries = self._to_list(samples.get("query"))
        codes = self._to_list(samples.get("code"))

        assert len(queries) == len(codes), (
            f"query/code size mismatch: {len(queries)} vs {len(codes)}"
        )

        return [
            f"Query: {q}\nCode:\n{c}"
            for q, c in zip(queries, codes)
        ]

    def train_step(self, model, samples):
        """
        Called by BaseTask.train_epoch as:
            loss = self.train_step(model, samples)
        and expected to return a scalar loss.
        """

        # samples already pass through prepare_sample(cuda_enabled=...)
        samples = dict(samples)

        samples["encoder_texts"] = self._build_encoder_texts(samples)
        samples["answer_id"] = self._to_list(samples.get("answer_id"))
        samples["hard_negative_answer_ids"] = self._to_list(samples.get("hard_negative_answer_ids"))
        samples["hard_negative_codes"] = self._to_list(samples.get("hard_negative_codes"))
        samples["rank_teacher_scores"] = self._to_list(samples.get("rank_teacher_scores"))
        samples["dual_phase_guidance_train_lambda_lex"] = self.dual_phase_guidance_train_lambda_lex
        samples["dual_phase_guidance_train_lambda_rank_distill"] = self.dual_phase_guidance_train_lambda_rank_distill
        samples["dual_phase_guidance_train_enable"] = self.dual_phase_guidance_train_enable

        out = model(samples)

        # Support different model output styles.
        if isinstance(out, dict):
            if "loss" in out:
                return out["loss"]
            if hasattr(out, "loss") and out.loss is not None:
                return out.loss
            raise KeyError("Model output dict-like object has no 'loss' key/attr.")
        return out

    def valid_step(self, model, samples):
        return self.train_step(model, samples)

    def _unwrap_model(self, model):
        return model.module if hasattr(model, "module") else model

    def _unwrap_dataset(self, data_loader):
        obj = data_loader
        visited = set()
        for _ in range(8):
            oid = id(obj)
            if oid in visited:
                break
            visited.add(oid)

            if hasattr(obj, "dataset"):
                return obj.dataset
            if hasattr(obj, "loader"):
                obj = obj.loader
                continue
            if hasattr(obj, "_dataloader"):
                obj = obj._dataloader
                continue
            break
        return None

    def _build_random_negative_payload(self, dataset, epoch):
        samples = list(getattr(dataset, "samples", []) or [])
        if len(samples) == 0:
            return {}

        docid_to_code = dict(getattr(dataset, "docid_to_code", {}) or {})
        if not docid_to_code:
            for s in samples:
                did = str(s.get("docid", "")).strip()
                code = s.get("code", "")
                if did:
                    docid_to_code[did] = code
        docids = list(docid_to_code.keys())
        num_docs = len(docids)
        if num_docs <= 1:
            return {}
        docid_to_idx = {d: i for i, d in enumerate(docids)}

        rng = random.Random(2024 + int(epoch))
        docid_to_answer_id = dict(getattr(dataset, "docid_to_answer_id", {}) or {})
        payload = {}
        for sample_idx, s in enumerate(samples):
            gold_docid = str(s.get("docid", "")).strip()
            if not gold_docid:
                continue
            gold_idx = docid_to_idx.get(gold_docid, -1)
            if gold_idx < 0:
                continue
            take = min(self.dual_phase_guidance_train_hard_neg_k, num_docs - 1)
            if take <= 0:
                continue

            # Fast path: sample doc indices directly, avoid O(num_docs) candidate
            # list construction per sample.
            neg_indices = []
            chosen = set()
            max_trials = max(16, take * 8)
            trials = 0
            while len(neg_indices) < take and trials < max_trials:
                trials += 1
                j = rng.randrange(num_docs)
                if j == gold_idx or j in chosen:
                    continue
                chosen.add(j)
                neg_indices.append(j)

            # Deterministic fill if rejection sampling is unlucky.
            if len(neg_indices) < take:
                for off in range(1, num_docs):
                    j = (gold_idx + off) % num_docs
                    if j in chosen:
                        continue
                    chosen.add(j)
                    neg_indices.append(j)
                    if len(neg_indices) >= take:
                        break

            neg_answer_ids = []
            neg_codes = []
            for j in neg_indices:
                d = docids[int(j)]
                neg_answer_ids.append(docid_to_answer_id.get(d, f"<DOCID> {d} </DOCID>"))
                neg_codes.append(docid_to_code[d])

            payload[sample_idx] = {
                "hard_negative_answer_ids": neg_answer_ids,
                "hard_negative_codes": neg_codes,
                # Keep rank distillation branch numerically valid with a simple
                # teacher prior: positive > negatives.
                "rank_teacher_scores": [1.0] + [0.0] * len(neg_indices),
            }
        return payload

    def _mine_hard_negatives(self, model, dataset, epoch):
        if not hasattr(dataset, "samples") or not hasattr(dataset, "set_hard_negative_cache"):
            logging.warning("[dual-phase-guidance-train] dataset does not support hard-negative cache, skip.")
            return

        raw_model = self._unwrap_model(model)
        encode_fn = getattr(raw_model, "encode_texts_for_dual_phase_guidance_train", None)
        if encode_fn is None:
            logging.warning("[dual-phase-guidance-train] model has no dual-phase-guidance-train text-encoder interface, skip.")
            return

        docid_to_code = dict(getattr(dataset, "docid_to_code", {}) or {})
        if not docid_to_code:
            for s in getattr(dataset, "samples", []):
                did = str(s.get("docid", "")).strip()
                code = s.get("code", "")
                if did:
                    docid_to_code[did] = code
        if len(docid_to_code) <= 1:
            logging.warning("[dual-phase-guidance-train] valid doc count <= 1, skip.")
            return

        is_dist = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0

        payload = None
        if rank == 0:
            was_training = raw_model.training
            raw_model.eval()
            try:
                with torch.no_grad():
                    docids = list(docid_to_code.keys())
                    docid_to_idx = {d: i for i, d in enumerate(docids)}
                    doc_texts = [f"Code:\n{docid_to_code[d]}" for d in docids]
                    doc_vecs = encode_fn(
                        doc_texts,
                        batch_size=self.dual_phase_guidance_train_mine_batch_size,
                        normalize=True,
                    )

                    total = len(dataset.samples)
                    queries = [s["query"] for s in dataset.samples]
                    gold_docids = [str(s["docid"]).strip() for s in dataset.samples]
                    payload = {}

                    mine_topk = min(self.dual_phase_guidance_train_mine_topk, len(docids))
                    for st in range(0, total, self.dual_phase_guidance_train_mine_batch_size):
                        ed = min(st + self.dual_phase_guidance_train_mine_batch_size, total)
                        q_texts = [f"Query: {q}\nCode:\n" for q in queries[st:ed]]
                        q_vecs = encode_fn(
                            q_texts,
                            batch_size=self.dual_phase_guidance_train_mine_batch_size,
                            normalize=True,
                        )
                        score_mat = torch.matmul(q_vecs, doc_vecs.t())
                        topk_idx = torch.topk(score_mat, k=mine_topk, dim=1).indices

                        for bi in range(ed - st):
                            sample_idx = st + bi
                            gold_docid = gold_docids[sample_idx]
                            gold_idx = docid_to_idx.get(gold_docid, None)
                            if gold_idx is None:
                                continue

                            row_scores = score_mat[bi]
                            pos_score = float(row_scores[gold_idx].item())

                            neg_docids = []
                            neg_scores = []
                            for doc_idx in topk_idx[bi].tolist():
                                d = docids[int(doc_idx)]
                                if d == gold_docid:
                                    continue
                                neg_docids.append(d)
                                neg_scores.append(float(row_scores[int(doc_idx)].item()))
                                if len(neg_docids) >= self.dual_phase_guidance_train_hard_neg_k:
                                    break

                            neg_answer_ids = []
                            neg_codes = []
                            for d in neg_docids:
                                neg_answer_ids.append(
                                    getattr(dataset, "docid_to_answer_id", {}).get(
                                        d, f"<DOCID> {d} </DOCID>"
                                    )
                                )
                                neg_codes.append(docid_to_code[d])

                            payload[sample_idx] = {
                                "hard_negative_answer_ids": neg_answer_ids,
                                "hard_negative_codes": neg_codes,
                                "rank_teacher_scores": [pos_score] + neg_scores,
                            }
            finally:
                raw_model.train(was_training)

        if is_dist:
            obj_list = [payload]
            dist.broadcast_object_list(obj_list, src=0)
            payload = obj_list[0]

        dataset.set_hard_negative_cache(payload)
        if rank == 0:
            n_assigned = len(payload) if payload is not None else 0
            logging.info(
                "[dual-phase-guidance-train] epoch=%d mined hard negatives for %d samples (neg_k=%d, topk=%d).",
                epoch,
                n_assigned,
                self.dual_phase_guidance_train_hard_neg_k,
                self.dual_phase_guidance_train_mine_topk,
            )

    def train_epoch(
        self,
        epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        amp_enabled=False,
        amp_dtype="float16",
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
        should_mine = (
            self.dual_phase_guidance_train_enable
            and epoch >= self.dual_phase_guidance_train_start_epoch
            and (epoch - self.dual_phase_guidance_train_start_epoch) % self.dual_phase_guidance_train_mine_every_n_epoch == 0
            and self._last_mined_epoch != epoch
        )
        if should_mine:
            dataset = self._unwrap_dataset(data_loader)
            if dataset is None:
                logging.warning("[dual-phase-guidance-train] cannot unwrap dataset from data_loader, skip mining.")
            else:
                self._mine_hard_negatives(model=model, dataset=dataset, epoch=epoch)
            self._last_mined_epoch = epoch

        return super().train_epoch(
            epoch=epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            scaler=scaler,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            cuda_enabled=cuda_enabled,
            log_freq=log_freq,
            accum_grad_iters=accum_grad_iters,
        )
