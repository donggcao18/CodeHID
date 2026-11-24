import logging
import os
import random
from time import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

from utils import ensure_dir, set_color, get_local_time, delete_file

import heapq


class Trainer(object):

    def __init__(self, args, model, data_num):
        self.args = args
        self.model = model
        self.logger = logging.getLogger()

        self.lr = args.lr
        self.learner = args.learner
        self.lr_scheduler_type = args.lr_scheduler_type

        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.warmup_steps = args.warmup_epochs * data_num
        self.max_steps = args.epochs * data_num

        self.save_limit = args.save_limit
        self.best_save_heap = []
        self.newest_save_queue = []
        self.eval_step = min(args.eval_step, self.epochs)
        self.device = args.device
        self.device = torch.device(self.device)
        self.ckpt_dir = args.ckpt_dir
        saved_model_dir = "{}".format(get_local_time())
        self.ckpt_dir = os.path.join(self.ckpt_dir,saved_model_dir)
        ensure_dir(self.ckpt_dir)

        self.best_loss = np.inf
        self.best_collision_rate = np.inf
        self.best_loss_ckpt = "best_loss_model.pth"
        self.best_collision_ckpt = "best_collision_model.pth"
        self.optimizer = self._build_optimizer()
        self.scheduler = self._get_scheduler()
        self.model = self.model.to(self.device)
        self._init_knn_prefix_config()

    def _init_knn_prefix_config(self):
        self.use_knn_prefix_loss = bool(getattr(self.args, "use_knn_prefix_loss", False))
        self.knn_loss_weight_base = float(getattr(self.args, "knn_loss_weight", 0.0))
        self.knn_loss_weight = self.knn_loss_weight_base
        self.knn_loss_schedule = str(getattr(self.args, "knn_loss_schedule", "constant"))
        self.knn_decay_start_ratio = float(getattr(self.args, "knn_decay_start_ratio", 0.3))
        self.knn_decay_end_ratio = float(getattr(self.args, "knn_decay_end_ratio", 0.7))
        self.knn_temperature = max(float(getattr(self.args, "knn_temperature", 0.2)), 1e-6)
        self.knn_k = int(getattr(self.args, "knn_k", 16))
        self.knn_num_negatives = int(getattr(self.args, "knn_num_negatives", 2))
        self.knn_negative_weight = float(getattr(self.args, "knn_negative_weight", 0.2))
        self.knn_auto_thresholds = bool(getattr(self.args, "knn_auto_thresholds", False))
        self.knn_indices = None
        self.knn_sims = None
        self.knn_prefix_weights = None
        self.knn_thresholds = None

        if not self.use_knn_prefix_loss:
            return
        levels = len(self.args.num_emb_list)
        if self.knn_loss_schedule not in {"constant", "linear_decay_to_zero"}:
            raise ValueError(
                "knn_loss_schedule must be one of {'constant', 'linear_decay_to_zero'}, "
                f"got {self.knn_loss_schedule}"
            )
        if not (0.0 <= self.knn_decay_start_ratio <= 1.0 and 0.0 <= self.knn_decay_end_ratio <= 1.0):
            raise ValueError(
                "knn_decay_start_ratio and knn_decay_end_ratio must be in [0, 1], "
                f"got start={self.knn_decay_start_ratio}, end={self.knn_decay_end_ratio}"
            )
        if self.knn_decay_end_ratio < self.knn_decay_start_ratio:
            raise ValueError(
                "knn_decay_end_ratio must be >= knn_decay_start_ratio, "
                f"got start={self.knn_decay_start_ratio}, end={self.knn_decay_end_ratio}"
            )

        knn_indices_path = getattr(self.args, "knn_indices_path", None)
        if not knn_indices_path or not os.path.exists(knn_indices_path):
            raise FileNotFoundError(f"KNN indices file not found: {knn_indices_path}")
        self.knn_indices = np.load(knn_indices_path).astype(np.int64)
        if self.knn_indices.ndim != 2:
            raise ValueError(f"knn_indices must be 2D [N, K], got shape={self.knn_indices.shape}")
        self.knn_k = min(self.knn_k, self.knn_indices.shape[1])

        knn_sims_path = getattr(self.args, "knn_sims_path", None)
        if knn_sims_path and os.path.exists(knn_sims_path):
            self.knn_sims = np.load(knn_sims_path).astype(np.float32)
            if self.knn_sims.shape != self.knn_indices.shape:
                raise ValueError(
                    "knn_sims shape must match knn_indices shape, "
                    f"got {self.knn_sims.shape} vs {self.knn_indices.shape}"
                )

        prefix_weights = list(getattr(self.args, "knn_prefix_weights", []))
        thresholds = list(getattr(self.args, "knn_thresholds", []))
        threshold_quantiles = list(getattr(self.args, "knn_threshold_quantiles", []))
        if len(prefix_weights) != levels:
            raise ValueError(f"knn_prefix_weights length must be {levels}, got {len(prefix_weights)}")
        if self.knn_auto_thresholds:
            if len(threshold_quantiles) != levels:
                raise ValueError(
                    f"knn_threshold_quantiles length must be {levels}, got {len(threshold_quantiles)}"
                )
            if self.knn_sims is None:
                raise ValueError("knn_auto_thresholds requires knn_sims_path to be provided.")
            flat = self.knn_sims.reshape(-1)
            thresholds = [float(np.quantile(flat, q)) for q in threshold_quantiles]
        else:
            if len(thresholds) != levels:
                raise ValueError(f"knn_thresholds length must be {levels}, got {len(thresholds)}")

        self.knn_prefix_weights = torch.tensor(prefix_weights, dtype=torch.float32, device=self.device)
        self.knn_thresholds = torch.tensor(thresholds, dtype=torch.float32, device=self.device)
        graph_ref = getattr(self.args, "knn_indices_path", None)
        self.logger.info(
            "Enable KNN prefix loss: "
            f"k={self.knn_k}, lambda={self.knn_loss_weight_base}, schedule={self.knn_loss_schedule}, "
            f"decay=({self.knn_decay_start_ratio:.2f}->{self.knn_decay_end_ratio:.2f}), "
            f"temperature={self.knn_temperature}, "
            f"neg={self.knn_num_negatives}, neg_w={self.knn_negative_weight}, "
            f"auto_thresholds={self.knn_auto_thresholds}, thresholds={thresholds}, "
            f"indices={graph_ref}"
        )

    def _build_optimizer(self):

        params = self.model.parameters()
        learner =  self.learner
        learning_rate = self.lr
        weight_decay = self.weight_decay

        if learner.lower() == "adam":
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(
                params, lr=learning_rate, weight_decay=weight_decay
            )
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == 'adamw':
            optimizer = optim.AdamW(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        else:
            self.logger.warning(
                "Received unrecognized optimizer, set default Adam optimizer"
            )
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer

    def _get_scheduler(self):
        if self.lr_scheduler_type.lower() == "linear":
            lr_scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                           num_warmup_steps=self.warmup_steps,
                                                           num_training_steps=self.max_steps)
        else:
            lr_scheduler = get_constant_schedule_with_warmup(optimizer=self.optimizer,
                                                             num_warmup_steps=self.warmup_steps)

        return lr_scheduler
    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")

    @staticmethod
    def _unpack_batch(batch):
        if isinstance(batch, (tuple, list)):
            if len(batch) == 2:
                return batch[0], batch[1]
        return batch, None

    def _get_effective_knn_weight(self, epoch_idx):
        if not self.use_knn_prefix_loss:
            return 0.0
        if self.knn_loss_schedule == "constant":
            return self.knn_loss_weight_base

        if self.epochs <= 1:
            return self.knn_loss_weight_base
        progress = float(epoch_idx) / float(self.epochs - 1)
        if progress <= self.knn_decay_start_ratio:
            return self.knn_loss_weight_base
        if progress >= self.knn_decay_end_ratio:
            return 0.0

        span = max(1e-8, self.knn_decay_end_ratio - self.knn_decay_start_ratio)
        ratio = (progress - self.knn_decay_start_ratio) / span
        return self.knn_loss_weight_base * max(0.0, 1.0 - ratio)

    def _compute_knn_prefix_loss(self, doc_ids, soft_probs):
        if (not self.use_knn_prefix_loss) or (doc_ids is None):
            return torch.zeros([], dtype=torch.float32, device=self.device), 0
        if self.knn_indices is None:
            return torch.zeros([], dtype=torch.float32, device=self.device), 0

        doc_ids_np = doc_ids.detach().cpu().numpy().astype(np.int64)
        position = {int(doc_id): idx for idx, doc_id in enumerate(doc_ids_np.tolist())}

        pair_i = []
        pair_j = []
        pair_scores = []
        neg_i = []
        neg_j = []
        max_doc = self.knn_indices.shape[0]
        max_rank = max(1, self.knn_k - 1)

        doc_ids_list = doc_ids_np.tolist()
        for i_pos, doc_id in enumerate(doc_ids_list):
            if doc_id < 0 or doc_id >= max_doc:
                continue
            neighbors = self.knn_indices[doc_id, : self.knn_k]
            sims_row = self.knn_sims[doc_id, : self.knn_k] if self.knn_sims is not None else None
            neighbor_set = set(int(x) for x in neighbors.tolist())
            neighbor_set.add(int(doc_id))
            for rank_idx, nbr_doc in enumerate(neighbors.tolist()):
                j_pos = position.get(int(nbr_doc))
                if j_pos is None or j_pos == i_pos:
                    continue
                pair_i.append(i_pos)
                pair_j.append(j_pos)
                if sims_row is not None:
                    pair_scores.append(float(sims_row[rank_idx]))
                else:
                    pair_scores.append(1.0 - float(rank_idx) / max_rank)

            if self.knn_num_negatives > 0:
                candidate_positions = [
                    j for j, other_doc_id in enumerate(doc_ids_list)
                    if j != i_pos and int(other_doc_id) not in neighbor_set
                ]
                if candidate_positions:
                    take = min(self.knn_num_negatives, len(candidate_positions))
                    sampled = random.sample(candidate_positions, k=take)
                    for j in sampled:
                        neg_i.append(i_pos)
                        neg_j.append(j)

        if not pair_i:
            return torch.zeros([], dtype=torch.float32, device=self.device), 0

        idx_i = torch.tensor(pair_i, dtype=torch.long, device=self.device)
        idx_j = torch.tensor(pair_j, dtype=torch.long, device=self.device)
        scores = torch.tensor(pair_scores, dtype=torch.float32, device=self.device)
        has_neg = len(neg_i) > 0
        if has_neg:
            idx_ni = torch.tensor(neg_i, dtype=torch.long, device=self.device)
            idx_nj = torch.tensor(neg_j, dtype=torch.long, device=self.device)

        prefix_prob = torch.ones(scores.shape[0], dtype=torch.float32, device=self.device)
        if has_neg:
            prefix_prob_neg = torch.ones(len(neg_i), dtype=torch.float32, device=self.device)
        loss_knn = torch.zeros([], dtype=torch.float32, device=self.device)
        for level, probs in enumerate(soft_probs):
            same_prob = torch.sum(probs[idx_i] * probs[idx_j], dim=-1)
            prefix_prob = torch.clamp(prefix_prob * same_prob, min=1e-6, max=1.0 - 1e-6)
            targets = (scores >= self.knn_thresholds[level]).float()
            level_loss = F.binary_cross_entropy(prefix_prob, targets, reduction="mean")
            if has_neg:
                same_prob_neg = torch.sum(probs[idx_ni] * probs[idx_nj], dim=-1)
                prefix_prob_neg = torch.clamp(prefix_prob_neg * same_prob_neg, min=1e-6, max=1.0 - 1e-6)
                neg_targets = torch.zeros_like(prefix_prob_neg)
                neg_loss = F.binary_cross_entropy(prefix_prob_neg, neg_targets, reduction="mean")
                level_loss = level_loss + self.knn_negative_weight * neg_loss
            loss_knn = loss_knn + self.knn_prefix_weights[level] * level_loss

        return loss_knn, len(pair_i) + len(neg_i)

    def _train_epoch(self, train_data, epoch_idx):

        self.model.train()

        total_loss = 0
        total_recon_loss = 0
        total_knn_loss = 0
        total_knn_pairs = 0
        effective_knn_weight = self._get_effective_knn_weight(epoch_idx)
        iter_data = tqdm(
                    train_data,
                    total=len(train_data),
                    ncols=100,
                    desc=set_color(f"Train {epoch_idx}","pink"),
                    )

        for batch_idx, batch in enumerate(iter_data):
            data, doc_ids = self._unpack_batch(batch)
            data = data.to(self.device)
            self.optimizer.zero_grad()
            if self.use_knn_prefix_loss and effective_knn_weight > 0.0:
                out, rq_loss, indices, soft_probs = self.model(
                    data,
                    return_soft_probs=True,
                    temperature=self.knn_temperature,
                )
            else:
                out, rq_loss, indices = self.model(data)
                soft_probs = None

            loss_base, loss_recon = self.model.compute_loss(out, rq_loss, xs=data)
            if self.use_knn_prefix_loss and effective_knn_weight > 0.0:
                loss_knn, pair_cnt = self._compute_knn_prefix_loss(doc_ids, soft_probs)
                loss = loss_base + effective_knn_weight * loss_knn
                total_knn_loss += loss_knn.item()
                total_knn_pairs += pair_cnt
            else:
                loss = loss_base
            self._check_nan(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            # print(self.scheduler.get_last_lr())
            total_loss += loss.item()
            total_recon_loss += loss_recon.item()

        return total_loss, total_recon_loss, total_knn_loss, total_knn_pairs, effective_knn_weight

    @torch.no_grad()
    def _valid_epoch(self, valid_data):

        self.model.eval()

        iter_data =tqdm(
                valid_data,
                total=len(valid_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", "pink"),
            )

        indices_set = set()
        num_sample = 0
        for batch_idx, batch in enumerate(iter_data):
            data, _ = self._unpack_batch(batch)
            num_sample += len(data)
            data = data.to(self.device)
            indices = self.model.get_indices(data)
            indices = indices.view(-1,indices.shape[-1]).cpu().numpy()
            for index in indices:
                code = "-".join([str(int(_)) for _ in index])
                indices_set.add(code)

        collision_rate = (num_sample - len(list(indices_set)))/num_sample

        return collision_rate

    def _save_checkpoint(self, epoch, collision_rate=1, ckpt_file=None):

        ckpt_path = os.path.join(self.ckpt_dir,ckpt_file) if ckpt_file \
            else os.path.join(self.ckpt_dir, 'epoch_%d_collision_%.4f_model.pth' % (epoch, collision_rate))
        state = {
            "args": self.args,
            "epoch": epoch,
            "best_loss": self.best_loss,
            "best_collision_rate": self.best_collision_rate,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, ckpt_path, pickle_protocol=4)

        self.logger.info(
            set_color("Saving current", "blue") + f": {ckpt_path}"
        )

        return ckpt_path

    def _generate_train_loss_output(
        self,
        epoch_idx,
        s_time,
        e_time,
        loss,
        recon_loss,
        knn_loss,
        knn_pairs,
        knn_weight,
    ):
        train_loss_output = (
            set_color("epoch %d training", "green")
            + " ["
            + set_color("time", "blue")
            + ": %.2fs, "
        ) % (epoch_idx, e_time - s_time)
        train_loss_output += set_color("train loss", "blue") + ": %.4f" % loss
        train_loss_output +=", "
        train_loss_output += set_color("reconstruction loss", "blue") + ": %.4f" % recon_loss
        if self.use_knn_prefix_loss:
            train_loss_output += ", "
            train_loss_output += set_color("knn_prefix loss", "blue") + ": %.4f" % knn_loss
            train_loss_output += ", "
            train_loss_output += set_color("knn pairs", "blue") + ": %d" % knn_pairs
            train_loss_output += ", "
            train_loss_output += set_color("knn lambda", "blue") + ": %.6f" % knn_weight
        return train_loss_output + "]"


    def fit(self, data):

        cur_eval_step = 0

        for epoch_idx in range(self.epochs):
            # train
            training_start_time = time()
            train_loss, train_recon_loss, train_knn_loss, train_knn_pairs, train_knn_weight = self._train_epoch(
                data, epoch_idx
            )
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx,
                training_start_time,
                training_end_time,
                train_loss,
                train_recon_loss,
                train_knn_loss,
                train_knn_pairs,
                train_knn_weight,
            )
            self.logger.info(train_loss_output)


            # eval
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                collision_rate = self._valid_epoch(data)

                if train_loss < self.best_loss:
                    self.best_loss = train_loss
                    self._save_checkpoint(epoch=epoch_idx, ckpt_file=self.best_loss_ckpt)

                if collision_rate < self.best_collision_rate:
                    self.best_collision_rate = collision_rate
                    cur_eval_step = 0
                    self._save_checkpoint(epoch_idx, collision_rate=collision_rate,
                                          ckpt_file=self.best_collision_ckpt)
                else:
                    cur_eval_step += 1


                valid_end_time = time()
                valid_score_output = (
                    set_color("epoch %d evaluating", "green")
                    + " ["
                    + set_color("time", "blue")
                    + ": %.2fs, "
                    + set_color("collision_rate", "blue")
                    + ": %f]"
                ) % (epoch_idx, valid_end_time - valid_start_time, collision_rate)

                self.logger.info(valid_score_output)
                ckpt_path = self._save_checkpoint(epoch_idx, collision_rate=collision_rate)
                now_save = (-collision_rate, ckpt_path)
                if len(self.newest_save_queue) < self.save_limit:
                    self.newest_save_queue.append(now_save)
                    heapq.heappush(self.best_save_heap, now_save)
                else:
                    old_save = self.newest_save_queue.pop(0)
                    self.newest_save_queue.append(now_save)
                    if collision_rate < -self.best_save_heap[0][0]:
                        bad_save = heapq.heappop(self.best_save_heap)
                        heapq.heappush(self.best_save_heap, now_save)

                        if bad_save not in self.newest_save_queue:
                            delete_file(bad_save[1])

                    if old_save not in self.best_save_heap:
                        delete_file(old_save[1])



        return self.best_loss, self.best_collision_rate
