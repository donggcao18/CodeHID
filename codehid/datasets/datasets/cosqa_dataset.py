import json
import re
from pathlib import Path

from codehid.datasets.datasets.base_dataset import BaseDataset


# Backward-compatible docid validation:
#   1) cosqa_00001234 / procqa_00001234 / 00001234
#   2) cosqa_a1b2c3d4 / procqa_a1b2c3d4 / a1b2c3d4
DOCID_PATTERN_DIGITS = re.compile(r"^(?:[a-z][a-z0-9]*_)?\d{8}$")
DOCID_PATTERN_LEVELS = re.compile(r"^(?:[a-z][a-z0-9]*_)?(?:[a-z]\d+){2,}$")


class CoSQADSIPretrainDataset(BaseDataset):
    """
    CoSQA Dataset for DSI-style generative retrieval pretraining.

    Each sample provides:
        {
            "query": str,
            "code": str,
            "answer_id": str   # structured docid
        }

    answer_id format (backward compatible):
        <DOCID> cosqa_XXXXXXXX </DOCID>
        <DOCID> 00001234 </DOCID>
        <DOCID> cosqa_a1b2c3d4 </DOCID>
        <DOCID> a1b2c3d4 </DOCID>
    """

    def __init__(self, ann_path):
        super().__init__()

        self.ann_path = Path(ann_path)
        assert self.ann_path.exists(), f"Annotation file not found: {self.ann_path}"

        self.samples = []
        self.docid_to_code = {}
        self.docid_to_answer_id = {}
        self._hard_negative_answer_ids = {}
        self._hard_negative_codes = {}
        self._rank_teacher_scores = {}
        self._load_annotations()

    def _load_annotations(self):
        """
        Load jsonl annotations into memory with strict validation.
        """
        dropped = 0

        with self.ann_path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                data = json.loads(line)

                # ---- required fields ----
                assert "query" in data, f"Missing 'query' at line {line_idx}"
                assert "code" in data, f"Missing 'code' at line {line_idx}"
                assert "docid" in data, f"Missing 'docid' at line {line_idx}"

                docid = data["docid"].strip()

                # ---- strict docid validation ----
                if not (DOCID_PATTERN_DIGITS.match(docid) or DOCID_PATTERN_LEVELS.match(docid)):
                    dropped += 1
                    continue

                # ---- structured supervision target ----
                answer_id = f"<DOCID> {docid} </DOCID>"

                self.samples.append(
                    {
                        "query": data["query"],
                        "code": data["code"],
                        "docid": docid,
                        "answer_id": answer_id,
                    }
                )
                self.docid_to_code[docid] = data["code"]
                self.docid_to_answer_id[docid] = answer_id

        if dropped > 0:
            print(
                f"[CoSQADSIPretrainDataset] Dropped {dropped} samples due to invalid docid format."
            )

        assert len(self.samples) > 0, "No valid samples loaded after docid filtering."

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        s = dict(self.samples[index])
        s["sample_idx"] = int(index)
        s["hard_negative_answer_ids"] = list(self._hard_negative_answer_ids.get(index, []))
        s["hard_negative_codes"] = list(self._hard_negative_codes.get(index, []))
        s["rank_teacher_scores"] = list(self._rank_teacher_scores.get(index, []))
        return s

    def set_hard_negative_cache(self, payload):
        """
        payload[index] = {
            "hard_negative_answer_ids": List[str],
            "hard_negative_codes": List[str],
            "rank_teacher_scores": List[float],  # [pos, neg1, neg2, ...]
        }
        """
        self._hard_negative_answer_ids = {}
        self._hard_negative_codes = {}
        self._rank_teacher_scores = {}

        if payload is None:
            return

        for k, v in payload.items():
            idx = int(k)
            item = v or {}
            self._hard_negative_answer_ids[idx] = list(item.get("hard_negative_answer_ids", []))
            self._hard_negative_codes[idx] = list(item.get("hard_negative_codes", []))
            self._rank_teacher_scores[idx] = [float(x) for x in item.get("rank_teacher_scores", [])]

    @staticmethod
    def collater(batch):
        """
        Collate function for DSI pretraining.
        """
        return {
            "query": [b["query"] for b in batch],
            "code": [b["code"] for b in batch],
            "docid": [b["docid"] for b in batch],
            "answer_id": [b["answer_id"] for b in batch],
            "sample_idx": [int(b["sample_idx"]) for b in batch],
            "hard_negative_answer_ids": [b.get("hard_negative_answer_ids", []) for b in batch],
            "hard_negative_codes": [b.get("hard_negative_codes", []) for b in batch],
            "rank_teacher_scores": [b.get("rank_teacher_scores", []) for b in batch],
        }
