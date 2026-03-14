from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple
import os

from core.io import read_jsonl, write_jsonl
from core.records import extract_model_outputs, get_record_id, normalize_record


Score = Any


@dataclass
class EvalResult:
    scored_entries: List[dict]
    total: int


class BaseEvaluator:
    name: str = "base"
    id_key: str = "id"

    def __init__(self, dataset: Any = None) -> None:
        self.dataset = dataset

    def _get_outputs(self, record: dict) -> List[str]:
        outputs = extract_model_outputs(record)
        if outputs:
            return outputs
        fallback = record.get("code")
        if fallback is None:
            fallback = record.get("model_output")
        if fallback is None:
            return []
        if isinstance(fallback, list):
            return [str(v) for v in fallback]
        return [str(fallback)]

    def _get_reference(self, record: dict, idx: int, dataset: Any) -> Any:
        data = dataset if dataset is not None else self.dataset
        if data is not None:
            record_id = get_record_id(record, idx)
            if isinstance(record_id, int):
                try:
                    return data[record_id]
                except Exception:
                    pass
            try:
                return data[idx]
            except Exception:
                pass
        entry = record.get("entry")
        if entry is not None:
            return entry
        doc = record.get("doc")
        if doc is not None:
            return doc
        return record

    def score_entry(self, outputs: List[str], reference: Any, record: dict, idx: int) -> List[Score]:
        raise NotImplementedError

    def _format_scored_entry(self, record_id: Any, scores: List[Score]) -> dict:
        payload = {"is_correct": scores}
        if self.id_key:
            payload[self.id_key] = record_id
        if self.id_key != "id":
            payload["id"] = record_id
        if self.id_key != "idx":
            payload["idx"] = record_id
        return payload

    def evaluate_file(
        self,
        input_path: str,
        scored_path: str,
        *,
        dataset: Any = None,
        skip_if_exists: bool = True,
    ) -> Optional[EvalResult]:
        if skip_if_exists and os.path.exists(scored_path):
            return None
        records = read_jsonl(input_path, allow_errors=True)
        scored_entries: List[dict] = []
        for idx, record in enumerate(records):
            normalized = normalize_record(record, idx)
            outputs = normalized.model_outputs or self._get_outputs(record)
            reference = self._get_reference(record, idx, dataset)
            scores = self.score_entry(outputs, reference, record, idx)
            scored_entries.append(self._format_scored_entry(normalized.idx, scores))
        write_jsonl(scored_path, scored_entries, mode="w")
        return EvalResult(scored_entries=scored_entries, total=len(scored_entries))


class GpqaEvaluator(BaseEvaluator):
    name = "gpqa"
    id_key = "id"

    def __init__(self, dataset: Sequence[Any]) -> None:
        super().__init__(dataset)

    def score_entry(self, outputs: List[str], reference: Any, record: dict, idx: int) -> List[Score]:
        from tasks.gpqa.gpqa_utils import LETTER_TO_INDEX, parse_sampled_answer

        correct_index = None
        if hasattr(reference, "correct_index"):
            correct_index = reference.correct_index
        elif isinstance(reference, dict):
            correct_index = reference.get("correct_index")
        if correct_index is None:
            return [False for _ in outputs]
        results = []
        for output in outputs:
            pred = parse_sampled_answer(output)
            is_correct = pred is not None and LETTER_TO_INDEX.get(pred) == correct_index
            results.append(bool(is_correct))
        return results


class DropEvaluator(BaseEvaluator):
    name = "drop"
    id_key = "id"

    def score_entry(self, outputs: List[str], reference: Any, record: dict, idx: int) -> List[Score]:
        from tasks.drop.drop_utils import extract_answer, get_answers, get_metrics

        entry = reference if isinstance(reference, dict) else record.get("entry") or record.get("doc") or {}
        golds = get_answers(entry) if entry else []
        results: List[Tuple[int, float]] = []
        for output in outputs:
            pred = extract_answer(output)
            max_em, max_f1 = 0.0, 0.0
            for gold_answer in golds:
                exact_match, f1_score = get_metrics(pred, gold_answer)
                if gold_answer and gold_answer[0].strip():
                    max_em = max(max_em, exact_match)
                    max_f1 = max(max_f1, f1_score)
            results.append((int(max_em), float(max_f1)))
        return results


class HotpotqaEvaluator(BaseEvaluator):
    name = "hotpotqa"
    id_key = "id"

    def score_entry(self, outputs: List[str], reference: Any, record: dict, idx: int) -> List[Score]:
        from tasks.hotpotqa.hotpotqa_utils import extract_answer, get_em_f1

        entry = reference if reference is not None else record.get("entry") or record.get("doc") or {}
        results: List[Tuple[int, float]] = []
        for output in outputs:
            pred = extract_answer(output)
            em, f1 = get_em_f1([entry], [pred])
            results.append((int(em[0]), float(f1[0])))
        return results


class MathEvaluator(BaseEvaluator):
    name = "math"
    id_key = "idx"

    def __init__(self, dataset: Sequence[Any], task: str) -> None:
        super().__init__(dataset)
        self.task = task

    def score_entry(self, outputs: List[str], reference: Any, record: dict, idx: int) -> List[Score]:
        from tasks.math500.grader import math_equal_process
        from tasks.math500.math_utils import process_results
        from tasks.math500.parser import extract_answer, parse_ground_truth

        if not outputs:
            outputs = self._get_outputs(record)
        if reference is None:
            return [False for _ in outputs]
        _, gt = parse_ground_truth(reference, self.task)
        results: List[bool] = []
        for output in outputs:
            pred = extract_answer(output, self.task)
            is_correct = math_equal_process((None, pred, gt))
            if not is_correct:
                is_correct = process_results(gt, [output])
                if not is_correct:
                    fallback_pred = extract_answer(pred, "math")
                    is_correct = math_equal_process((None, fallback_pred, gt))
            results.append(bool(is_correct))
        return results


class MusrEvaluator(BaseEvaluator):
    name = "musr"
    id_key = "idx"

    def __init__(self, dataset: Any, evaluator_fn: Optional[Callable[[List[str], Any], Any]] = None) -> None:
        super().__init__(dataset)
        self.evaluator_fn = evaluator_fn or getattr(dataset, "evaluate_response", None)
        if self.evaluator_fn is None:
            raise ValueError("MuSR evaluator requires a dataset with evaluate_response or a custom evaluator.")

    def score_entry(self, outputs: List[str], reference: Any, record: dict, idx: int) -> List[Score]:
        if reference is None:
            return [False for _ in outputs]
        results: List[bool] = []
        for output in outputs:
            metrics = self.evaluator_fn([output], reference)
            is_correct = bool(metrics[0].get("correct")) if metrics else False
            results.append(is_correct)
        return results


class MmluProEvaluator(BaseEvaluator):
    name = "mmlu_pro"
    id_key = "idx"

    def __init__(self, extract_answer_fn: Callable[[str], Optional[str]]) -> None:
        super().__init__(None)
        self.extract_answer = extract_answer_fn

    def _get_gold(self, record: dict) -> Optional[str]:
        gold = record.get("doc", {}).get("gold")
        if gold is None:
            gold = record.get("entry", {}).get("answer")
        if gold is None:
            gold = record.get("answer")
        return gold

    def score_entry(self, outputs: List[str], reference: Any, record: dict, idx: int) -> List[Score]:
        gold = self._get_gold(record)
        if gold is None:
            return [False for _ in outputs]
        preds = [self.extract_answer(output) for output in outputs]
        return [pred == gold for pred in preds]

    def evaluate_subject(
        self,
        raw_path: str,
        scored_path: str,
        *,
        skip_if_exists: bool = True,
    ) -> Optional[List[float]]:
        result = self.evaluate_file(raw_path, scored_path, skip_if_exists=skip_if_exists)
        if result is None:
            scored_entries = read_jsonl(scored_path, allow_errors=True)
        else:
            scored_entries = result.scored_entries
        if not scored_entries:
            return None
        n_repeats = min(
            len(e.get("is_correct", []))
            for e in scored_entries
            if isinstance(e.get("is_correct"), list)
        )
        if n_repeats == 0:
            return None
        counts = [0.0] * n_repeats
        for entry in scored_entries:
            res = entry.get("is_correct", [])
            for i in range(min(n_repeats, len(res))):
                counts[i] += 1.0 if res[i] else 0.0
        total = len(scored_entries)
        return [count / total for count in counts]
