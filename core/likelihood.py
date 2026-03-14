from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

from core.modeling import calculate_cross_entropy_loss_with_topk
from core.records import extract_model_outputs, get_record_id
from core.io import read_json, write_json


@dataclass
class PromptSet:
    no_replace_prompt: str
    replace_prompt: str
    zero_prompt: str
    answer: str
    no_key: str = "no_replace_likelihoods"
    replace_key: str = "replace_likelihoods"
    zero_key: str = "zero_likelihoods"


@dataclass
class LikelihoodResult:
    likelihoods: List[List[Dict[str, Any]]]
    gpu_time: Dict[str, float]
    gpu_calls: int


def _count_candidates(records: List[Dict[str, Any]]) -> int:
    if not records:
        return 0
    outputs = extract_model_outputs(records[0])
    return len(outputs)


def load_existing_likelihoods(path: str) -> List[List[Dict[str, Any]]]:
    data = read_json(path, default=[])
    if not isinstance(data, list):
        return []
    return data


def init_likelihoods(existing: List[List[Dict[str, Any]]], n: int) -> List[List[Dict[str, Any]]]:
    if existing and isinstance(existing, list):
        return existing
    return [[] for _ in range(n)]


def run_likelihood(
    records: List[Dict[str, Any]],
    build_prompt_sets: Callable[[Dict[str, Any], str, int], Iterable[PromptSet]],
    record_factory: Callable[[Any, str, Any, int], Dict[str, Any]],
    get_answer: Callable[[Dict[str, Any], int], Any],
    *,
    model,
    tokenizer,
    device,
    likelihoods_path: str,
    truncate_output_len: int = 5000,
    existing_likelihoods: Optional[List[List[Dict[str, Any]]]] = None,
    show_progress: bool = True,
) -> LikelihoodResult:
    gpu_time = {"no": 0.0, "rep": 0.0, "zero": 0.0}
    gpu_calls = 0

    n = _count_candidates(records)
    likelihoods = init_likelihoods(existing_likelihoods or [], n)

    iterator = tqdm(records, total=len(records), desc="likelihood", disable=not show_progress)
    for idx, entry in enumerate(iterator):
        doc_id = get_record_id(entry, idx)
        id_exists = any(item.get("id") == doc_id for item in likelihoods[0]) if likelihoods else False
        if id_exists:
            continue

        outputs = extract_model_outputs(entry)
        if not outputs:
            continue

        answer = get_answer(entry, idx)

        for output_idx, model_output in enumerate(outputs):
            full_output = model_output
            prompt_output = model_output
            if isinstance(model_output, str) and len(model_output) > truncate_output_len:
                prompt_output = model_output[:truncate_output_len]

            current = record_factory(doc_id, full_output, answer, idx)

            for prompt_set in build_prompt_sets(entry, prompt_output, idx):
                no_res, dt_no = calculate_cross_entropy_loss_with_topk(
                    prompt_set.no_replace_prompt, prompt_set.answer, model, tokenizer, device
                )
                rep_res, dt_rep = calculate_cross_entropy_loss_with_topk(
                    prompt_set.replace_prompt, prompt_set.answer, model, tokenizer, device
                )
                zero_res, dt_zero = calculate_cross_entropy_loss_with_topk(
                    prompt_set.zero_prompt, prompt_set.answer, model, tokenizer, device
                )

                current.setdefault(prompt_set.no_key, []).append(no_res)
                current.setdefault(prompt_set.replace_key, []).append(rep_res)
                current.setdefault(prompt_set.zero_key, []).append(zero_res)

                gpu_time["no"] += dt_no
                gpu_time["rep"] += dt_rep
                gpu_time["zero"] += dt_zero
                gpu_calls += 3

            likelihoods[output_idx].append(current)

    write_json(likelihoods_path, likelihoods, indent=4)

    return LikelihoodResult(likelihoods=likelihoods, gpu_time=gpu_time, gpu_calls=gpu_calls)
