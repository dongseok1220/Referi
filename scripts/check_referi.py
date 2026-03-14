import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils import calculate_diffs


FALLBACK_TASKS = ["math500", "gpqa", "musr_efficiently"]
DEFAULT_MODELS = ["gpt-4o-mini"]
DEFAULT_SUBJECTS = [
    "business",
    "law",
    "psychology",
    "biology",
    "chemistry",
    "history",
    "other",
    "health",
    "economics",
    "math",
    "physics",
    "computer science",
    "philosophy",
    "engineering",
]

DROP_TASKS = {"drop", "hotpotqa"}
TRUNCATE_OUTPUT_LEN = 5000


@dataclass
class Paths:
    forward_dir: str = "forward"
    backward_dir: str = "backward"
    result_dir: str = "result"
    embedding_dir: str = "embedding_oneshot"


def discover_tasks(paths: Paths) -> List[str]:
    if not os.path.isdir(paths.forward_dir):
        return []
    tasks = []
    for name in os.listdir(paths.forward_dir):
        if name.startswith("."):
            continue
        full_path = os.path.join(paths.forward_dir, name)
        if os.path.isdir(full_path):
            tasks.append(name)
    return sorted(tasks)


def resolve_tasks(requested: Optional[List[str]], paths: Paths) -> List[str]:
    if requested:
        return list(requested)
    discovered = discover_tasks(paths)
    if discovered:
        return discovered
    print(f"[warning] no tasks found under {paths.forward_dir}; using fallback list")
    return FALLBACK_TASKS


def read_json(path: str, *, required: bool = True):
    if not os.path.exists(path):
        if required:
            print(f"[missing] {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: str, *, required: bool = True):
    if not os.path.exists(path):
        if required:
            print(f"[missing] {path}")
        return None
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[error] {path}:{line_no} {exc}")
                return None
    return records


def load_topk_indices(path: str, num_tests: Optional[int] = None):
    if not os.path.exists(path):
        print(f"[missing] {path}")
        return None

    raw = {}
    max_idx = -1
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[error] {path}:{line_no} {exc}")
                return None
            idx = rec.get("test_idx")
            if idx is None:
                continue
            raw[idx] = rec.get("fewshot_topk", [])
            max_idx = max(max_idx, idx)

    if num_tests is None:
        num_tests = max_idx + 1

    selected = [[] for _ in range(num_tests)]
    for idx, topk in raw.items():
        if 0 <= idx < num_tests:
            selected[idx] = topk
    return selected


def _normalize_result(entry: Optional[object]) -> Dict:
    if isinstance(entry, (list, tuple)):
        return entry[0] if entry else {}
    return entry or {}


def _mean_ce(entry: Dict) -> float:
    ce_losses = entry.get("ce_losses") or []
    return float(np.mean(ce_losses)) if ce_losses else float("nan")


def _outputs_compatible(left: str, right: str) -> bool:
    if not left or not right:
        return True
    if left == right:
        return True
    if left.startswith(right) or right.startswith(left):
        return True
    if TRUNCATE_OUTPUT_LEN:
        return left[:TRUNCATE_OUTPUT_LEN] == right[:TRUNCATE_OUTPUT_LEN]
    return False


def _outputs_match(few: Dict, problem_list: List[Dict]) -> bool:
    model_outputs = few.get("model_outputs") or []
    if not model_outputs or not problem_list:
        return True
    few_output = str(model_outputs[0])
    backward_output = str(problem_list[0].get("model_output") or "")
    return _outputs_compatible(few_output, backward_output)


def compute_referi_values(
    few_res: List[Dict],
    likelihoods: List[List[Dict]],
    scored_entries: List[Dict],
    selected_k: List[List[int]],
    strict_outputs: bool,
    context: str = "",
):
    referi_rows = []
    correct_rows = []

    problem_groups = zip(*likelihoods)
    for idx, (few, scored_entry, problem_list) in enumerate(
        zip(few_res, scored_entries, problem_groups)
    ):
        problem_list = list(problem_list)

        if strict_outputs and not _outputs_match(few, problem_list):
            suffix = f" ({context})" if context else ""
            print(f"[mismatch] output check failed at idx={idx}{suffix}")
            return None

        k_set = set(selected_k[idx]) if idx < len(selected_k) else None
        for candidate in problem_list:
            calculate_diffs(candidate, selected_indices=k_set)

        few_results = few.get("results") or []
        correct_list = scored_entry.get("is_correct") or []

        limit = min(len(few_results), len(problem_list), len(correct_list))
        if limit == 0:
            continue

        referi_row = []
        correct_row = []
        for cand_idx in range(limit):
            few_result = _normalize_result(few_results[cand_idx])
            forward = _mean_ce(few_result)

            backward_raw = problem_list[cand_idx].get("replace_ce_diff")
            if backward_raw is None:
                print(f"[warning] missing replace_ce_diff at idx={idx}, cand={cand_idx}")
                backward_raw = float("nan")

            referi = forward - float(backward_raw)

            referi_row.append(referi)
            correct_row.append(correct_list[cand_idx])

        referi_rows.append(referi_row)
        correct_rows.append(correct_row)

    if not referi_rows:
        return None

    return {
        "referi": np.array(referi_rows),
        "is_correct": np.array(correct_rows),
    }


def score_referi(result: Dict, is_drop: bool):
    if result is None:
        return None

    referi = result.get("referi")
    is_correct = result.get("is_correct")
    if referi is None or is_correct is None or referi.size == 0:
        return None

    chosen = referi.argmin(axis=-1)
    if is_drop and is_correct.ndim == 3:
        em = 100 * is_correct[np.arange(len(chosen)), chosen, 0].mean()
        f1 = 100 * is_correct[np.arange(len(chosen)), chosen, 1].mean()
        return em, f1

    if is_correct.ndim >= 2:
        acc = 100 * is_correct[np.arange(len(chosen)), chosen].mean()
        return acc

    return None


def format_score(score) -> object:
    if score is None:
        return np.nan
    if isinstance(score, tuple):
        return f"EM: {score[0]:.1f}, F1: {score[1]:.1f}"
    return round(float(score), 1)


def score_scalar(score) -> Optional[float]:
    if score is None:
        return None
    if isinstance(score, tuple):
        return float(score[0])
    try:
        return float(score)
    except (TypeError, ValueError):
        return None


def resolve_task_topk_path(task: str, model: str, paths: Paths) -> str:
    if task == "math500" and model == "llama":
        filename = f"{task}_llama_k.jsonl"
    else:
        filename = f"{task}_k.jsonl"
    return os.path.join(paths.embedding_dir, "sim", task, filename)


def task_paths(task: str, model: str, paths: Paths) -> Dict[str, str]:
    return {
        "forward_few": os.path.join(
            paths.forward_dir, task, model, f"{task}_few_few.jsonl"
        ),
        "backward": os.path.join(
            paths.backward_dir, task, model, "few", "all_likelihoods.json"
        ),
        "scored": os.path.join(
            paths.result_dir, task, model, f"{task}_few_scored.jsonl"
        ),
    }


def mmlu_paths(model: str, subject: str, paths: Paths) -> Dict[str, str]:
    return {
        "forward_few": os.path.join(
            paths.forward_dir,
            "mmlu_pro",
            model,
            subject,
            "mmlu_pro_few_few.jsonl",
        ),
        "backward": os.path.join(
            paths.backward_dir,
            "mmlu_pro",
            model,
            subject,
            "few",
            "all_likelihoods.json",
        ),
        "scored": os.path.join(
            paths.result_dir,
            "mmlu_pro",
            model,
            f"{subject}_result_scored.jsonl",
        ),
        "topk": os.path.join(
            paths.embedding_dir, "sim", "mmlu_pro", f"{subject}_k.jsonl"
        ),
    }


def compute_task_referi(
    task: str,
    model: str,
    paths: Paths,
    strict_outputs: bool,
):
    files = task_paths(task, model, paths)

    few_res = read_jsonl(files["forward_few"])
    scored_entries = read_jsonl(files["scored"])
    likelihoods = read_json(files["backward"])

    if not all([few_res, scored_entries, likelihoods]):
        return None

    topk_path = resolve_task_topk_path(task, model, paths)
    selected_k = load_topk_indices(topk_path, num_tests=len(scored_entries))
    if selected_k is None:
        return None

    return compute_referi_values(
        few_res,
        likelihoods,
        scored_entries,
        selected_k,
        strict_outputs,
        context=(
            f"task={task} model={model} "
            f"forward={files['forward_few']} "
            f"backward={files['backward']} "
            f"scored={files['scored']}"
        )
    )


def compute_mmlu_referi(
    model: str,
    subjects: List[str],
    paths: Paths,
    strict_outputs: bool,
) -> Optional[float]:
    subject_scores = []

    for subject in subjects:
        files = mmlu_paths(model, subject, paths)

        few_res = read_jsonl(files["forward_few"])
        scored_entries = read_jsonl(files["scored"])
        likelihoods = read_json(files["backward"])
        selected_k = load_topk_indices(files["topk"], num_tests=len(scored_entries or []))

        if not all([few_res, scored_entries, likelihoods, selected_k]):
            continue

        result = compute_referi_values(
            few_res,
            likelihoods,
            scored_entries,
            selected_k,
            strict_outputs,
            context=(
                f"task=mmlu_pro subject={subject} model={model} "
                f"forward={files['forward_few']} "
                f"backward={files['backward']} "
                f"scored={files['scored']}"
            ),
        )

        score = score_referi(result, is_drop=False)
        if score is not None:
            subject_scores.append(float(score))

    if not subject_scores:
        return None

    return float(np.mean(subject_scores))


def build_table(
    task_scores: Dict[str, object],
    task_order: List[str],
) -> pd.DataFrame:
    row = {}
    for task in task_order:
        row[task] = format_score(task_scores.get(task))

    scalars = [score_scalar(task_scores.get(task)) for task in task_order]
    scalars = [val for val in scalars if val is not None]
    row["Avg"] = round(float(np.mean(scalars)), 1) if scalars else np.nan

    return pd.DataFrame([row], index=["Referi"])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS)
    parser.add_argument("--forward_dir", "--baseline_dir", dest="forward_dir", type=str, default="forward")
    parser.add_argument("--backward_dir", "--likelihood_dir", dest="backward_dir", type=str, default="backward")
    parser.add_argument("--result_dir", type=str, default="result")
    parser.add_argument("--embedding_dir", type=str, default="embedding_oneshot")
    parser.add_argument("--no_strict_outputs", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    paths = Paths(
        forward_dir=args.forward_dir,
        backward_dir=args.backward_dir,
        result_dir=args.result_dir,
        embedding_dir=args.embedding_dir,
    )

    tasks = resolve_tasks(args.tasks, paths)
    include_mmlu = "mmlu_pro" in tasks
    task_list = [task for task in tasks if task != "mmlu_pro"]

    for model in args.models:
        task_scores: Dict[str, object] = {}

        for task in task_list:
            result = compute_task_referi(
                task,
                model,
                paths,
                strict_outputs=not args.no_strict_outputs,
            )
            task_scores[task] = score_referi(result, is_drop=task in DROP_TASKS)

        if include_mmlu:
            task_scores["mmlu_pro"] = compute_mmlu_referi(
                model,
                args.subjects,
                paths,
                strict_outputs=not args.no_strict_outputs,
            )

        ordered = task_list + (["mmlu_pro"] if include_mmlu else [])
        df = build_table(task_scores, ordered)

        print(f"===== MODEL: {model} =====")
        print(df.to_string())
        print()


if __name__ == "__main__":
    main()
