import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd

from core.evaluation import (
    DropEvaluator,
    GpqaEvaluator,
    HotpotqaEvaluator,
    MathEvaluator,
    MmluProEvaluator,
    MusrEvaluator,
)
from tasks.gpqa.gpqa_utils import load_examples
from tasks.mmlu_pro.mmlu_utils import extract_answer, load_mmlu_pro
from tasks.musr.musr_ import MuSRDataset
from utils import process_scored_file


def _parse_csv(value: str | None) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _resolve_models(args) -> List[str]:
    models = _parse_csv(args.models)
    if not models and args.model:
        models = [args.model]
    if not models:
        raise ValueError("Provide --models or --model.")
    return models


def _resolve_shot_types(args) -> List[str]:
    shot_types = _parse_csv(args.shot_types)
    if not shot_types:
        shot_types = ["zero", "few"]
    return shot_types


def _math_dataset_path(task: str) -> Path:
    return ROOT / "data" / task / "test.jsonl"


def _is_math_task(task: str) -> bool:
    return _math_dataset_path(task).exists()


def _evaluate_file(evaluator, file_path: str, scored_path: str, shot_type: str) -> None:
    evaluator.evaluate_file(file_path, scored_path)
    process_scored_file(scored_path, shot_type)
    print(f"Saved scored file: {scored_path}")


def _evaluate_gpqa(args) -> None:
    dataset = load_examples("data/gpqa/gpqa_diamond.csv", seed=args.seed)
    evaluator = GpqaEvaluator(dataset)
    models = _resolve_models(args)
    shot_types = _resolve_shot_types(args)

    for model in models:
        print(model)
        for shot_type in shot_types:
            file_path = os.path.join(args.base_dir, "gpqa", model, f"gpqa_{shot_type}.jsonl")
            scored_path = os.path.join(args.base_dir, "gpqa", model, f"gpqa_{shot_type}_scored.jsonl")
            if os.path.exists(scored_path) and args.skip_existing:
                process_scored_file(scored_path, shot_type)
                continue
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue
            _evaluate_file(evaluator, file_path, scored_path, shot_type)


def _evaluate_drop(args) -> None:
    dataset = pd.read_parquet("data/drop/drop_sub.parquet", engine="pyarrow")
    dataset = dataset.to_dict(orient="records")
    evaluator = DropEvaluator(dataset)
    models = _resolve_models(args)
    shot_types = _resolve_shot_types(args)

    for model in models:
        print(model)
        for shot_type in shot_types:
            file_path = os.path.join(args.base_dir, "drop", model, f"drop_{shot_type}.jsonl")
            scored_path = os.path.join(args.base_dir, "drop", model, f"drop_{shot_type}_scored.jsonl")
            if os.path.exists(scored_path) and args.skip_existing:
                process_scored_file(scored_path, shot_type)
                continue
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue
            _evaluate_file(evaluator, file_path, scored_path, shot_type)


def _evaluate_hotpotqa(args) -> None:
    with open("data/hotpotqa/hotpotqa.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
    evaluator = HotpotqaEvaluator(dataset)
    models = _resolve_models(args)
    shot_types = _resolve_shot_types(args)

    for model in models:
        print(model)
        for shot_type in shot_types:
            file_path = os.path.join(args.base_dir, "hotpotqa", model, f"hotpotqa_{shot_type}.jsonl")
            scored_path = os.path.join(args.base_dir, "hotpotqa", model, f"hotpotqa_{shot_type}_scored.jsonl")
            if os.path.exists(scored_path) and args.skip_existing:
                process_scored_file(scored_path, shot_type)
                continue
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue
            _evaluate_file(evaluator, file_path, scored_path, shot_type)


def _evaluate_math(args) -> None:
    dataset_path = _math_dataset_path(args.task)
    with dataset_path.open("r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f]

    evaluator = MathEvaluator(dataset, args.task)
    models = _resolve_models(args)
    shot_types = _resolve_shot_types(args)

    for shot_type in shot_types:
        for model in models:
            file_path = os.path.join(args.base_dir, args.task, model, f"{args.task}_{shot_type}.jsonl")
            scored_path = os.path.join(args.base_dir, args.task, model, f"{args.task}_{shot_type}_scored.jsonl")
            if os.path.exists(scored_path) and args.skip_existing:
                process_scored_file(scored_path, shot_type)
                continue
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue
            _evaluate_file(evaluator, file_path, scored_path, shot_type)


def _evaluate_mmlu_pro(args) -> None:
    test_df, _ = load_mmlu_pro()
    if args.all_subjects:
        subjects = list(test_df.keys())
    else:
        subjects = _parse_csv(args.subjects)
        if not subjects:
            subjects = list(test_df.keys())
    models = _resolve_models(args)
    types = _parse_csv(args.mmlu_types) or ["_result", "_zero_result"]

    evaluator = MmluProEvaluator(extract_answer)

    base_dir = args.base_dir
    if Path(base_dir).name != "mmlu_pro":
        base_dir = os.path.join(base_dir, "mmlu_pro")
    for model in models:
        for t in types:
            subject_rep_accs = []

            for subject in subjects:
                raw_path = os.path.join(base_dir, model, f"{subject}{t}.jsonl")
                scored_path = os.path.join(base_dir, model, f"{subject}{t}_scored.jsonl")
                if not os.path.exists(raw_path):
                    continue
                rep_acc = evaluator.evaluate_subject(raw_path, scored_path, skip_if_exists=args.skip_existing)
                if rep_acc:
                    subject_rep_accs.append(rep_acc)

            if not subject_rep_accs:
                print(f"\n=== Results (model={model}, type={t}): no data ===")
                continue

            common_reps = min(len(a) for a in subject_rep_accs)
            rep_avgs = []
            for i in range(common_reps):
                rep_avgs.append(sum(acc[i] for acc in subject_rep_accs) / len(subject_rep_accs))
            overall_avg = sum(rep_avgs) / len(rep_avgs) if rep_avgs else 0.0

            print(f"\n=== Results (model={model}, type={t}) ===")
            for i, a in enumerate(rep_avgs):
                print(f"Idx{i}: {a:.4f}")
            print(f"Avg : {overall_avg:.4f}")


def _evaluate_musr(args) -> None:
    models = _resolve_models(args)
    shot_types = _resolve_shot_types(args)

    if args.task == "musr_location":
        dataset = MuSRDataset("data/musr/object_placements.json")
    else:
        dataset = MuSRDataset("data/musr/team_allocation.json")
    evaluator = MusrEvaluator(dataset)

    for model in models:
        print(model)
        for shot_type in shot_types:
            file_path = os.path.join(args.base_dir, args.task, model, f"{args.task}_{shot_type}.jsonl")
            scored_path = os.path.join(args.base_dir, args.task, model, f"{args.task}_{shot_type}_scored.jsonl")
            if os.path.exists(scored_path) and args.skip_existing:
                process_scored_file(scored_path, shot_type)
                continue
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue
            _evaluate_file(evaluator, file_path, scored_path, shot_type)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, help="Task name (gpqa, drop, hotpotqa, mmlu_pro, musr_*, or math dataset name)")
    parser.add_argument("--models", default=None, help="Comma-separated model list")
    parser.add_argument("--model", default=None, help="Single model name")
    parser.add_argument("--shot_types", default="zero,few", help="Comma-separated shot types")
    parser.add_argument("--base_dir", default="result", help="Base directory for outputs")
    parser.add_argument("--seed", type=int, default=0, help="Seed for GPQA sampling")
    parser.add_argument("--skip_existing", action="store_true", help="Skip scoring when *_scored.jsonl exists")
    parser.add_argument("--subjects", default=None, help="Comma-separated MMLU-Pro subjects")
    parser.add_argument("--all_subjects", action="store_true", help="Use all MMLU-Pro subjects")
    parser.add_argument("--mmlu_types", default="_result,_zero_result", help="Comma-separated MMLU-Pro result suffixes")

    args = parser.parse_args()

    if args.task == "gpqa":
        _evaluate_gpqa(args)
    elif args.task == "drop":
        _evaluate_drop(args)
    elif args.task == "hotpotqa":
        _evaluate_hotpotqa(args)
    elif args.task == "mmlu_pro":
        _evaluate_mmlu_pro(args)
    elif args.task.startswith("musr_"):
        _evaluate_musr(args)
    elif _is_math_task(args.task):
        _evaluate_math(args)
    else:
        raise ValueError(f"Unsupported task: {args.task}")


if __name__ == "__main__":
    main()
