import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from tasks.drop.drop_utils import convert_ndarray_to_list
from tasks.gpqa.gpqa_utils import load_examples
from tasks.hotpotqa.hotpotqa_utils import *
from tasks.math500.grader import *
from tasks.math500.math_utils import *
from tasks.math500.parser import *
from tasks.mmlu_pro.mmlu_utils import load_mmlu_pro
from tasks.musr.musr import MuSRDataset

MODEL_ID = "sentence-transformers/all-mpnet-base-v2"
BATCH_SIZE = 1
TOP_K = 1
DEFAULT_OUTPUT_DIR = Path("embedding_oneshot") / "sim"
DEVICE = "cpu"


def mean_pooling(model_output, attention_mask):
    token_emb = model_output[0]
    mask = attention_mask.unsqueeze(-1).expand(token_emb.shape).float()
    return (token_emb * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)


@torch.inference_mode()
def encode_texts(texts, tokenizer, model):
    all_embs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        outputs = model(**enc)
        emb = mean_pooling(outputs, enc["attention_mask"])
        emb = F.normalize(emb, p=2, dim=1)
        all_embs.append(emb.cpu())
    return torch.cat(all_embs, dim=0)


def write_topk(path, similarity):
    path = str(path)
    topk_idxs = torch.topk(similarity, TOP_K, dim=1).indices
    os.makedirs(os.path.dirname(path), exist_ok=True)
    records = [
        {"test_idx": int(i), "fewshot_topk": topk_idxs[i].tolist()}
        for i in range(topk_idxs.size(0))
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)


def compute_and_write(few_questions, test_questions, output_path, tokenizer, model):
    if not few_questions or not test_questions:
        print(f"[warning] empty inputs for {output_path}")
        return
    few_embs = encode_texts(few_questions, tokenizer, model)
    test_embs = encode_texts(test_questions, tokenizer, model)
    similarity = torch.matmul(test_embs, few_embs.T)
    write_topk(output_path, similarity)
    print(f"[saved] {output_path}")


def run_math500(tokenizer, model, output_root: Path):
    with open(ROOT / "data" / "math500" / "test.jsonl", "r", encoding="utf-8") as f:
        test_data = [json.loads(ln) for ln in f]
    with open(
        ROOT / "data" / "fewshot" / "math500" / "few_shot_seed_2025.json",
        "r",
        encoding="utf-8",
    ) as f:
        few_shot = json.load(f)

    few_questions = [ex[0] for ex in few_shot]
    test_questions = [ex["problem"] for ex in test_data]
    output_path = output_root / "math500" / "math500_k.jsonl"
    compute_and_write(few_questions, test_questions, output_path, tokenizer, model)


def run_mmlu_pro(tokenizer, model, output_root: Path):
    test_df, dev_df = load_mmlu_pro()
    subjects = list(test_df.keys())
    print("assigned subjects", subjects)
    for subject in subjects:
        few_shot = dev_df[subject]
        dataset = test_df[subject]
        random.seed(42)
        test_data = random.sample(dataset, min(300, len(dataset)))
        few_questions = [ex["question"] for ex in few_shot]
        test_questions = [ex["question"] for ex in test_data]
        output_path = output_root / "mmlu_pro" / f"{subject}_k.jsonl"
        compute_and_write(few_questions, test_questions, output_path, tokenizer, model)


def run_gpqa(tokenizer, model, output_root: Path):
    prompt_path = ROOT / "data" / "fewshot" / "gpqa" / "few_shot_seed_2025.json"
    with open(prompt_path, "r", encoding="utf-8") as f:
        few_shot = json.load(f)["questions"]
    test_data = load_examples(str(ROOT / "data" / "gpqa" / "gpqa_diamond.csv"), seed=0)

    few_questions = [ex["question"] for ex in few_shot]
    test_questions = [ex.question for ex in test_data]
    output_path = output_root / "gpqa" / "gpqa_k.jsonl"
    compute_and_write(few_questions, test_questions, output_path, tokenizer, model)


def run_drop(tokenizer, model, output_root: Path):
    test_data = pd.read_parquet(ROOT / "data" / "drop" / "drop_sub.parquet", engine="pyarrow")
    test_data = test_data.to_dict(orient="records")
    test_data = convert_ndarray_to_list(test_data)
    test_data = convert_ndarray_to_list(test_data)
    with open(ROOT / "tasks" / "drop" / "prompt.json", "r", encoding="utf-8") as f:
        few_shot = json.load(f)

    few_questions = [ex["Q"] for ex in few_shot]
    test_questions = [f"{ex['passage']} {ex['question']}" for ex in test_data]
    output_path = output_root / "drop" / "drop_k.jsonl"
    compute_and_write(few_questions, test_questions, output_path, tokenizer, model)


def run_hotpotqa(tokenizer, model, output_root: Path):
    with open(ROOT / "data" / "hotpotqa" / "hotpotqa.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)
    with open(ROOT / "tasks" / "hotpotqa" / "react_prompt.json", "r", encoding="utf-8") as f:
        few_shot = json.load(f)

    few_questions = [ex["Q"] for ex in few_shot]
    test_questions = [ex["question"] for ex in test_data]
    output_path = output_root / "hotpotqa" / "hotpotqa_k.jsonl"
    compute_and_write(few_questions, test_questions, output_path, tokenizer, model)


def run_musr_location(tokenizer, model, output_root: Path):
    from tasks.musr.icl import (
        OBJECT_PLACEMENTS_FEWSHOT,
        OBJECT_PLACEMENTS_TEST_INSTRUCTION,
    )

    op_path = str(ROOT / "data" / "musr" / "object_placements.json")
    test_data = MuSRDataset(op_path)

    few_questions = [ex[0] for ex in OBJECT_PLACEMENTS_FEWSHOT]
    test_questions = []

    for entry in test_data:
        question = entry["question"].strip()
        context = entry["context"].strip()
        choices = entry["choices"]["text"]
        labels = ["A", "B", "C", "D", "E", "F"][0 : len(choices)]

        choice_str = "\n".join(
            [f"{labels[idx]}: {choices[idx]}" for idx in range(len(choices))]
        )
        question_block = f"{context}\n\n{question}\n\n{choice_str}"
        question_block = f"{question_block}\n\n{OBJECT_PLACEMENTS_TEST_INSTRUCTION}".strip()
        test_questions.append(question_block)

    output_path = output_root / "musr_location" / "musr_location_k.jsonl"
    compute_and_write(few_questions, test_questions, output_path, tokenizer, model)


def run_musr_efficiently(tokenizer, model, output_root: Path):
    from tasks.musr.icl import TEAM_ALLOCATION_FEWSHOT

    ta_path = str(ROOT / "data" / "musr" / "team_allocation.json")
    test_data = MuSRDataset(ta_path)

    few_questions = [ex[0] for ex in TEAM_ALLOCATION_FEWSHOT]
    test_questions = []

    for entry in test_data:
        question = entry["question"].strip()
        context = entry["context"].strip()
        choices = entry["choices"]["text"]
        labels = ["A", "B", "C", "D", "E", "F"][0 : len(choices)]

        choice_str = "\n".join(
            [f"{labels[idx]}: {choices[idx]}" for idx in range(len(choices))]
        )
        question_block = f"{context}\n\n{question}\n\n{choice_str}".strip()
        test_questions.append(question_block)

    output_path = output_root / "musr_efficiently" / "musr_efficiently_k.jsonl"
    compute_and_write(few_questions, test_questions, output_path, tokenizer, model)


TASK_RUNNERS = {
    "math500": run_math500,
    "mmlu_pro": run_mmlu_pro,
    "gpqa": run_gpqa,
    "drop": run_drop,
    "hotpotqa": run_hotpotqa,
    "musr_location": run_musr_location,
    "musr_efficiently": run_musr_efficiently,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = ROOT / output_root

    tasks = args.tasks or list(TASK_RUNNERS.keys())
    print(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE).eval()

    for task in tasks:
        runner = TASK_RUNNERS.get(task)
        if runner is None:
            print(f"[warning] unknown task: {task}")
            continue
        print(f"[task] {task}")
        runner(tokenizer, model, output_root)


if __name__ == "__main__":
    main()
