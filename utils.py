import os
import json

from core.io import read_json, read_jsonl
from core.modeling import (
    load_llm as _load_llm,
    calculate_cross_entropy_loss_with_topk as _calculate_cross_entropy_loss_with_topk,
    timed_call as _timed_call,
)


def load_llm(model_name: str, hf_token: str | None = None):
    return _load_llm(model_name, hf_token)

def load_existing_likelihoods(file_path):
    data = read_json(file_path, default=[])
    return data if isinstance(data, list) else []

def load_model_outputs(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return None
    return read_jsonl(file_path, allow_errors=True)
    
def calculate_diffs(current_likelihoods, selected_indices=None):
    selected = set(selected_indices) if selected_indices is not None else None

    def mean_list(values):
        return (sum(values) / len(values)) if values else 0.0

    def compute_sums(entry_list):
        ce_sums = []
        for idx, entry in enumerate(entry_list):
            if selected is not None and idx not in selected:
                continue
            ce_sums.append(mean_list(entry.get("ce_losses", [])))
        return sum(ce_sums) / len(ce_sums) if ce_sums else 0.0

    no_replace_ce = compute_sums(current_likelihoods.get("no_replace_likelihoods", []))
    replace_ce = compute_sums(current_likelihoods.get("replace_likelihoods", []))
    zero_ce = compute_sums(current_likelihoods.get("zero_likelihoods", []))

    current_likelihoods["no_replace_ce_diff"] = no_replace_ce - zero_ce
    current_likelihoods["replace_ce_diff"] = replace_ce - zero_ce

    selected = None

    no_replace_ce_full  = compute_sums(current_likelihoods.get("no_replace_likelihoods", []))
    replace_ce_full = compute_sums(current_likelihoods.get("replace_likelihoods", []))
    zero_ce_full = compute_sums(current_likelihoods.get("zero_likelihoods", []))

    current_likelihoods["full_no_replace_ce_diff"] = no_replace_ce_full  - zero_ce_full
    current_likelihoods["full_replace_ce_diff"] = replace_ce_full - zero_ce_full   


def calculate_cross_entropy_loss_with_topk(prompt, answer, model, tokenizer, device, top_k=2):
    return _calculate_cross_entropy_loss_with_topk(prompt, answer, model, tokenizer, device, top_k=top_k)

def timed_gpu(fn, *args, **kwargs):
    return _timed_call(fn, *args, **kwargs)


def process_scored_file(scored_path: str, _type: str):
    with open(scored_path, 'r', encoding='utf-8') as f:
        scored_entries = [json.loads(line) for line in f]

    if not scored_entries:
        print(f"Scored file {scored_path} is empty.\n")
        return

    first = scored_entries[0]["is_correct"]

    if isinstance(first, list) and first and isinstance(first[0], bool):
        num_questions = len(scored_entries)
        n_repeats = len(first)

        scores = [[] for _ in range(n_repeats)]
        any_correct = []

        for entry in scored_entries:
            results = entry["is_correct"]
            for i, r in enumerate(results):
                scores[i].append(r)
            any_correct.append(any(results))

        rep_accuracies = [
            sum(scores[i]) / num_questions for i in range(n_repeats)
        ]
        avg_rep_acc = sum(rep_accuracies) / n_repeats
        any_correct_acc = sum(any_correct) / num_questions

        for i, acc_val in enumerate(rep_accuracies):
            print(f"{i}:  {acc_val:.3f}")
        print(f"{_type} Avg: {avg_rep_acc:.3f}")
        print(f"Any_correct: {any_correct_acc:.3f}\n")

    elif isinstance(first, list) and first and isinstance(first[0], (list, tuple)) and len(first[0]) == 2:
        n_repeats = len(first)
        num_questions = len(scored_entries)

        em_scores = [[] for _ in range(n_repeats)]
        f1_scores = [[] for _ in range(n_repeats)]

        for entry in scored_entries:
            results = entry["is_correct"]
            for i, (em, f1) in enumerate(results):
                em_scores[i].append(em)
                f1_scores[i].append(f1)

        rep_em = [sum(em_scores[i]) / num_questions for i in range(n_repeats)]
        rep_f1 = [sum(f1_scores[i]) / num_questions for i in range(n_repeats)]

        avg_em = sum(rep_em) / n_repeats
        avg_f1 = sum(rep_f1) / n_repeats

        for i in range(n_repeats):
            print(f"{i}: EM={rep_em[i]:.3f}, F1={rep_f1[i]:.3f}")
        print(f"{_type} Avg EM: {avg_em:.3f}, Avg F1: {avg_f1:.3f}\n")

    else:
        print(f"Unrecognized format for is_correct in {scored_path}")

def load_gpu_time_state(time_txt: str):
    gpu_time = {"no": 0.0, "rep": 0.0, "zero": 0.0}
    gpu_calls = 0

    if not os.path.exists(time_txt):
        return gpu_time, gpu_calls

    try:
        with open(time_txt, "r") as f:
            for raw in f:
                line = raw.strip().lower()
                if line.startswith("no_replace"):
                    # "no_replace : 123.45"
                    gpu_time["no"] = float(line.split(":", 1)[1].strip())
                elif line.startswith("replace"):
                    gpu_time["rep"] = float(line.split(":", 1)[1].strip())
                elif line.startswith("zero"):
                    gpu_time["zero"] = float(line.split(":", 1)[1].strip())
                elif line.startswith("total calls"):
                    gpu_calls = int(line.split(":", 1)[1].strip())
        print(f"[ℹ] Loaded previous GPU times from {time_txt}")
    except Exception as e:
        print(f"[!] Failed to parse existing gpu_times.txt ({e}). Starting from zeros.")
    return gpu_time, gpu_calls


def save_gpu_time_state(time_txt: str, gpu_time: dict, gpu_calls: int):
    os.makedirs(os.path.dirname(time_txt), exist_ok=True)
    with open(time_txt, "w") as f:
        f.write("=== GPU-only elapsed time (ms) ===\n")
        f.write(f"no_replace : {gpu_time['no']:.2f}\n")
        f.write(f"replace    : {gpu_time['rep']:.2f}\n")
        f.write(f"zero       : {gpu_time['zero']:.2f}\n")
        f.write(f"-------------------------------\n")
        f.write(f"total calls: {gpu_calls}\n")
    print(f"[✓] GPU timing saved to {time_txt}")
