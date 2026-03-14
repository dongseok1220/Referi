import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.io import read_jsonl
from core.modeling import load_llm, calculate_cross_entropy_loss_with_topk
from core.records import normalize_record


load_dotenv()


MMLU_PRO_SUBJECTS = [
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
SHARED_ZERO_PROMPT_ROOT = "result"


def _prefer_shared_zero_prompt(shot_type: str, shared_path: str, default_path: str) -> str:
    if shot_type != "zero":
        return default_path
    if os.path.exists(shared_path):
        return shared_path
    return default_path


def _resolve_mmlu_path(input_dir: str, model: str, subject: str, shot_type: str | None = None) -> str:
    suffix = f"{subject}_zero_result.jsonl" if shot_type == "zero" else f"{subject}_result.jsonl"
    direct = os.path.join(input_dir, "mmlu_pro", model, suffix)
    nested = os.path.join(input_dir, "mmlu_pro", model, subject, suffix)
    if os.path.exists(direct):
        return direct
    return nested


def _resolve_input_paths(args, subject: str | None) -> tuple[str, str]:
    if args.input_file:
        return args.input_file, args.input_file

    prompt_shot = args.shot_type
    output_shot = "few" if args.shot_type == "zero" else args.shot_type

    if args.task == "mmlu_pro":
        if not subject:
            raise ValueError("mmlu_pro requires --subject or --all_subjects")
        default_prompt_path = _resolve_mmlu_path(args.input_dir, args.model, subject, prompt_shot)
        shared_prompt_path = _resolve_mmlu_path(SHARED_ZERO_PROMPT_ROOT, args.model, subject, prompt_shot)
        prompt_path = _prefer_shared_zero_prompt(prompt_shot, shared_prompt_path, default_prompt_path)
        output_path = _resolve_mmlu_path(args.input_dir, args.model, subject, output_shot)
    else:
        default_prompt_path = os.path.join(
            args.input_dir, args.task, args.model, f"{args.task}_{prompt_shot}.jsonl"
        )
        shared_prompt_path = os.path.join(
            SHARED_ZERO_PROMPT_ROOT, args.task, args.model, f"{args.task}_{prompt_shot}.jsonl"
        )
        prompt_path = _prefer_shared_zero_prompt(prompt_shot, shared_prompt_path, default_prompt_path)
        output_path = os.path.join(
            args.input_dir, args.task, args.model, f"{args.task}_{output_shot}.jsonl"
        )

    if prompt_path != output_path:
        if not os.path.exists(prompt_path) and os.path.exists(output_path):
            prompt_path = output_path
        elif not os.path.exists(output_path) and os.path.exists(prompt_path):
            output_path = prompt_path

    return prompt_path, output_path


def _build_samples(prompt_path: str, output_path: str):
    records = read_jsonl(prompt_path, allow_errors=True)
    if prompt_path == output_path:
        samples = []
        for idx, record in enumerate(records):
            norm = normalize_record(record, idx)
            if not norm.model_outputs:
                continue
            samples.append(
                {
                    "idx": norm.idx,
                    "prompt": norm.prompt,
                    "model_outputs": norm.model_outputs,
                }
            )
        return samples

    output_records = read_jsonl(output_path, allow_errors=True)
    output_map = {}
    for idx, record in enumerate(output_records):
        norm = normalize_record(record, idx)
        if not norm.model_outputs:
            continue
        output_map[norm.idx] = norm.model_outputs

    samples = []
    missing_outputs = 0
    for idx, record in enumerate(records):
        norm = normalize_record(record, idx)
        outputs = output_map.get(norm.idx)
        if not outputs:
            missing_outputs += 1
            continue
        samples.append(
            {
                "idx": norm.idx,
                "prompt": norm.prompt,
                "model_outputs": outputs,
            }
        )
    if missing_outputs:
        print(f"[warn] missing outputs for {missing_outputs} records in {output_path}")
    return samples


def _resolve_paths(args, subject: str | None):
    if args.input_file:
        input_path = args.input_file
    elif args.task == "mmlu_pro":
        if not subject:
            raise ValueError("mmlu_pro requires --subject or --all_subjects")
        input_path = _resolve_mmlu_path(args.input_dir, args.model, subject)
    else:
        input_path = os.path.join(
            args.input_dir, args.task, args.model, f"{args.task}_{args.shot_type}.jsonl"
        )

    if args.task == "mmlu_pro":
        save_path = os.path.join(
            args.output_dir,
            args.task,
            args.model,
            subject,
            f"{args.task}_few_{args.shot_type}.jsonl",
        )
        time_txt = os.path.join(
            args.output_dir,
            args.task,
            args.model,
            subject,
            f"gpu_times_{args.shot_type}.txt",
        )
    else:
        save_path = os.path.join(
            args.output_dir,
            args.task,
            args.model,
            f"{args.task}_few_{args.shot_type}.jsonl",
        )
        time_txt = os.path.join(
            args.output_dir,
            args.task,
            args.model,
            f"gpu_times_{args.shot_type}.txt",
        )

    return input_path, save_path, time_txt


def _log_first_sample(samples, args, subject: str | None) -> None:
    if not samples:
        print("No samples available for debug logging.")
        return
    sample = samples[0]
    subject_part = f" subject={subject}" if subject else ""
    print("=== [First Sample] ===")
    print(f"task={args.task} model={args.model}{subject_part} id={sample.get('idx')}")
    print("--- prompt ---")
    print(sample.get("prompt", ""))
    print("--- model_outputs[0] ---")
    outputs = sample.get("model_outputs") or []
    print(outputs[0] if outputs else "")
    print("=== [End First Sample] ===")


def main(args):
    tokenizer, model, device = load_llm(args.model_name, args.hf_token)

    if args.task == "mmlu_pro":
        subjects = [args.subject] if args.subject else MMLU_PRO_SUBJECTS
    else:
        subjects = [None]

    for subject in subjects:
        prompt_path, output_path = _resolve_input_paths(args, subject)
        _, save_path, time_txt = _resolve_paths(args, subject)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        samples = _build_samples(prompt_path, output_path)
        if args.num_examples != -1:
            samples = samples[: args.num_examples]

        if not samples:
            if prompt_path == output_path:
                print(f"No samples found at {prompt_path}")
            else:
                print(f"No samples found for prompts={prompt_path} outputs={output_path}")
            continue
        if args.log_first:
            _log_first_sample(samples, args, subject)

        existing_ids = set()
        if os.path.exists(save_path):
            for record in read_jsonl(save_path, allow_errors=True):
                if "idx" in record:
                    existing_ids.add(record["idx"])

        gpu_time = {"response": 0.0}
        gpu_calls = 0

        with open(save_path, "a", encoding="utf-8") as f:
            for sample in samples:
                if sample["idx"] in existing_ids:
                    continue

                results = []
                for output in sample["model_outputs"]:
                    if len(output) > args.max_output_len:
                        output = output[: args.max_output_len]
                    result, dt_ms = calculate_cross_entropy_loss_with_topk(
                        sample["prompt"], output, model, tokenizer, device
                    )
                    results.append(result)
                    gpu_time["response"] += dt_ms
                    gpu_calls += 1

                sample["results"] = results
                json.dump(sample, f, ensure_ascii=False)
                f.write("\n")

        avg = (gpu_time["response"] / gpu_calls) if gpu_calls > 0 else 0.0
        with open(time_txt, "w", encoding="utf-8") as f:
            f.write("=== GPU-only elapsed time (ms) ===\n")
            f.write(f"response : {gpu_time['response']:.2f}\n")
            f.write("-------------------------------\n")
            f.write(f"total calls: {gpu_calls}\n")
            f.write(f"avg / call : {avg:.2f}\n")
        print(f"Results saved to {save_path}")
        print(f"GPU timing saved to {time_txt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o", help="output model name")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--task", type=str, default="math500")
    parser.add_argument("--shot_type", type=str, default="few")
    parser.add_argument("--output_dir", type=str, default="forward")
    parser.add_argument("--input_dir", type=str, default="result")
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--num_examples", type=int, default=-1)
    parser.add_argument("--subject", type=str, default=None)
    parser.add_argument("--max_output_len", type=int, default=5000)
    parser.add_argument("--hf_token", type=str, default=os.getenv("HUGGINGFACE_TOKEN"))
    parser.add_argument("--log_first", action="store_true", help="print the first sample for debugging")

    args = parser.parse_args()
    main(args)
