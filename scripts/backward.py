import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from dotenv import load_dotenv

from core.io import read_jsonl
from core.likelihood import PromptSet, load_existing_likelihoods, run_likelihood
from core.llama_template import LLAMA_ASSISTANT, LLAMA_EOT, LLAMA_SYSTEM, LLAMA_USER
from core.modeling import load_llm
from core.records import extract_model_outputs, get_record_id
from tasks.gpqa.gpqa_utils import load_examples
from tasks.math500.math_utils import load_prompt, list_fewshot_samples
from tasks.math500.parser import parse_ground_truth
from tasks.mmlu_pro.mmlu_utils import load_mmlu_pro
from tasks.musr.musr_ import MuSRDataset
from tasks.musr.icl import OBJECT_PLACEMENTS_FEWSHOT, OBJECT_PLACEMENTS_FEWSHOT_INSTRUCTION, OBJECT_PLACEMENTS_TEST_INSTRUCTION
from tasks.musr.icl import TEAM_ALLOCATION_FEWSHOT, TEAM_ALLOCATION_FEWSHOT_INSTRUCTION, TEAM_ALLOCATION_TEST_INSTRUCTION


load_dotenv()


def _resolve_mmlu_path(input_dir: str, model: str, subject: str) -> str:
    direct = os.path.join(input_dir, "mmlu_pro", model, f"{subject}_result.jsonl")
    nested = os.path.join(input_dir, "mmlu_pro", model, subject, f"{subject}_result.jsonl")
    if os.path.exists(direct):
        return direct
    return nested


def _get_input_path(args):
    if args.task == "mmlu_pro":
        return _resolve_mmlu_path(args.input_dir, args.model, args.subject)
    return os.path.join(args.input_dir, args.task, args.model, f"{args.task}_few.jsonl")


def _get_output_dir(args):
    if args.task == "mmlu_pro":
        return os.path.join(args.output_dir, args.task, args.model, args.subject, "few")
    return os.path.join(args.output_dir, args.task, args.model, "few")


def _write_gpu_times(path: str, gpu_time: Dict[str, float], gpu_calls: int) -> None:
    avg = (sum(gpu_time.values()) / gpu_calls) if gpu_calls > 0 else 0.0
    with open(path, "w", encoding="utf-8") as f:
        f.write("=== GPU-only elapsed time (ms) ===\n")
        f.write(f"no_replace : {gpu_time['no']:.2f}\n")
        f.write(f"replace    : {gpu_time['rep']:.2f}\n")
        f.write(f"zero       : {gpu_time['zero']:.2f}\n")
        f.write("-------------------------------\n")
        f.write(f"total calls: {gpu_calls}\n")
        f.write(f"avg / call : {avg:.2f}\n")


def _log_first_prompt_set(records, build_prompt_sets, args) -> None:
    if not records:
        print("No records available for debug logging.")
        return

    first_record = None
    first_idx = None
    first_outputs = None
    for idx, record in enumerate(records):
        outputs = extract_model_outputs(record)
        if outputs:
            first_record = record
            first_idx = idx
            first_outputs = outputs
            break

    if not first_record or not first_outputs:
        print("No model outputs available for debug logging.")
        return

    model_output = first_outputs[0]
    doc_id = get_record_id(first_record, first_idx)
    print("=== [First PromptSet(s)] ===")
    print(f"task={args.task} style={args.style} model={args.model} id={doc_id}")
    print("--- model_output[0] ---")
    print(model_output)
    prompt_sets = list(build_prompt_sets(first_record, model_output, first_idx))
    if not prompt_sets:
        print("No prompt sets produced for the first record.")
        return
    print("--- no_replace_prompt (first) ---")
    print(prompt_sets[0].no_replace_prompt)
    print("--- zero_prompt (first) ---")
    print(prompt_sets[0].zero_prompt)
    for idx, prompt_set in enumerate(prompt_sets):
        print(f"--- replace_prompt[{idx}] ---")
        print(prompt_set.replace_prompt)
        print(f"--- answer[{idx}] ---")
        print(prompt_set.answer)
    print("=== [End First PromptSet(s)] ===")


# ---------------------------- GPQA ----------------------------

def _gpqa_generate_prompt_from_examples(json_data, replace_idx, with_explanations=True):
    output = ""
    for idx, q in enumerate(json_data):
        output += f"Question: {q['question']}\nChoices:\n"
        for choice, value in q["choices"].items():
            output += f"({choice}) {value}\n"
        if with_explanations:
            output += f"Let's think step by step: \n{q['explanation']}\n"
        if idx != replace_idx:
            output += f"The correct answer is ({q['correct_answer']})\n"
    return output


def _gpqa_chain_of_thought_prompt(json_data, replace_idx, example):
    prompt = (
        "Here are some example questions from experts. "
        "An explanation is given before the final answer. "
        "Answer the final question yourself, giving your reasoning beforehand.\n"
    )
    prompt += _gpqa_generate_prompt_from_examples(json_data, replace_idx, with_explanations=True)
    prompt += f"Question: {example['question']}"
    prompt += (
        f"\nChoices:\n(A) {example['choices']['A']}\n(B) {example['choices']['B']}"
        f"\n(C) {example['choices']['C']}\n(D) {example['choices']['D']}"
    )
    prompt += (
        "\nGive step by step reasoning before you answer, and when you're ready to answer, "
        "please use the format \"The correct answer is (insert answer here)\":\n"
    )
    return prompt


def _gpqa_zero_shot_prompt(example):
    prompt = f"What is the correct answer to this question: {example['question']}"
    prompt += (
        f"\nChoices:\n(A) {example['choices']['A']}\n(B) {example['choices']['B']}"
        f"\n(C) {example['choices']['C']}\n(D) {example['choices']['D']}"
    )
    prompt += (
        "\nGive step by step reasoning before you answer, and when you're ready to answer, "
        "please use the format \"The correct answer is (insert answer here)\":\n"
    )
    return prompt


def _gpqa_llama_generate_prompt(json_data, replace_idx, with_explanations=True):
    output = ""
    for idx, q in enumerate(json_data):
        output += f"{LLAMA_USER}\n"
        output += (
            "Here are some example questions from experts. An explanation is given before the final answer. "
            "Answer the final question yourself, giving your reasoning beforehand.\n"
        )
        output += f"Question: {q['question']}\nChoices:\n"
        for choice, value in q["choices"].items():
            output += f"({choice}) {value}\n"
        output += (
            "\nGive step by step reasoning before you answer, and when you're ready to answer, "
            "please use the format \"The correct answer is (insert answer here)\":\n"
        )
        if with_explanations:
            output += f"Let's think step by step:{LLAMA_EOT}{LLAMA_ASSISTANT} \n{q['explanation']}"
        if idx != replace_idx:
            output += f"\nThe correct answer is ({q['correct_answer']}){LLAMA_EOT}\n"
        else:
            output += f"{LLAMA_EOT}\n"
    return output


def _gpqa_llama_chain_prompt(json_data, replace_idx, example):
    prompt = _gpqa_llama_generate_prompt(json_data, replace_idx, with_explanations=True)
    prompt += f"{LLAMA_USER}\n"
    prompt += (
        "Here are some example questions from experts. An explanation is given before the final answer. "
        "Answer the final question yourself, giving your reasoning beforehand.\n"
    )
    prompt += f"Question: {example['question']}"
    prompt += (
        f"\nChoices:\n(A) {example['choices']['A']}\n(B) {example['choices']['B']}"
        f"\n(C) {example['choices']['C']}\n(D) {example['choices']['D']}"
    )
    prompt += (
        "\nGive step by step reasoning before you answer, and when you're ready to answer, "
        "please use the format \"The correct answer is (insert answer here)\":\n"
    )
    prompt += "Let's think step by step:"
    return prompt


def _gpqa_llama_zero_prompt(example):
    prompt = f"What is the correct answer to this question: {example['question']}"
    prompt += (
        f"\nChoices:\n(A) {example['choices']['A']}\n(B) {example['choices']['B']}"
        f"\n(C) {example['choices']['C']}\n(D) {example['choices']['D']}"
    )
    prompt += (
        "\nGive step by step reasoning before you answer, and when you're ready to answer, "
        "please use the format \"The correct answer is (insert answer here)\":\n"
    )
    prompt += "Let's think step by step:"
    return prompt


# ---------------------------- DROP / HOTPOTQA ----------------------------

def _build_qa_block(fewshot):
    return "".join(f"Q: {qa['Q']}\nA: {qa['A']}\n\n" for qa in fewshot)


def _drop_end_prompt():
    return "\n\nEnd your answer with \"So the answer is <answer>\". Think step by step.\n\nA: "


def _hotpot_end_prompt():
    return "\n\nEnd your answer with \"Answer <answer>\". Think step by step.\n\nA: "


# ---------------------------- MATH500 ----------------------------

def _math500_system_prompt():
    return "Please reason step by step, and put your final answer within \\boxed{{}}.\n\n"


def _math500_llama_construct_prompt(few_shot_examples, query):
    start_prompt = (
        f"{LLAMA_USER}\n\n"
        "Solve the following math problem efficiently and clearly:\n\n"
        "- For simple problems (2 steps or fewer):\n"
        "Provide a concise solution with minimal explanation.\n\n"
        "- For complex problems (3 steps or more):\n"
        "Use this step-by-step format:\n\n"
        "## Step 1: [Concise description]\n"
        "[Brief explanation and calculations]\n\n"
        "## Step 2: [Concise description]\n"
        "[Brief explanation and calculations]\n\n"
        "...Regardless of the approach, always conclude with:\n\n"
        "Therefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\n"
        "Where [answer] is just the final number or expression that solves the problem.\n\n"
    )
    prompt = ""
    if few_shot_examples is not None:
        for example in few_shot_examples:
            prompt += (
                start_prompt
                + "Problem: "
                + example["problem"]
                + f"{LLAMA_EOT}{LLAMA_ASSISTANT}\n\n"
            )
            prompt += example["solution"] + LLAMA_EOT
    prompt += (
        start_prompt
        + "Problem: "
        + query
        + f"{LLAMA_EOT}{LLAMA_ASSISTANT}\n"
    )
    return prompt


# ---------------------------- MMLU-PRO ----------------------------

def _format_mmlu_example(question, options, cot_content=""):
    if cot_content == "":
        cot_content = "Let's think step by step."
    if cot_content.startswith("A: "):
        cot_content = cot_content[3:]
    example = f"Question: {question}\nOptions: "
    choice_map = "ABCDEFGHIJ"
    for i, opt in enumerate(options):
        example += f"{choice_map[i]}. {opt}\n"
    if cot_content == "":
        example += "Answer: "
    else:
        example += "Answer: " + cot_content + "\n\n"
    return example


def _remove_paren_after_answer(cot_content: str) -> str:
    import re

    pattern = re.compile(r"(The answer is )\(([^)]+)\)(\.)?")
    return pattern.sub(r"The best answer is \2\3", cot_content)


def _format_mmlu_llama_example(start_prompt, question, options, cot_content="", end_prompt=""):
    text = "A: Let's think step by step. "
    if cot_content.startswith(text):
        cot_content = cot_content[len(text) :]
    example = start_prompt + f"Question: {question}\n"
    choice_map = "ABCDEFGHIJ"
    for i in range(len(choice_map)):
        if i < len(options):
            example += f"{choice_map[i]}. {options[i]}\n"
        else:
            example += f"{choice_map[i]}. N/A\n"
    example += end_prompt
    example += "\n\n" + cot_content + LLAMA_EOT
    return example


def _format_mmlu_llama_question(start_prompt, question, options, end_prompt):
    example = start_prompt + f"Question: {question}\n"
    choice_map = "ABCDEFGHIJ"
    for i in range(len(choice_map)):
        if i < len(options):
            example += f"{choice_map[i]}. {options[i]}\n"
        else:
            example += f"{choice_map[i]}. N/A\n"
    example += end_prompt
    return example


# ---------------------------- MuSR ----------------------------

def _musr_system_prompt(entry_data, eval_kind: str):
    if eval_kind == "plan":
        return (
            "Let’s first understand the story, extract each person’s suitability (great/ok/bad) and pair compatibilities, "
            "and make a complete plan. Then, carry out the plan step by step, compare options (A|B|C), pick the best, "
            "and end with ANSWER: <A|B|C>.\n"
        )
    if eval_kind == "role":
        return (
            "From now on, you are an excellent operations manager who assigns people to tasks correctly. "
            "Use only story facts and consider pair compatibility. "
            "Put your final line as ANSWER: <A|B|C>.\n"
        )
    return entry_data.get("prompt_parts", {}).get("cot_system_prompt") or ""


def main(args):
    tokenizer, model, device = load_llm(args.model_name, args.hf_token)

    input_path = _get_input_path(args)
    output_dir = _get_output_dir(args)
    os.makedirs(output_dir, exist_ok=True)

    records = read_jsonl(input_path, allow_errors=True)
    if not records:
        print(f"No input records found at {input_path}")
        return

    likelihoods_path = os.path.join(output_dir, "all_likelihoods.json")
    existing = load_existing_likelihoods(likelihoods_path)

    if args.task == "gpqa":
        examples = load_examples("data/gpqa/gpqa_diamond.csv", seed=0)
        with open("tasks/gpqa/chain_of_thought_examples.json", "r", encoding="utf-8") as f:
            json_data = json.load(f)["questions"]

        if args.style == "gpt":
            system_prompt = "You are a very intelligent assistant, who follows instructions directly.\n\n"

            def build_prompt_sets(entry, model_output, idx):
                example = examples[idx]
                choices = {
                    "A": example.choice1,
                    "B": example.choice2,
                    "C": example.choice3,
                    "D": example.choice4,
                }
                question = example.question
                for replace_idx in range(5):
                    new_json_data = copy.deepcopy(json_data)
                    fewshot_example = json_data[replace_idx]
                    new_json_data[replace_idx]["question"] = question
                    new_json_data[replace_idx]["choices"] = choices
                    new_json_data[replace_idx]["explanation"] = model_output

                    answer_part = (
                        f"Let's think step by step: \n{fewshot_example['explanation']}\n"
                        f"The correct answer is ({fewshot_example['correct_answer']})"
                    )

                    replace_prompt = system_prompt + _gpqa_chain_of_thought_prompt(new_json_data, replace_idx, fewshot_example)
                    no_replace_prompt = system_prompt + _gpqa_chain_of_thought_prompt([new_json_data[replace_idx]], 0, fewshot_example)
                    zero_prompt = system_prompt + _gpqa_zero_shot_prompt(fewshot_example)

                    yield PromptSet(no_replace_prompt, replace_prompt, zero_prompt, answer_part)

            def record_factory(doc_id, model_output, answer, _idx):
                return {
                    "id": doc_id,
                    "model_output": model_output,
                    "no_replace_likelihoods": [],
                    "replace_likelihoods": [],
                    "zero_likelihoods": [],
                }

            def get_answer(_entry, idx):
                example = examples[idx]
                return example[example.correct_index + 1]

        else:
            system_prompt = (
                f"{LLAMA_SYSTEM}\n"
                f"You are a very intelligent assistant, who follows instructions directly.{LLAMA_EOT}\n\n"
            )
            start_prompt = f"{LLAMA_USER}\n"
            end_prompt = f"{LLAMA_EOT}{LLAMA_ASSISTANT}\n\n"

            def build_prompt_sets(entry, model_output, idx):
                example = examples[idx]
                choices = {
                    "A": example.choice1,
                    "B": example.choice2,
                    "C": example.choice3,
                    "D": example.choice4,
                }
                question = example.question
                for replace_idx in range(5):
                    new_json_data = copy.deepcopy(json_data)
                    fewshot_example = json_data[replace_idx]
                    new_json_data[replace_idx]["question"] = question
                    new_json_data[replace_idx]["choices"] = choices
                    new_json_data[replace_idx]["explanation"] = model_output

                    answer_part = (
                        f"Let's think step by step: \n{fewshot_example['explanation']}\n"
                        f"The correct answer is ({fewshot_example['correct_answer']})"
                    )

                    replace_prompt = system_prompt + _gpqa_llama_chain_prompt(new_json_data, replace_idx, fewshot_example) + end_prompt
                    no_replace_prompt = system_prompt + _gpqa_llama_chain_prompt([new_json_data[replace_idx]], 0, fewshot_example) + end_prompt
                    zero_prompt = system_prompt + start_prompt + _gpqa_llama_zero_prompt(fewshot_example) + end_prompt

                    yield PromptSet(no_replace_prompt, replace_prompt, zero_prompt, answer_part)

            def record_factory(doc_id, model_output, answer, _idx):
                return {
                    "id": doc_id,
                    "model_output": model_output,
                    "no_replace_likelihoods": [],
                    "replace_likelihoods": [],
                    "zero_likelihoods": [],
                }

            def get_answer(entry, _idx):
                return entry.get("target")

        if args.log_first:
            _log_first_prompt_set(records, build_prompt_sets, args)

        result = run_likelihood(
            records,
            build_prompt_sets,
            record_factory,
            get_answer,
            model=model,
            tokenizer=tokenizer,
            device=device,
            likelihoods_path=likelihoods_path,
            existing_likelihoods=existing,
        )
        _write_gpu_times(os.path.join(output_dir, "gpu_times.txt"), result.gpu_time, result.gpu_calls)
        return

    if args.task in {"drop", "hotpotqa"}:
        is_drop = args.task == "drop"
        fewshot_path = "tasks/drop/prompt.json" if is_drop else "tasks/hotpotqa/react_prompt.json"
        with open(fewshot_path, "r", encoding="utf-8") as f:
            fewshot = json.load(f)

        end_prompt = _drop_end_prompt() if is_drop else _hotpot_end_prompt()

        if args.style == "llama":
            start_prompt = f"{LLAMA_USER}\n"
            end_prompt_llama = (
                f"\n\nEnd your answer with \"Answer <answer>\". Think step by step.{LLAMA_EOT}"
                f"{LLAMA_ASSISTANT}\n\n"
            )

            def create_final_prompt(question, shot):
                input_final_prompt = ""
                for qa in shot:
                    input_final_prompt += (
                        f"{start_prompt}Q: {qa['Q']}{end_prompt_llama}A: {qa['A']}{LLAMA_EOT}\n"
                    )
                input_final_prompt += f"{start_prompt}Q: {question}{end_prompt_llama}"
                return input_final_prompt

            def build_prompt_sets(entry, model_output, _idx):
                entry_data = entry.get("entry") or entry.get("doc") or {}
                question = entry_data.get("question", "")
                passage = entry_data.get("passage", "")

                for replace_idx in range(len(fewshot)):
                    new_fewshot = copy.deepcopy(fewshot)
                    question_part = fewshot[replace_idx]["Q"]
                    answer_part = fewshot[replace_idx]["A"]

                    if is_drop:
                        new_fewshot[replace_idx]["Q"] = f"{passage} {question}".strip()
                    else:
                        new_fewshot[replace_idx]["Q"] = question
                    new_fewshot[replace_idx]["A"] = model_output

                    replace_prompt = create_final_prompt(question_part, new_fewshot)
                    no_replace_prompt = (
                        f"{start_prompt}Q: {passage}{question}{end_prompt_llama}A: {model_output}{LLAMA_EOT}\n"
                        f"{start_prompt}Q: {question_part}{end_prompt_llama}"
                    )
                    zero_prompt = start_prompt + f"Q: {question_part}" + end_prompt_llama

                    yield PromptSet(
                        no_replace_prompt,
                        replace_prompt,
                        zero_prompt,
                        answer_part,
                        no_key="recost_likelihoods",
                    )

            def record_factory(doc_id, model_output, _answer, _idx):
                return {
                    "id": doc_id,
                    "model_output": model_output,
                    "recost_likelihoods": [],
                    "replace_likelihoods": [],
                    "zero_likelihoods": [],
                }

        else:
            def build_prompt_sets(entry, model_output, _idx):
                entry_data = entry.get("entry") or entry.get("doc", {}).get("entry", {}) or entry.get("doc", {})
                question = entry_data.get("question", "")
                passage = entry_data.get("passage", "")

                for replace_idx in range(len(fewshot)):
                    new_fewshot = copy.deepcopy(fewshot)
                    question_part = fewshot[replace_idx]["Q"]
                    answer_part = fewshot[replace_idx]["A"]

                    if is_drop:
                        new_fewshot[replace_idx]["Q"] = f"{passage} {question}".strip()
                    else:
                        new_fewshot[replace_idx]["Q"] = question
                    new_fewshot[replace_idx]["A"] = model_output

                    replace_prompt = _build_qa_block(new_fewshot) + f"Q: {question_part}" + end_prompt
                    if is_drop:
                        no_replace_prompt = (
                            f"Q: {passage}{question}\nA: {model_output}\n\n" + f"Q: {question_part}" + end_prompt
                        )
                    else:
                        no_replace_prompt = (
                            f"Q: {question}\nA:{model_output}\n\n" + f"Q: {question_part}" + end_prompt
                        )
                    zero_prompt = f"Q: {question_part}" + end_prompt

                    yield PromptSet(no_replace_prompt, replace_prompt, zero_prompt, answer_part)

            def record_factory(doc_id, model_output, _answer, _idx):
                return {
                    "id": doc_id,
                    "model_output": model_output,
                    "no_replace_likelihoods": [],
                    "replace_likelihoods": [],
                    "zero_likelihoods": [],
                }

        def get_answer(entry, _idx):
            entry_data = entry.get("entry") or entry.get("doc", {}).get("entry", {}) or entry.get("doc", {})
            return entry_data

        if args.log_first:
            _log_first_prompt_set(records, build_prompt_sets, args)

        result = run_likelihood(
            records,
            build_prompt_sets,
            record_factory,
            get_answer,
            model=model,
            tokenizer=tokenizer,
            device=device,
            likelihoods_path=likelihoods_path,
            existing_likelihoods=existing,
        )
        _write_gpu_times(os.path.join(output_dir, "gpu_times.txt"), result.gpu_time, result.gpu_calls)
        return

    if args.task == "math500":
        if args.style == "gpt":
            system_prompt = _math500_system_prompt()
            fewshot = load_prompt(num_shots=5)

            def build_prompt_sets(entry, model_output, _idx):
                entry_data = entry.get("entry") or entry.get("doc", {}).get("entry", {}) or entry.get("doc", {})
                question = entry_data.get("problem", "")
                for replace_idx in range(5):
                    modified_prompt = fewshot.copy()
                    question_part, answer_part = modified_prompt[replace_idx]
                    modified_prompt[replace_idx] = (question, model_output)
                    replace_prompt = (
                        system_prompt
                        + "\n\n".join([f"{q}\n\n{a}" for q, a in modified_prompt])
                        + "\n\n"
                        + question_part
                        + "\n"
                    )
                    no_replace_prompt = (
                        system_prompt + question + "\n\n" + model_output + "\n\n" + question_part + "\n"
                    )
                    zero_prompt = system_prompt + question_part + "\n"

                    yield PromptSet(no_replace_prompt, replace_prompt, zero_prompt, answer_part)

            def record_factory(doc_id, model_output, _answer, _idx):
                return {
                    "id": doc_id,
                    "model_output": model_output,
                    "no_replace_likelihoods": [],
                    "replace_likelihoods": [],
                    "zero_likelihoods": [],
                }

            def get_answer(entry, _idx):
                entry_data = entry.get("entry") or entry.get("doc", {}).get("entry", {}) or entry.get("doc", {})
                _, answer = parse_ground_truth(entry_data, "math")
                return answer

        else:
            def build_prompt_sets(entry, model_output, _idx):
                entry_data = entry.get("doc", {})
                question = entry_data.get("problem", "")
                for replace_idx in range(4):
                    modified_prompt = list_fewshot_samples()
                    question_part = modified_prompt[replace_idx]["problem"]
                    answer_part = modified_prompt[replace_idx]["solution"]
                    modified_prompt[replace_idx]["problem"] = question
                    modified_prompt[replace_idx]["solution"] = model_output

                    replace_prompt = _math500_llama_construct_prompt(modified_prompt, question_part)
                    no_replace_prompt = _math500_llama_construct_prompt([modified_prompt[replace_idx]], question_part)
                    zero_prompt = _math500_llama_construct_prompt(None, question_part)

                    yield PromptSet(no_replace_prompt, replace_prompt, zero_prompt, answer_part)

            def record_factory(doc_id, model_output, _answer, _idx):
                return {
                    "id": doc_id,
                    "model_output": model_output,
                    "no_replace_likelihoods": [],
                    "replace_likelihoods": [],
                    "zero_likelihoods": [],
                }

            def get_answer(entry, _idx):
                return entry.get("doc", {}).get("answer")

        if args.log_first:
            _log_first_prompt_set(records, build_prompt_sets, args)

        result = run_likelihood(
            records,
            build_prompt_sets,
            record_factory,
            get_answer,
            model=model,
            tokenizer=tokenizer,
            device=device,
            likelihoods_path=likelihoods_path,
            existing_likelihoods=existing,
        )
        _write_gpu_times(os.path.join(output_dir, "gpu_times.txt"), result.gpu_time, result.gpu_calls)
        return

    if args.task == "mmlu_pro":
        _, dev_df = load_mmlu_pro()

        if args.style == "gpt":
            def build_prompt_sets(entry, model_output, _idx):
                entry_data = entry.get("entry") or entry.get("doc", {}).get("entry", {}) or entry.get("doc", {})
                question = entry_data.get("question")
                options = entry_data.get("options")
                model_example = _format_mmlu_example(question, options, model_output)

                for replace_idx in range(5):
                    base_prompt = (
                        "The following are multiple choice questions (with answers) about {}. Think step by"
                        " step and then output the answer in the format of \"The answer is (X)\" at the end.\n\n"
                    ).format(args.subject)

                    few_shot = dev_df[args.subject][replace_idx]
                    question_part = _format_mmlu_example(few_shot["question"], few_shot["options"])
                    answer_part = few_shot["cot_content"]
                    if answer_part.startswith("A: "):
                        answer_part = answer_part[3:]

                    no_replace_prompt = base_prompt + model_example + question_part
                    zero_prompt = base_prompt + question_part

                    replace_prompt = base_prompt
                    for i, each in enumerate(dev_df[args.subject]):
                        if i != replace_idx:
                            replace_prompt += _format_mmlu_example(each["question"], each["options"], each["cot_content"])
                        else:
                            replace_prompt += model_example
                    replace_prompt = replace_prompt + question_part

                    yield PromptSet(no_replace_prompt, replace_prompt, zero_prompt, answer_part)

            def record_factory(doc_id, model_output, _answer, _idx):
                return {
                    "id": doc_id,
                    "model_output": model_output,
                    "no_replace_likelihoods": [],
                    "replace_likelihoods": [],
                    "zero_likelihoods": [],
                }

            def get_answer(entry, _idx):
                entry_data = entry.get("entry") or entry.get("doc", {}).get("entry", {}) or entry.get("doc", {})
                return entry_data.get("answer")

        else:
            start_prompt = (
                f"{LLAMA_USER}\n\n"
                "Given the following question and candidate answers, choose the best answer.\n"
            )
            end_prompt = (
                "\nYour response should end with \"The best answer is [the_answer_letter].\" "
                "where the [the_answer_letter] is a letter from the provided choices.\n\n"
                f"Let's think step by step.{LLAMA_EOT}{LLAMA_ASSISTANT}"
            )

            def build_prompt_sets(entry, model_output, _idx):
                entry_data = entry.get("doc", {})
                question = entry_data.get("problem")
                options = entry_data.get("input_choice_list")

                model_example = _format_mmlu_llama_example(start_prompt, question, list(options), model_output, end_prompt)

                for replace_idx in range(5):
                    few_shot = dev_df[args.subject][replace_idx]
                    question_part = _format_mmlu_llama_question(start_prompt, few_shot["question"], few_shot["options"], end_prompt)
                    answer_part = _remove_paren_after_answer(few_shot["cot_content"])
                    if answer_part.startswith("A: "):
                        answer_part = answer_part[len("A: Let's think step by step. ") :]

                    no_replace_prompt = model_example + question_part
                    zero_prompt = question_part

                    prompt = ""
                    for i, each in enumerate(dev_df[args.subject]):
                        if i != replace_idx:
                            prompt += _format_mmlu_llama_example(
                                start_prompt,
                                each["question"],
                                each["options"],
                                _remove_paren_after_answer(each["cot_content"]),
                                end_prompt,
                            )
                        else:
                            prompt += model_example
                    replace_prompt = prompt + question_part

                    yield PromptSet(
                        no_replace_prompt,
                        replace_prompt,
                        zero_prompt,
                        answer_part,
                        no_key="recost_likelihoods",
                    )

            def record_factory(doc_id, model_output, _answer, _idx):
                return {
                    "id": doc_id,
                    "model_output": model_output,
                    "recost_likelihoods": [],
                    "replace_likelihoods": [],
                    "zero_likelihoods": [],
                }

            def get_answer(entry, _idx):
                return entry.get("doc", {}).get("gold")

        if args.log_first:
            _log_first_prompt_set(records, build_prompt_sets, args)

        result = run_likelihood(
            records,
            build_prompt_sets,
            record_factory,
            get_answer,
            model=model,
            tokenizer=tokenizer,
            device=device,
            likelihoods_path=likelihoods_path,
            existing_likelihoods=existing,
        )
        _write_gpu_times(os.path.join(output_dir, "gpu_times.txt"), result.gpu_time, result.gpu_calls)
        return

    if args.task in {"musr_efficiently", "musr_location"}:
        if args.task == "musr_location":
            few_shot = OBJECT_PLACEMENTS_FEWSHOT
            few_instruction = OBJECT_PLACEMENTS_FEWSHOT_INSTRUCTION
            test_instruction = OBJECT_PLACEMENTS_TEST_INSTRUCTION
            dataset = MuSRDataset("data/musr/object_placements.json")
        else:
            few_shot = TEAM_ALLOCATION_FEWSHOT
            few_instruction = TEAM_ALLOCATION_FEWSHOT_INSTRUCTION
            test_instruction = TEAM_ALLOCATION_TEST_INSTRUCTION
            dataset = MuSRDataset("data/musr/team_allocation.json")

        if args.style == "gpt":
            def build_prompt_sets(entry, model_output, _idx):
                entry_data = entry.get("entry") or entry.get("doc", {}).get("entry", {}) or entry.get("doc", {})
                question = (entry_data.get("question") or "").strip()
                context = (entry_data.get("context") or "").strip()
                choices = (entry_data.get("choices") or {}).get("text") or []
                labels = ["A", "B", "C", "D", "E", "F"][: len(choices)]

                system_prompt = entry_data.get("prompt_parts", {}).get("cot_system_prompt") or ""
                choice_str = "\n".join([f"{labels[i]}: {choices[i]}" for i in range(len(choices))])
                question_block = f"{context}\n\n{question}\n\n{choice_str}"

                for replace_idx in range(3):
                    original_question_part, answer_part = few_shot[replace_idx]
                    replaced_few_shot = few_shot.copy()
                    replaced_few_shot[replace_idx] = (question_block, model_output)

                    replaced_blocks = []
                    for (q, a) in replaced_few_shot:
                        block = f"{q}\n\n{few_instruction}\n\n{a}"
                        replaced_blocks.append(block)

                    replace_prompt = f"{system_prompt}\n\n" + "\n\n".join(replaced_blocks)
                    replace_prompt += f"\n\n{original_question_part}\n\n{test_instruction}"

                    no_replace_prompt = f"{system_prompt}\n\n{question_block}\n\n{few_instruction}\n\n{model_output}"
                    no_replace_prompt += f"\n\n{original_question_part}\n\n{test_instruction}"

                    zero_prompt = f"{system_prompt}\n\n{original_question_part}\n\n{test_instruction}"

                    yield PromptSet(no_replace_prompt, replace_prompt, zero_prompt, answer_part)

            def record_factory(doc_id, model_output, _answer, _idx):
                return {
                    "id": doc_id,
                    "model_output": model_output,
                    "no_replace_likelihoods": [],
                    "replace_likelihoods": [],
                    "zero_likelihoods": [],
                    "remaining_likelihoods": [],
                }

            def get_answer(entry, _idx):
                entry_data = entry.get("entry") or entry.get("doc", {}).get("entry", {}) or entry.get("doc", {})
                return entry_data.get("answer")

        else:
            def build_prompt_sets(entry, model_output, _idx):
                doc_id = entry.get("doc_id")
                if doc_id is None:
                    return []
                example = dataset[doc_id]
                question = example["question"].strip()
                context = example["context"].strip()
                choices = example["choices"]["text"]
                labels = ["A", "B", "C", "D", "E", "F"][: len(choices)]
                choice_str = "\n".join([f"{labels[i]}: {choices[i]}" for i in range(len(choices))])
                question_block = f"{context}\n\n{question}\n\n{choice_str}"
                system_prompt = example["prompt_parts"]["cot_system_prompt"]

                for replace_idx in range(3):
                    original_question_part, answer_part = few_shot[replace_idx]

                    zero_prompt = f"{LLAMA_USER}\n"
                    zero_prompt += system_prompt + "\n\n" + original_question_part + "\n\n" + test_instruction + "\n"
                    zero_prompt += f"{LLAMA_EOT}\n"
                    zero_prompt += f"{LLAMA_ASSISTANT}\n"

                    replaced_few_shot = few_shot.copy()
                    replaced_few_shot[replace_idx] = (question_block, model_output)

                    replace_prompt = ""
                    for (q, a) in replaced_few_shot:
                        replace_prompt += f"{LLAMA_USER}\n"
                        replace_prompt += system_prompt + "\n\n" + q + "\n\n" + few_instruction + "\n"
                        replace_prompt += f"{LLAMA_EOT}\n"
                        replace_prompt += f"{LLAMA_ASSISTANT}\n"
                        replace_prompt += a + "\n"
                        replace_prompt += f"{LLAMA_EOT}\n"

                    replace_prompt += zero_prompt

                    no_replace_prompt = f"{LLAMA_USER}\n"
                    no_replace_prompt += system_prompt + "\n\n" + question_block + "\n\n" + few_instruction + "\n"
                    no_replace_prompt += f"{LLAMA_EOT}\n"
                    no_replace_prompt += f"{LLAMA_ASSISTANT}\n"
                    no_replace_prompt += model_output + "\n"
                    no_replace_prompt += f"{LLAMA_EOT}\n"
                    no_replace_prompt += zero_prompt

                    yield PromptSet(no_replace_prompt, replace_prompt, zero_prompt, answer_part)

            def record_factory(doc_id, model_output, _answer, _idx):
                return {
                    "id": doc_id,
                    "model_output": model_output,
                    "no_replace_likelihoods": [],
                    "replace_likelihoods": [],
                    "zero_likelihoods": [],
                    "remaining_likelihoods": [],
                }

            def get_answer(entry, _idx):
                return entry.get("target")

        if args.log_first:
            _log_first_prompt_set(records, build_prompt_sets, args)

        result = run_likelihood(
            records,
            build_prompt_sets,
            record_factory,
            get_answer,
            model=model,
            tokenizer=tokenizer,
            device=device,
            likelihoods_path=likelihoods_path,
            existing_likelihoods=existing,
        )
        _write_gpu_times(os.path.join(output_dir, "gpu_times.txt"), result.gpu_time, result.gpu_calls)
        return

    print(f"Unsupported task: {args.task}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--style", type=str, choices=["gpt", "llama"], default="gpt")
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--output_dir", type=str, default="backward")
    parser.add_argument("--input_dir", type=str, default="result")
    parser.add_argument("--subject", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf_token", type=str, default=os.getenv("HUGGINGFACE_TOKEN"))
    parser.add_argument("--log_first", action="store_true", help="print the first prompt set for debugging")

    args = parser.parse_args()
    main(args)
