import os
import json
import random
import argparse
import pandas as pd
from tqdm import tqdm 

from dotenv import load_dotenv
from openai import OpenAI

from tasks.gpqa.gpqa_utils import * 

from tasks.math500.math_utils import * 
from tasks.math500.parser import *
from tasks.math500.grader import * 

from tasks.mmlu_pro.mmlu_utils import * 

from tasks.hotpotqa.hotpotqa_utils import *

from tasks.drop.drop_utils import *

from tasks.musr.musr_ import MuSRDataset

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)


class Config:
    def __init__(self):
        self.model = "gpt-4o"  
        self.task = "drop"  
        self.shot_type = "few" 
        self.output_dir = "result"
        self.num_examples = -1
        self.temperature = 1
        self.n = 5

def parse_args() -> argparse.Namespace:
    defaults = Config()
    parser = argparse.ArgumentParser(description="Generate prompts and model outputs.")
    parser.add_argument("--model", default=defaults.model, help="Single model name.")
    parser.add_argument("--models", default=None, help="Comma-separated list of models.")
    parser.add_argument("--task", default=defaults.task, help="Single task name.")
    parser.add_argument("--tasks", default=None, help="Comma-separated list of tasks.")
    parser.add_argument("--shot-type", dest="shot_type", default=defaults.shot_type, help="Shot type.")
    parser.add_argument("--shots", default=None, help="Comma-separated list of shot types.")
    parser.add_argument("--output-dir", default=defaults.output_dir, help="Output directory.")
    parser.add_argument("--num-examples", type=int, default=defaults.num_examples, help="Limit examples.")
    parser.add_argument("--temperature", type=float, default=defaults.temperature, help="Sampling temperature.")
    parser.add_argument("--n", type=int, default=defaults.n, help="Number of generations per prompt.")
    # parser.add_argument("--seed", type=int, default=defaults.seed, help="Random seed.")
    return parser.parse_args()

def split_csv(value: str) -> list:
    return [item.strip() for item in value.split(",") if item.strip()]

def load_data_and_fewshot(args):
    if args.task == "mmlu_pro":
        dataset, fewshot = load_mmlu_pro()

    elif args.task == "math500" or args.task == "math500_role" or args.task == "math500_plan": 
        file_path = f"data/math500/test.jsonl"
        with open(file_path, 'r', encoding='utf-8') as f:
            dataset = [json.loads(line) for line in f]
        fewshot = load_prompt(num_shots=5)

    elif args.task == "gpqa" or args.task == "gpqa_role" or args.task == "gpqa_plan":
        dataset = load_examples("data/gpqa/gpqa_diamond.csv", seed=0)
        prompt_path = "tasks/gpqa/chain_of_thought_examples.json"
        with open(prompt_path, 'r') as f:
            fewshot = json.load(f)

    elif args.task == "hotpotqa":
        dataset = json.load(open(f'data/hotpotqa/{args.task}.json'))
        with open("tasks/hotpotqa/react_prompt.json", 'r') as f:
            fewshot = json.load(f)

    elif args.task == "drop":
        dataset = pd.read_parquet("data/drop/drop_sub.parquet", engine="pyarrow")
        dataset = dataset.to_dict(orient="records") 
        dataset = convert_ndarray_to_list(dataset)
        dataset = convert_ndarray_to_list(dataset)
        with open("tasks/drop/prompt.json", 'r') as f:
            fewshot = json.load(f)

    elif args.task in ["musr_efficiently", 'musr_efficiently_role', 'musr_efficiently_plan']:
        ta_path = 'data/musr/team_allocation.json'
        dataset = MuSRDataset(ta_path)
        fewshot = 1

    elif args.task == "musr_location":
        op_path = 'data/musr/object_placements.json'
        dataset = MuSRDataset(op_path)
        fewshot = 1
    else: 
        return None, None
    
    return dataset, fewshot

def construct_prompt(args, dataset, fewshot): 
    samples = []

    if args.task in ["math500", "math500_role", "math500_plan"]:
        if args.task == "math500":
            system_prompt = "Please reason step by step, and put your final answer within \\boxed{{}}.\n"
        elif args.task == "math500_role": 
            system_prompt = "From now on, you are an excellent math teacher and always teach your students math problems correctly. And I am one of your students. Put your final answer within \\boxed{{}}.\n"
        elif args.task == "math500_plan": 
            system_prompt = "Let’s first understand the problem, extract relevant variables and their corresponding numerals, and make a complete plan. Then, let’s carry out the plan, calculate intermediate variables (pay attention to correct numerical calculation and commonsense), solve the problem step by step, and put your final answer within \\boxed{{}}.\n"

        for idx, entry in tqdm(enumerate(dataset)): 
            user_prompt = ""
            if fewshot != None: 
                user_prompt = "\n\n".join([f"{q}\n\n{a}" for q, a in fewshot]) + "\n\n" 
            user_prompt += entry['problem'] + "\n"
            message = [{"role": "system","content": system_prompt},{"role": "user","content": user_prompt}]
            sample = {"idx": idx,"prompt": message,"entry": entry}
            samples.append(sample)

    elif args.task == "mmlu_pro":
        subjects = list(dataset.keys())
        for subject in tqdm(subjects): 
            user_prompt = "The following are multiple choice questions (with answers) about {}. Think step by" \
                " step and then output the answer in the format of \"The answer is (X)\" at the end.\n\n" \
            .format(subject)
            if fewshot != None:
                for each in fewshot[subject]:
                    user_prompt += format_example(each["question"], each["options"], each["cot_content"])
            random.seed(42)
            test_data = random.sample(dataset[subject], min(300, len(dataset[subject])))
    
            for entry in test_data:
                input_text = format_example(entry['question'], entry['options'])
                message = [{"role": "user", "content": user_prompt + input_text}]
                sample = {"idx": entry['question_id'],"prompt": message,"entry": entry}
                samples.append(sample)
    
    elif args.task in ["gpqa", "gpqa_role", "gpqa_plan"]: 
        if args.task == "gpqa": 
            system_prompt = "You are a very intelligent assistant, who follows instructions directly.\n\n"
        elif args.task == "gpqa_role": 
            system_prompt = "From now on, you are an excellent science professor who always explains graduate‑level biology, chemistry, and physics questions correctly. And I am one of your students.\n\n"
        elif args.task == "gpqa_plan":
            system_prompt = "Let’s first understand the question, extract the key scientific concepts, relevant facts, equations, and any numerals, and make a complete plan. Then, let’s carry out the plan, derive or calculate intermediate quantities (paying attention to correct reasoning, units, and commonsense), solve the question step by step, and show the answer.\n\n"

        samples = []
        for example_id, example in tqdm(enumerate(dataset)):
            if fewshot != None: 
                user_prompt = chain_of_thought_prompt(fewshot, example)
            else:
                user_prompt = zero_shot_chain_of_thought_prompt(example)
            
            message = [{"role": "system","content": system_prompt},{"role": "user","content": user_prompt}]

            sample = {"idx": example_id,"prompt": message,"entry": example}
            samples.append(sample)
    
    elif args.task == "hotpotqa": 
        system_prompt = ""

        samples = []
        if fewshot != None: 
            fewshot_prompt = ""
            for qa in fewshot:
                question = qa["Q"]
                answer = qa["A"]

                fewshot_prompt += f"Q: {question}\nA: {answer}\n\n"

        for idx, entry in tqdm(enumerate(dataset)):
            if fewshot != None: 
                user_prompt = fewshot_prompt + f"Q: {entry['question']}." + "\n\nEnd your answer with \"Answer <answer>\". Think step by step." + "\n\nA: " 
            else:
                user_prompt = f"Q: {entry['question']}." + "\n\nEnd your answer with \"Answer <answer>\". Think step by step." + "\n\nA: " 
            
            message = [{"role": "system","content": system_prompt},{"role": "user","content": user_prompt}]

            sample = {"idx": idx,"prompt": message,"entry": entry, "answer": entry['answers']}
            samples.append(sample)

    elif args.task == "drop": 
        system_prompt = ""

        samples = []
        if fewshot != None: 
            fewshot_prompt = ""
            for qa in fewshot:
                question = qa["Q"]
                answer = qa["A"]

                fewshot_prompt += f"Q: {question}\nA: {answer}\n\n"

        for idx, entry in tqdm(enumerate(dataset)):
            if fewshot != None: 
                user_prompt = fewshot_prompt + f"Q: {entry['passage']}{entry['question']}" + "\n\nEnd your answer with \"So the answer is <answer>\". Think step by step." + "\n\nA: " 
            else:
                user_prompt = f"Q: {entry['passage']}{entry['question']}" + "\n\nEnd your answer with \"So the answer is <answer>\". Think step by step." + "\n\nA: " 
            
            message = [{"role": "system","content": system_prompt},{"role": "user","content": user_prompt}]

            sample = {"idx": idx,"prompt": message,"entry": entry}
            samples.append(sample)

    elif args.task == "musr_efficiently" or args.task == "musr_location":
        for idx, entry in tqdm(enumerate(dataset)):
            if fewshot != None:
                message = entry['fs_cot_messages']
            else:
                message = entry['zs_cot_messages']
            sample = {"idx": idx, "prompt": message, "entry": entry}
            samples.append(sample)

    elif args.task in ['musr_efficiently_role', 'musr_efficiently_plan']: 
        from tasks.musr.icl import TEAM_ALLOCATION_FEWSHOT, TEAM_ALLOCATION_FEWSHOT_INSTRUCTION, TEAM_ALLOCATION_TEST_INSTRUCTION

        few_shot = TEAM_ALLOCATION_FEWSHOT
        few_instruction = TEAM_ALLOCATION_FEWSHOT_INSTRUCTION
        test_instruction = TEAM_ALLOCATION_TEST_INSTRUCTION

        if args.task == "musr_efficiently_role":
            system_prompt = (
                "From now on, you are an excellent operations manager who assigns people to tasks correctly. "
                "Use only story facts and consider pair compatibility. "
                "Put your final line as ANSWER: <A|B|C>.\n"
            )

        if args.task == "musr_efficiently_plan":
            system_prompt = (
                "Let’s first understand the story, extract each person’s suitability (great/ok/bad) and pair compatibilities, "
                "and make a complete plan. Then, carry out the plan step by step, compare options (A|B|C), pick the best, "
                "and end with ANSWER: <A|B|C>.\n"
            )

        for idx, entry in tqdm(enumerate(dataset)):

            question = entry['question'].strip()
            context = entry['context'].strip()
            choices = entry['choices']['text']
            labels = ['A','B','C','D','E','F'][0:len(choices)]

            choice_str = '\n'.join(
                        [f'{labels[idx]}: {choices[idx]}' for idx in range(len(choices))])
            
            question_block = f"{context}\n\n{question}\n\n{choice_str}"
        
            if few_shot != None: 
                blocks = []
                for (q, a) in few_shot:
                    block = f"{q}\n\n{few_instruction}\n\n{a}"
                    blocks.append(block)
            
                user_prompt = "\n\n".join(blocks)
                user_prompt += f"\n\n{question_block}\n\n{test_instruction}".strip()
            else: 
                user_prompt = f"{question_block}\n\n{test_instruction}".strip()

            message = [{"role": "system","content": system_prompt},{"role": "user","content": user_prompt}]

            sample = {"idx": idx, "prompt": message, "entry": entry}
            samples.append(sample)

    else:
        return None

    return samples

def generate_model_output(model: str, prompt: str, temperature: float = 1.0, n: int = 5) -> str:        
    responses = client.chat.completions.create(
        model=model,
        messages=prompt,
        n=n,
        temperature=temperature,
    )

    outputs = [choice.message.content for choice in responses.choices]


    return outputs 

def save_samples(samples: list, save_path: str, config: Config, task_label: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if samples:
        print(
            f"Model: {config.model} Task: {task_label}, Shot: {config.shot_type}, "
            f"Temperature: {config.temperature}, N: {config.n}"
        )
        print("-" * 50)
        prompt = samples[0]["prompt"]
        for message in prompt:
            print(f"Role:\n{message['role']}")
            print(f"Content:\n{message['content']}")
            print("-" * 50)
    else:
        print(
            f"No samples found for Task: {task_label}, Shot: {config.shot_type}, "
            f"Temperature: {config.temperature}, N: {config.n}"
        )

    existing_data = set()
    if os.path.exists(save_path):
        with open(save_path, 'r', encoding='utf-8') as f:
            existing_data = {json.loads(line)['idx'] for line in f}

    flag = 0
    if samples:
        with open(save_path, "a", encoding='utf-8') as f:
            for sample in tqdm(samples, total=len(samples)):
                if sample['idx'] in existing_data:
                    continue
                try:
                    model_outputs = generate_model_output(config.model, sample["prompt"], config.temperature, config.n)
                    sample["model_outputs"] = model_outputs
                    if flag == 0:
                        print(model_outputs[0])
                        print("-" * 50)
                        flag = 1
                    json.dump(sample, f)
                    f.write("\n")
                except Exception as e:
                    print(f"Error processing sample {sample['idx']}: {e}")
                    break
        print(f"Results saved to {save_path}")


def _mmlu_save_suffix(shot_type: str) -> str:
    if shot_type == "zero":
        return "_zero_result.jsonl"
    return "_result.jsonl"


def _save_mmlu_samples(samples: list, config: Config) -> None:
    grouped = {}
    for sample in samples:
        entry = sample.get("entry") or {}
        subject = entry.get("category") or entry.get("subject")
        if not subject:
            continue
        grouped.setdefault(subject, []).append(sample)

    for subject in sorted(grouped.keys()):
        subject_samples = grouped[subject]
        if config.num_examples != -1:
            subject_samples = subject_samples[: config.num_examples]
        save_path = (
            f"{config.output_dir}/{config.task}/{config.model}/"
            f"{subject}{_mmlu_save_suffix(config.shot_type)}"
        )
        save_samples(subject_samples, save_path, config, f"{config.task}/{subject}")

def main(config):
    dataset, fewshot = load_data_and_fewshot(config)

    if config.shot_type == "zero":
        fewshot = None
        
    samples = construct_prompt(config, dataset, fewshot)
    if config.task != "mmlu_pro" and config.num_examples != -1: 
        samples = samples[:config.num_examples] 

    if config.task == "mmlu_pro":
        _save_mmlu_samples(samples, config)
        return

    save_path = f"{config.output_dir}/{config.task}/{config.model}/{config.task}_{config.shot_type}.jsonl"
    save_samples(samples, save_path, config, config.task)


if __name__ == "__main__":
    args = parse_args()
    config = Config()
    config.output_dir = args.output_dir
    config.num_examples = args.num_examples
    config.temperature = args.temperature
    config.n = args.n

    models = split_csv(args.models) if args.models else [args.model]
    tasks = split_csv(args.tasks) if args.tasks else [args.task]
    shots = split_csv(args.shots) if args.shots else [args.shot_type]

    for model in models:
        for task in tasks:
            for shot in shots:
                config.model = model
                config.task = task
                config.shot_type = shot
                print("=" * 50)
                main(config)
