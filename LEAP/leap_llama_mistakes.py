import os
import json
import sys
from tqdm import tqdm 
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gpqa.gpqa_utils import * 

from math500.math_utils import * 
from math500.parser import *
from math500.grader import * 

from mmlu_pro.mmlu_utils import * 

from hotpotqa.hotpotqa_utils import *

from drop.drop_utils import *

from musr.musr import MuSRDataset
from musr.op_icl_fixed import op_fewshot, few_shot_op_instruction, test_op_instruction
from musr.ta_icl_fixed import ta_fewshot, few_shot_ta_instruction, test_ta_instruction

from utils import load_model_outputs


load_dotenv()

hf_token = os.getenv('HUGGINGFACE_TOKEN')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = "meta-llama/Llama-3.1-8B-Instruct" 
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", use_auth_token=hf_token)
model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if model.config.pad_token_id is None:
    model.config.pad_token_id = tokenizer.eos_token_id

def generate_model_output(prompt, max_new_tokens=5120):
  
    encoding = tokenizer(prompt, 
                         return_tensors="pt", 
                         padding=True, 
                         truncation=True)
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,      
            max_new_tokens=max_new_tokens,
            temperature=1,                    
            pad_token_id=tokenizer.pad_token_id 
        )


    gen_ids = output_ids[0][len(input_ids[0]):]
    output_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    return output_text.strip()

def load_data_and_fewshot(args):
    if args.task == "mmlu_pro":
        dataset, fewshot = load_mmlu_pro()
    elif args.task == "math500": 
        file_path = f"data/{args.task}/test.jsonl"
        with open(file_path, 'r', encoding='utf-8') as f:
            dataset = [json.loads(line) for line in f]

        fewshot = list_fewshot_samples()
    elif args.task == "gpqa":
        dataset = load_examples("data/gpqa/gpqa_diamond.csv", seed=0)
        with open("gpqa/chain_of_thought_examples.json", 'r') as f:
            fewshot = json.load(f)
    elif args.task == "hotpotqa":
        dataset = json.load(open(f'data/hotpotqa/{args.task}.json'))
        with open("hotpotqa/react_prompt.json", 'r') as f:
            fewshot = json.load(f)
    elif args.task == "drop":
        dataset = pd.read_parquet("data/drop/drop_sub.parquet", engine="pyarrow")
        dataset = dataset.to_dict(orient="records")  
        dataset = convert_ndarray_to_list(dataset)
        dataset = convert_ndarray_to_list(dataset)

        with open("drop/prompt.json", 'r') as f:
            fewshot = json.load(f)
    elif args.task == "musr_location":
        op_path = 'data/musr/object_placements.json'
        dataset = MuSRDataset(op_path)
        fewshot = op_fewshot
        
    elif args.task == 'musr_efficiently': 
        ta_path = 'data/musr/team_allocation.json'
        dataset = MuSRDataset(ta_path)
        fewshot = ta_fewshot
    else: 
        return None, None
    
    return dataset, fewshot

def construct_prompt(args, dataset, fewshot): 
    start_prompt = "<|start_header_id|>user<|end_header_id|>"
    samples = []
    system_prompt = ""
    
    if args.task == "math500":
        user_prompt = ""
        if fewshot != None: 
            start_prompt = "<|start_header_id|>user<|end_header_id|>\n\nSolve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...Regardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem.\n\n"
            for example in fewshot:
                message = start_prompt + "Problem: " + example["problem"] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>" + "\n\n" 
                sample = {"question": example["problem"], "answer": example["solution"], "prompt": message, "subject": "math500"}
                samples.append(sample)
        
    elif args.task == "mmlu_pro":
        def format_example(question, options):
            example = "{}\nOptions: ".format(question)
            choice_map = "ABCDEFGHIJ"
            for i, opt in enumerate(options):
                example += "{}. {}\n".format(choice_map[i], opt)
            return example
        
        subjects = list(dataset.keys())
        for subject in tqdm(subjects): 
            if fewshot != None:
                for each in fewshot[subject]:
                    user_prompt = "The following are multiple choice questions (with answers) about {}. Think step by" \
                        " step and then output the answer in the format of \"The answer is (X)\" at the end.\n\n" \
                    .format(subject)
                    question = format_example(each["question"], each["options"])
                    answer = each["cot_content"]
                    if answer.startswith("A: "):
                        answer = answer[3:]
                    message = start_prompt + user_prompt + "Question: " + question + "Answer: " + "Let's think step by step." + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

                    sample = {"question": question, "answer": answer, "prompt": message, "subject": subject}
                    samples.append(sample)

    elif args.task == "gpqa": 
        system_prompt = "You are a very intelligent assistant, who follows instructions directly.\n\n"

        if fewshot != None: 
            for q in fewshot["questions"]:
                question = f"What is the correct answer to this question: {q['question']}\nChoices:\n"
                for choice, value in q["choices"].items():
                    question += f'({choice}) {value}\n'
                user_prompt = "\nGive step by step reasoning before you answer, and when you're ready to answer, please use the format \"The correct answer is (insert answer here)\":\n"
                
                answer = f"Let's think step by step: \n{q['explanation']}\n"
                answer += f'The correct answer is ({q["correct_answer"]})\n'
                message = start_prompt + system_prompt + question + user_prompt + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                sample = {"question": question, "answer": answer, "prompt": message, "subject": "gpqa"}
                samples.append(sample)
    
    elif args.task == "hotpotqa": 
        system_prompt = ""

        if fewshot != None: 
            for qa in fewshot:
                question = qa["Q"]
                answer = qa["A"]
                user_prompt = f"Q: {question}."
                user_prompt += "\n\nEnd your answer with \"Answer <answer>\". Think step by step." + "\n\nA: " 
                
                message = start_prompt + system_prompt + user_prompt + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

                sample = {"question": question, "answer": answer, "prompt": message, "subject": "hotpotqa"}
                samples.append(sample)

    elif args.task == "drop": 
        system_prompt = ""

        if fewshot != None: 
            for qa in fewshot:
                question = qa["Q"]
                answer = qa["A"]

                user_prompt = f"Q: {question}" + "\n\nEnd your answer with \"So the answer is <answer>\". Think step by step." + "\n\nA: " 
                message = start_prompt + system_prompt + user_prompt + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

                sample = {"question": question, "answer": answer, "prompt": message, "subject": "drop"}
                samples.append(sample)
    
    elif args.task == "musr_location":
        system_prompt = dataset[0]['prompt_parts']['cot_system_prompt']

        few_instruction = few_shot_op_instruction 
        for (q, a) in fewshot:
            user_prompt = f"{q}\n\n{few_instruction}"
            
            message = start_prompt + system_prompt + user_prompt + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

            sample = {"question": q, "answer": a, "prompt": message, "subject": "musr_location"}
            samples.append(sample)

    elif args.task == "musr_efficiently": 
        system_prompt = dataset[0]['prompt_parts']['cot_system_prompt']

        few_instruction = few_shot_ta_instruction 
        for (q, a) in fewshot:
            user_prompt = f"{q}\n\n{few_instruction}"

            message = start_prompt + system_prompt + user_prompt + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

            sample = {"question": q, "answer": a, "prompt": message, "subject": "musr_efficiently"}
            samples.append(sample)

    else: 
        return None

    return samples

def main(config):

    save_path = f"{config.output_dir}/{config.task}/{config.model}/{config.task}_mistakes.jsonl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)


    dataset, fewshot = load_data_and_fewshot(config)
    if config.shot_type == "zero":
        fewshot = None
        
    samples = construct_prompt(config, dataset, fewshot)
    if samples:
        print(f"Model: {config.model} Task: {config.task}, Shot: {config.shot_type}")
        print("-" * 50)
        prompt = samples[0]["prompt"]
        print(f"Question: {samples[0]['question']}")
        print(f"Answer: {samples[0]['answer']}")
        print(f"Message: \n{prompt}")
        print("-" * 50)
    else:
        print(f"No samples found for Task: {config.task}, Shot: {config.shot_type}")

    if os.path.exists(save_path):
        with open(save_path, 'r', encoding='utf-8') as f:

            existing_data = {json.loads(line)['question'] for line in f}  
    else:
        existing_data = set()  
    
    if samples:

        with open(save_path, "a", encoding='utf-8') as f:  
            for sample in tqdm(samples, total=len(samples)):
                if sample['question'] in existing_data:  
                    continue
                try:

                    model_outputs = []
                    for i in range(15): 
                        model_output = generate_model_output(sample["prompt"])
                        model_outputs.append(model_output)
                    sample["model_outputs"] = model_outputs


                    json.dump(sample, f)
                    f.write("\n")
                except Exception as e:
                    print(f"Error processing sample {sample['question']}: {e}")
                    break

        print(f"Results saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama", help="model")
    parser.add_argument("--task", type=str, default="math500", help="task")
    parser.add_argument("--shot_type", type=str, default="few", help="shot_type")
    parser.add_argument("--output_dir", type=str, default="leap", help="output_dir")
    parser.add_argument("--num_examples", type=int, default=-1, help="output_dir")


    args = parser.parse_args()
    print("=== [Parsed Arguments] ===")
    for key, value in vars(args).items():
        print(f"{key} = {value}")
    print("\n\n")

    main(args)
