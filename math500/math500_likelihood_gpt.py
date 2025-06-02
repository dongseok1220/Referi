import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
import argparse
from tqdm import tqdm
from math_utils import load_prompt, load_prompt_seed

from parser import parse_ground_truth

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import calculate_cross_entropy_loss_with_topk, load_model_outputs, load_existing_likelihoods, load_llm

load_dotenv()

hf_token = os.getenv('HUGGINGFACE_TOKEN')


def run(args):
    tokenizer, model, device = load_llm(args.model_name, hf_token) 

    input_file_path = f"{args.input_dir}/{args.task}/{args.model}/{args.task}_few.jsonl"

    model_outputs = load_model_outputs(input_file_path)
    if model_outputs is None:
        return
    
    n = len(model_outputs[0]['model_outputs'])

    output_dir = f"{args.output_dir}/{args.task}/{args.model}/few"
    os.makedirs(output_dir, exist_ok=True)

    likelihoods_file = os.path.join(output_dir, f"all_likelihoods.json")
    existing_likelihoods = load_existing_likelihoods(likelihoods_file)
    
    likelihoods = existing_likelihoods if len(existing_likelihoods) > 0 else [[] for _ in range(n)]  
    
    system_prompt = "Please reason step by step, and put your final answer within \\boxed{{}}.\n\n"

    for entry in tqdm(model_outputs, desc="Processing entries"):
        doc_id = entry['idx']
        id_exists = any(item['id'] == doc_id for item in likelihoods[0])
    
        if id_exists:
            continue

        question = entry["entry"]['problem']
        _, answer = parse_ground_truth(entry['entry'], "math")

        code_outputs = entry['model_outputs']
        
        for output_idx, model_output in enumerate(code_outputs):
            current_likelihoods = {
                'id': doc_id,
                'model_output': model_output,
                'answer': answer,
                'no_replace_likelihoods': [],
                'replace_likelihoods': [],
                'zero_likelihoods': [],
            }
            if len(model_output) > 5000:
                model_output = model_output[:5000]
                print("Output is too long!!!!")

            for replace_idx in range(5):
                original_modified_prompt = load_prompt(num_shots=5)  

                modified_prompt = original_modified_prompt.copy()     

                question_part, answer_part = modified_prompt[replace_idx]
                modified_prompt[replace_idx] = (question, model_output)

                replace_prompt = system_prompt + "\n\n".join([f"{q}\n\n{a}" for q, a in modified_prompt]) + "\n\n" + question_part + "\n"
                no_replace_prompt  = system_prompt + question + "\n\n" + model_output + "\n\n" + question_part + "\n"
                zero_prompt    = system_prompt + question_part + "\n"

                no_replace_results  = calculate_cross_entropy_loss_with_topk(no_replace_prompt, answer_part, model, tokenizer, device)
                replace_results = calculate_cross_entropy_loss_with_topk(replace_prompt, answer_part, model, tokenizer, device)
                zero_results    = calculate_cross_entropy_loss_with_topk(zero_prompt, answer_part, model, tokenizer, device)

                current_likelihoods['no_replace_likelihoods'].append(no_replace_results)
                current_likelihoods['replace_likelihoods'].append(replace_results)
                current_likelihoods['zero_likelihoods'].append(zero_results)

            likelihoods[output_idx].append(current_likelihoods)

    with open(likelihoods_file, "w") as f:
        json.dump(likelihoods, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="gpt")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="llama,qwen")
    parser.add_argument("--task", type=str, default="math500", help="Filter by problem type")
    parser.add_argument("--output_dir", type=str, default="likelihood", help="output_dir")
    parser.add_argument("--input_dir", type=str, default="result", help="input_dir")
    parser.add_argument("--seed", type=int, default=42, help="seed")

    args = parser.parse_args()
    print("=== [Parsed Arguments] ===")
    for key, value in vars(args).items():
        print(f"{key} = {value}")
    print("\n\n")
    
    run(args)
