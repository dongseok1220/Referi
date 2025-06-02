import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
import argparse
from tqdm import tqdm
from typing import List, Dict

from math_utils import list_fewshot_samples

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import calculate_cross_entropy_loss_with_topk, load_model_outputs, load_existing_likelihoods, load_llm

load_dotenv()

hf_token = os.getenv('HUGGINGFACE_TOKEN')


def construct_prompt(few_shot_examples: List[Dict], query: str) -> str:
    start_prompt = "<|start_header_id|>user<|end_header_id|>\n\nSolve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...Regardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem.\n\n"
    prompt = ""
    if few_shot_examples != None:
        for example in few_shot_examples:
            prompt += start_prompt + "Problem: " + example["problem"] + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>" + "\n\n" 
            prompt += example["solution"] + "<|eot_id|>"
        

    prompt += start_prompt +  "Problem: " + query + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    
    return prompt
    
def run(args):
    tokenizer, model, device = load_llm(args.model_name, hf_token) 

    input_file_path = f"{args.input_dir}/{args.task}/{args.model}/{args.task}_few.jsonl"
    
    model_outputs = load_model_outputs(input_file_path)
    if model_outputs is None:
        return

    n = len(model_outputs[0]['resps'][0])  

    output_dir = f"{args.output_dir}/{args.task}/{args.model}/few"    
    os.makedirs(output_dir, exist_ok=True)

    likelihoods_file = os.path.join(output_dir, "all_likelihoods.json")
    existing_likelihoods = load_existing_likelihoods(likelihoods_file)
    
    likelihoods = existing_likelihoods if len(existing_likelihoods) > 0 else [[] for _ in range(n)]  
        
    for entry in tqdm(model_outputs, desc="Processing entries"):
        doc_id = entry['doc_id']
        question = entry['doc']['problem']
        answer = entry['doc']['answer']  # Ground truth answer

        code_outputs = entry['resps'][0]
        id_exists = any(item['id'] == doc_id for item in likelihoods[0])
    
        if id_exists:
            continue

        for output_idx, model_output in enumerate(code_outputs):
            current_likelihoods = {
                'id': doc_id,
                'model_output': model_output,
                'answer': answer,
                'no_replace_likelihoods': [],
                'replace_likelihoods': [],
                'zero_likelihoods': [],
            }

            for replace_idx in range(4): 
                modified_prompt = list_fewshot_samples()

                question_part, answer_part = modified_prompt[replace_idx]['problem'], modified_prompt[replace_idx]['solution']
                modified_prompt[replace_idx]['problem'] = question
                modified_prompt[replace_idx]['solution'] = model_output
 
                replace_prompt = construct_prompt(modified_prompt, question_part)
                no_replace_prompt = construct_prompt([modified_prompt[replace_idx]], question_part)
                zero_prompt = construct_prompt(None, question_part)

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
    parser.add_argument("--model", type=str, default="llama", help="llama")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="llama,qwen")
    parser.add_argument("--task", type=str, default="math500", help="Filter by problem type")
    parser.add_argument("--output_dir", type=str, default="likelihood", help="output_dir")
    parser.add_argument("--input_dir", type=str, default="result", help="input_dir")

    args = parser.parse_args()
    print("=== [Parsed Arguments] ===")
    for key, value in vars(args).items():
        print(f"{key} = {value}")
    print("\n\n")
 
    run(args)
