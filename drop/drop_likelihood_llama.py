import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
import argparse
from tqdm import tqdm
import copy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import calculate_cross_entropy_loss_with_topk, load_model_outputs, load_existing_likelihoods, load_llm


load_dotenv()

hf_token = os.getenv('HUGGINGFACE_TOKEN')

start_prompt = "<|start_header_id|>user<|end_header_id|>\n" 
end_prompt = "\n\nEnd your answer with \"Answer <answer>\". Think step by step.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" 
    
    
def create_final_prompt(question, fewshot):
    input_final_prompt = ""
    for qa in fewshot:
        input_final_prompt += f"{start_prompt}Q: {qa['Q']}{end_prompt}A: {qa['A']}<|eot_id|>\n"
    input_final_prompt += f"{start_prompt}Q: {question}{end_prompt}"
    return input_final_prompt

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
        
    with open("drop/prompt.json", 'r') as f:
        fewshot = json.load(f)

    for entry in tqdm(model_outputs, desc="Processing entries"):
        doc_id = entry['doc_id']
        id_exists = any(item['id'] == doc_id for item in likelihoods[0])
    
        if id_exists:
            continue

        question = entry['doc']['question']
        passage = entry['doc']['passage']
        answer = entry['doc']

        model_outputs = entry['resps'][0]

        for output_idx, model_output in enumerate(model_outputs):
            current_likelihoods = {
                'id': doc_id,
                'model_output': model_output,
                'answer': answer,
                'no_replace_likelihoods': [],
                'replace_likelihoods': [],
                'zero_likelihoods': [],
            }

            for replace_idx in range(3):  
                new_fewshot = copy.deepcopy(fewshot)

                question_part = fewshot[replace_idx]['Q']
                answer_part = fewshot[replace_idx]['A']

                new_fewshot[replace_idx]['Q'] = passage + " " + question
                new_fewshot[replace_idx]['A'] = model_output

                replace_prompt = create_final_prompt(question_part, new_fewshot)
                no_replace_prompt = f"{start_prompt}Q: {passage}{question}{end_prompt}A: {model_output}<|eot_id|>\n" + f"{start_prompt}Q: {question_part}{end_prompt}"
                zero_prompt =  start_prompt + f"Q: {question_part}" + end_prompt

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
    parser.add_argument("--task", type=str, default="drop", help="Filter by problem type")
    parser.add_argument("--output_dir", type=str, default="likelihood", help="output_dir")
    parser.add_argument("--input_dir", type=str, default="result", help="input_dir")

    args = parser.parse_args()
    print("=== [Parsed Arguments] ===")
    for key, value in vars(args).items():
        print(f"{key} = {value}")
    print("\n\n")
    
    run(args)

