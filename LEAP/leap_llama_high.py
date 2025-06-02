import os
import json
import sys

import re
from typing import List

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
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token=hf_token)
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
            do_sample=False,                    # Greedy
            temperature=0,                    
            pad_token_id=tokenizer.pad_token_id 
        )

    gen_ids = output_ids[0][len(input_ids[0]):]
    output_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    return output_text.strip()

def extract_from_insight(text: str) -> str:
    match = re.search(r'Insight.*', text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(0).strip()
    else:
        return text.strip()

def construct_high_level_prompt(low_principles: List[List[str]]) -> str:


    joined_principles = "\n".join(
        f"{i+1}. {extract_from_insight(lp[0])}"
        for i, lp in enumerate(low_principles)
    )
    
    prompt = (
        "<|start_header_id|>user<|end_header_id|>"
        f"Low-level principles:\n{joined_principles}\n\n"
        "Create a list of *unique* and insightful principles to improve future responses based on the analysis above.\n"
        "Focus on capturing the essence of the feedback while eliminating redundancies.\n"
        "Ensure that each point is clear, concise, and directly derived from the introspection results.\n"
        "Create a numbered list of principles. Leave specific details in place.\n"
        "Limit to at most 8 principles.\n\n"
        "List of Principles:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return prompt


def get_high_level_principle(model, path_to_low_level):
    with open(path_to_low_level, 'r', encoding='utf-8') as f:
        data = json.load(f)

    low_principles = data
    
    if not low_principles:
        print("[WARN] No low-level principles found in file.")
        return None
    
    prompt = construct_high_level_prompt(low_principles)

    print("=== Prompt Preview ===")
    print(prompt)
    print("="*30)
    
    high_principle = generate_model_output(prompt)
    
    print("=== Model Output ===")
    print(high_principle)
    print("="*30)
    
    return [high_principle]




def main(config):
    model = config.model 
    if config.task == 'mmlu_pro': 
        subjects = ['business', 'law', 'psychology', 'biology', 'chemistry', 'history', 'other', 'health', 'economics', 'math', 'physics', 'computer science', 'philosophy', 'engineering']
        for subject in subjects:
            path = f"leap/{config.task}/{model}/{subject}/{config.task}_wrong_predictions.jsonl"
            low_level_path = f"leap/{config.task}/{model}/{subject}/{config.task}_low_level_principle.json"
            high_level_path = f"leap/{config.task}/{model}/{subject}/{config.task}_high_level_principle.json"

            if not os.path.exists(low_level_path):
                print(f"[WARN] File not found: {low_level_path}")
                return None

            os.makedirs(os.path.dirname(high_level_path), exist_ok=True)
            
            high_principles = get_high_level_principle(model, low_level_path)
            
            with open(high_level_path, 'w', encoding='utf-8') as wf:
                json.dump(high_principles, wf, ensure_ascii=False, indent=2)
            
            print(f"[*] Saved {len(high_principles)} higt-level principles to {high_level_path}")
    
    else:
        path = f"leap/{config.task}/{model}/{config.task}_wrong_predictions.jsonl"
        low_level_path = f"leap/{config.task}/{model}/{config.task}_low_level_principle.json"
        high_level_path = f"leap/{config.task}/{model}/{config.task}_high_level_principle.json"

        if not os.path.exists(low_level_path):
            print(f"[WARN] File not found: {low_level_path}")
            return None

        os.makedirs(os.path.dirname(high_level_path), exist_ok=True)

        high_principles = get_high_level_principle(model, low_level_path)

        with open(high_level_path, 'w', encoding='utf-8') as wf:
            json.dump(high_principles, wf, ensure_ascii=False, indent=2)

        print(f"[*] Saved {len(high_principles)} low-level principles to {high_level_path}")


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
