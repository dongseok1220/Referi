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
            do_sample=False,                    # Greedy
            temperature=0.0,                    
            pad_token_id=tokenizer.pad_token_id 
        )


    gen_ids = output_ids[0][len(input_ids[0]):]
    output_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    return output_text.strip()

def construct_low_level_principle(question, response, generated_answer, correct_reasoning, correct_answer):
    prompt = (
        "<|start_header_id|>user<|end_header_id|>"
        f"Question: {question}\n"
        f"Generated Reasoning: {response}\n\n"
        f"Generated Answer: {generated_answer}\n\n"
        f"Correct Reasoning: {correct_reasoning}\n\n"
        f"Correct Answer: {correct_answer}\n\n"
        "Instruction: Conduct a thorough analysis of the generated answer in comparison "
        "to the correct answer. Also observe how the generated reasoning differs from the "
        "correct reasoning. Identify any discrepancies, misunderstandings, or errors. "
        "Provide clear insights, principles, or guidelines that can be derived from this "
        "analysis to improve future responses. We are not focused on this one data point, "
        "but rather on the general principle.\n\n"
        "Reasoning: <discuss why the generated answer is wrong>\n"
        "Insights: <what principle should be looked at carefully to improve the performance "
        "in the future><|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return prompt

def get_low_level_principle(model, path): 
    with open(path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    low_principles = []
    for idx, entry in enumerate(data): 
        question = entry['entry']['question']
        response = entry['model_output']
        generated_answer = entry['pred']
        correct_reasoning = entry['entry']['answer']
        correct_answer = entry['gt'] 

        prompt = construct_low_level_principle(question, response, generated_answer, correct_reasoning, correct_answer)
        
        low_principle = generate_model_output(prompt)
        low_principles.append([low_principle])
        if idx == 0: 
            print(prompt)
            print("="*30)
            print(low_principle)
            print("="*30)

    return low_principles

def main(config):
    model = config.model 
    if config.task == 'mmlu_pro': 
        subjects = ['business', 'law', 'psychology', 'biology', 'chemistry', 'history', 'other', 'health', 'economics', 'math', 'physics', 'computer science', 'philosophy', 'engineering']
        for subject in subjects:
            path = f"leap/{config.task}/{config.model}/{subject}/{config.task}_wrong_predictions.jsonl"
            save_path = f"leap/{config.task}/{config.model}/{subject}/{config.task}_low_level_principle.json"

            if not os.path.exists(path):
                print(f"[WARN] File not found: {path}")
                return None

            
            with open(path, 'r', encoding='utf-8') as f:
                wrong = [json.loads(line) for line in f]

            if os.path.exists(save_path):
                print(save_path)
                with open(save_path, 'r', encoding='utf-8') as f:
                    low_level = json.load(f)
                if len(low_level) != 0: 
                    print(len(wrong), len(low_level))
                    return None


            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            

            low_principles = get_low_level_principle(model, path)
            
            with open(save_path, 'w', encoding='utf-8') as wf:
                json.dump(low_principles, wf, ensure_ascii=False, indent=2)
            
            print(f"[*] Saved {len(low_principles)} low-level principles to {save_path}")
    
    else:
        path = f"leap/{config.task}/{model}/{config.task}_wrong_predictions.jsonl"
        save_path = f"leap/{config.task}/{model}/{config.task}_low_level_principle.json"

        if not os.path.exists(path):
                print(f"[WARN] File not found: {path}")
                return None            
        with open(path, 'r', encoding='utf-8') as f:
            wrong = [json.loads(line) for line in f]

        if os.path.exists(save_path):
            print(save_path)
            with open(save_path, 'r', encoding='utf-8') as f:
                low_level = json.load(f)
            if len(low_level) != 0: 
                print(len(wrong), len(low_level))
                return None

        os.makedirs(os.path.dirname(save_path), exist_ok=True)


        low_principles = get_low_level_principle(model, path)

        with open(save_path, 'w', encoding='utf-8') as wf:
            json.dump(low_principles, wf, ensure_ascii=False, indent=2)

        print(f"[*] Saved {len(low_principles)} low-level principles to {save_path}")


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
