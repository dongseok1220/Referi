import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
import argparse
from tqdm import tqdm

from musr import MuSRDataset

from op_icl_fixed import op_fewshot, few_shot_op_instruction, test_op_instruction
from ta_icl_fixed import ta_fewshot, few_shot_ta_instruction, test_ta_instruction

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import calculate_cross_entropy_loss_with_topk, load_model_outputs, load_existing_likelihoods, load_llm

load_dotenv()

hf_token = os.getenv('HUGGINGFACE_TOKEN')

def run(args):
    tokenizer, model, device = load_llm(args.model_name, hf_token) 

    input_file_path = f"{args.input_dir}/{args.task}/{args.model}/{args.task}_few.jsonl"

    if args.task == "musr_location":
        few_shot = op_fewshot
        few_instruction =  few_shot_op_instruction
        test_instruction = test_op_instruction
    else: 
        few_shot = ta_fewshot
        few_instruction = few_shot_ta_instruction
        test_instruction = test_ta_instruction

    model_outputs = load_model_outputs(input_file_path)
    if model_outputs is None:
        return
    
    n = len(model_outputs[0]['model_outputs'])

    output_dir = f"{args.output_dir}/{args.task}/{args.model}/few"
    os.makedirs(output_dir, exist_ok=True)

    likelihoods_file = os.path.join(output_dir, "all_likelihoods.json")
    existing_likelihoods = load_existing_likelihoods(likelihoods_file)
    
    likelihoods = existing_likelihoods if len(existing_likelihoods) > 0 else [[] for _ in range(n)]  
    
    for entry in tqdm(model_outputs, desc="Processing entries"):
        doc_id = entry['idx']
        id_exists = any(item['id'] == doc_id for item in likelihoods[0])
    
        if id_exists:
            continue

        question = entry["entry"]['question'].strip()
        context = entry['entry']['context'].strip()
        choices = entry['entry']['choices']['text']
        labels = ['A','B','C','D','E','F'][0:len(choices)]

        choice_str = '\n'.join(
                    [f'{labels[idx]}: {choices[idx]}' for idx in range(len(choices))])
        
        question_block = f"{context}\n\n{question}\n\n{choice_str}"

        answer = entry['entry']['answer']

        code_outputs = entry['model_outputs']

        system_prompt = entry['entry']['prompt_parts']['cot_system_prompt']
        
        
        for output_idx, model_output in enumerate(code_outputs):
            current_likelihoods = {
                'id': doc_id,
                'model_output': model_output,
                'answer': answer,
                'no_replace_likelihoods': [],
                'replace_likelihoods': [],
                'zero_likelihoods': [],
                'remaining_likelihoods': [],
            }
            if len(model_output) > 5000:
                model_output = model_output[:5000]
                print("Output is too long!!!!")

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




                remaining_few_shot = [
                    (q, a)
                    for i, (q, a) in enumerate(few_shot)
                    if i != replace_idx
                ]

                remaining_blocks = []
                for (q, a) in remaining_few_shot:
                    block = f"{q}\n\n{few_instruction}\n\n{a}"
                    remaining_blocks.append(block)

                remaining_prompt = f"{system_prompt}\n\n" + "\n\n".join(remaining_blocks)

                remaining_prompt += f"\n\n{original_question_part}\n\n{test_instruction}"




                no_replace_prompt = f"{system_prompt}\n\n{question_block}\n\n{few_instruction}\n\n{model_output}"
                no_replace_prompt += f"\n\n{original_question_part}\n\n{test_instruction}"


                zero_prompt = f"{system_prompt}\n\n{original_question_part}\n\n{test_instruction}"

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
    parser.add_argument("--model", type=str, default=None, help="gpt")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="llama,qwen")
    parser.add_argument("--task", type=str, default=None, help="Filter by problem type")
    parser.add_argument("--output_dir", type=str, default="baseline", help="output_dir")
    parser.add_argument("--input_dir", type=str, default="result_gpt", help="input_dir")

    args = parser.parse_args()
    run(args)
