import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
import argparse
from tqdm import tqdm

import random
import re
from mmlu_utils import load_mmlu_pro

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import calculate_cross_entropy_loss_with_topk, load_model_outputs, load_existing_likelihoods, load_llm


load_dotenv()


hf_token = os.getenv('HUGGINGFACE_TOKEN')

def remove_paren_after_answer(cot_content: str) -> str:
    pattern = re.compile(r'(The answer is )\(([^)]+)\)(\.)?')
    return pattern.sub(r'The best answer is \2\3', cot_content)


def format_example_with_content(start_prompt, question, options, cot_content="", end_prompt=""):
    text = "A: Let's think step by step. "
    if cot_content.startswith("A: "):
        cot_content = cot_content[len(text):]
    example = start_prompt+ "Question: {}\n".format(question)
    choice_map = "ABCDEFGHIJ"
    
    for i in range(len(choice_map)):
        if i < len(options):

            example += f"{choice_map[i]}. {options[i]}\n"
        else:

            example += f"{choice_map[i]}. N/A\n"
    example += end_prompt
    
    example += "\n\n" + cot_content + "<|eot_id|>"
    return example

def format_example(question, options, end_prompt):
    example = "Question: {}\n".format(question)
    choice_map = "ABCDEFGHIJ"
    for i, opt in enumerate(options):
        if options[opt] == None: 
            options[opt] = 'N/A'
        example += "{}. {}\n".format(choice_map[i], options[opt])
    example += end_prompt
    return example


def run(args):
    tokenizer, model, device = load_llm(args.model_name, hf_token) 

    print("assigned subjects", args.subject)

    input_file_path = f"{args.input_dir}/{args.task}/{args.model}/{args.subject}_result.jsonl"

    res = load_model_outputs(input_file_path)
    if res is None:
        return
    n = len(res[0]['resps'][0])

    _, dev_df = load_mmlu_pro()
    
    output_dir = f"{args.output_dir}/{args.task}/{args.model}/{args.subject}/few"
    os.makedirs(output_dir, exist_ok=True)

    likelihoods_file = os.path.join(output_dir, "all_likelihoods.json")
    existing_likelihoods = load_existing_likelihoods(likelihoods_file)
    
    likelihoods = existing_likelihoods if len(existing_likelihoods) > 0 else [[] for _ in range(n)]  
    
    start_prompt = "<|start_header_id|>user<|end_header_id|>\n\nGiven the following question and candidate answers, choose the best answer.\n"
    end_prompt = "\nYour response should end with \"The best answer is [the_answer_letter].\" where the [the_answer_letter] is a letter from the provided choices.\n\nLet's think step by step.<|eot_id|><|start_header_id|>assistant<|end_header_id>"

    cut_idx =  len("A: Let's think step by step. ")
    for r in tqdm(res, total=len(res)):
        id = r['doc']['question_id']
        id_exists = any(item['id'] == id for item in likelihoods[0])
    
        if id_exists:
            continue

        question = r['doc']['problem']
        options = r['doc']['input_choice_list']
        answer = r['doc']['gold']
        model_outputs = r['resps'][0]

        for output_idx, model_output in enumerate(model_outputs):
            current_likelihoods = {
                'id': id,
                'model_output': model_output,
                'answer': answer, 
                'no_replace_likelihoods': [],
                'replace_likelihoods': [],
                'zero_likelihoods': []
            }
            if len(model_output) > 5000:
                model_output = model_output[:5000]
                print("Output is too long!!!!")

            model_example = format_example_with_content(start_prompt, question, list(options), model_output, end_prompt)


            for replace_idx in range(5): # MMLU-PRO is 5-shot
                few_shot = dev_df[args.subject][replace_idx]
                question_part = start_prompt + "Question: {}\n".format(few_shot['question'])
                choice_map = "ABCDEFGHIJ"
                for i in range(len(choice_map)):
                    if i < len(few_shot['options']):

                        question_part += f"{choice_map[i]}. {few_shot['options'][i]}\n"
                    else:

                        question_part += f"{choice_map[i]}. N/A\n"
                question_part += end_prompt + "\n\n"

                answer_part = remove_paren_after_answer(few_shot["cot_content"])
                if answer_part.startswith("A: "):
                    answer_part = answer_part[cut_idx:]

                no_replace_prompt = model_example + question_part
                zero_prompt = question_part
                prompt = ""
                for i, each in enumerate(dev_df[args.subject]):
                    if i != replace_idx: 
                        prompt += format_example_with_content(start_prompt, each["question"], each["options"], remove_paren_after_answer(each["cot_content"]), end_prompt)
                    else: 
                        prompt += model_example
                replace_prompt = prompt + question_part
                
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
    parser.add_argument("--model", type=str, default="llama", help="gpt")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="llama,qwen")
    parser.add_argument("--task", type=str, default="mmlu_pro", help="Filter by problem type")
    parser.add_argument("--subject", type=str, default="law", help="subject")
    parser.add_argument("--output_dir", type=str, default="likelihood", help="output_dir")
    parser.add_argument("--input_dir", type=str, default="result", help="input_dir")

    args = parser.parse_args()
    print("=== [Parsed Arguments] ===")
    for key, value in vars(args).items():
        print(f"{key} = {value}")
    print("\n\n")

    run(args)