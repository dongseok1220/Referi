import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
import argparse
from tqdm import tqdm
from mmlu_utils import load_mmlu_pro

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import calculate_cross_entropy_loss_with_topk, load_model_outputs, load_existing_likelihoods, load_llm


load_dotenv()

hf_token = os.getenv('HUGGINGFACE_TOKEN')

def format_example(question, options, cot_content=""):
    if cot_content == "":
        cot_content = "Let's think step by step."
    if cot_content.startswith("A: "):
        cot_content = cot_content[3:]
    example = "Question: {}\nOptions: ".format(question)
    choice_map = "ABCDEFGHIJ"
    for i, opt in enumerate(options):
        example += "{}. {}\n".format(choice_map[i], opt)
    if cot_content == "":
        example += "Answer: "
    else:
        example += "Answer: " + cot_content + "\n\n"
    return example


def run(args):
    tokenizer, model, device = load_llm(args.model_name, hf_token) 

    print("assigned subjects", args.subject)

    input_file_path = f"{args.input_dir}/{args.task}/{args.model}/{args.subject}_result.jsonl"

    model_outputs = load_model_outputs(input_file_path)
    if model_outputs is None:
        return
    n = len(model_outputs[0]['model_outputs'])

    _, dev_df = load_mmlu_pro()
    
    output_dir = f"{args.output_dir}/{args.task}/{args.model}/{args.subject}/few"
    os.makedirs(output_dir, exist_ok=True)

    likelihoods_file = os.path.join(output_dir, "all_likelihoods.json")
    existing_likelihoods = load_existing_likelihoods(likelihoods_file)
    
    likelihoods = existing_likelihoods if len(existing_likelihoods) > 0 else [[] for _ in range(n)]  
    
    for r in tqdm(model_outputs, total=len(model_outputs)):
        id = r['idx']
        id_exists = any(item['id'] == id for item in likelihoods[0])
    
        if id_exists:
            continue
        
        question = r['entry']['question']
        options = r['entry']['options']
        answer = r['entry']['answer']
        model_outputs = r['model_outputs']

        for output_idx, model_output in enumerate(model_outputs):
            current_likelihoods = {
                'id': id,
                'model_output': model_output,
                'answer': answer, 
                'no_replace_likelihoods': [],
                'replace_likelihoods': [],
                'zero_likelihoods': []
            }
            model_example = format_example(question, options, model_output)


            for replace_idx in range(5): # MMLU-PRO is 5-shot
                base_prompt = "The following are multiple choice questions (with answers) about {}. Think step by" \
                    " step and then output the answer in the format of \"The answer is (X)\" at the end.\n\n" \
                .format(args.subject)

                few_shot = dev_df[args.subject][replace_idx]
                question_part = format_example(few_shot["question"], few_shot["options"])
                answer_part = few_shot["cot_content"]
                if answer_part.startswith("A: "):
                    answer_part = answer_part[3:]

                no_replace_prompt = base_prompt + model_example + question_part
                zero_prompt = base_prompt + question_part

                replace_prompt = base_prompt
                for i, each in enumerate(dev_df[args.subject]):
                    if i != replace_idx: 
                        replace_prompt += format_example(each["question"], each["options"], each["cot_content"])
                    else: 
                        replace_prompt += model_example
                replace_prompt = replace_prompt + question_part

            
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