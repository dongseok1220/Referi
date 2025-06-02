import os
import json
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
import argparse
from tqdm import tqdm
import copy


from collections import namedtuple
from typing import Dict, List, Tuple, Union
import random

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import calculate_cross_entropy_loss_with_topk, load_model_outputs, load_existing_likelihoods, load_llm


Example = namedtuple('Example', ['question', 'choice1', 'choice2', 'choice3', 'choice4', 'correct_index', 'explanation'])


load_dotenv()

hf_token = os.getenv('HUGGINGFACE_TOKEN')

def generate_prompt_from_examples(json_data, replace_idx, with_explanations=True):
    output = ""
    for idx, q in enumerate(json_data):
        output += "<|start_header_id|>user<|end_header_id|>\n" 
        output += f"Here are some example questions from experts. An explanation is given before the final answer. Answer the final question yourself, giving your reasoning beforehand.\n"
        output += f'Question: {q["question"]}\nChoices:\n'
        for choice, value in q["choices"].items():
            output += f'({choice}) {value}\n'
        output += "\nGive step by step reasoning before you answer, and when you're ready to answer, please use the format \"The correct answer is (insert answer here)\":\n"

        if with_explanations:
            output += f"Let's think step by step:<|eot_id|><|start_header_id|>assistant<|end_header_id|> \n{q['explanation']}"
        if idx != replace_idx:
            output += f'\nThe correct answer is ({q["correct_answer"]})<|eot_id|>\n'
        else:
            output += "<|eot_id|>\n"
    return output

def chain_of_thought_prompt(json_data, replace_idx, example: Example) -> str:
    """Creates a chain-of-thought prompt given a single example."""
    prompt = generate_prompt_from_examples(json_data, replace_idx, with_explanations=True)
    prompt += "<|start_header_id|>user<|end_header_id|>\n" 
    prompt += f"Here are some example questions from experts. An explanation is given before the final answer. Answer the final question yourself, giving your reasoning beforehand.\n"
    prompt += f"Question: {example['question']}"
    prompt += f"\nChoices:\n(A) {example['choices']['A']}\n(B) {example['choices']['B']}\n(C) {example['choices']['C']}\n(D) {example['choices']['D']}"
    prompt += "\nGive step by step reasoning before you answer, and when you're ready to answer, please use the format \"The correct answer is (insert answer here)\":\n"
    prompt += "Let's think step by step:"
    return prompt

def zero_shot_chain_of_thought_prompt(example: Example) -> str: 
    prompt = f"What is the correct answer to this question: {example['question']}"
    prompt += f"\nChoices:\n(A) {example['choices']['A']}\n(B) {example['choices']['B']}\n(C) {example['choices']['C']}\n(D) {example['choices']['D']}"
    prompt += "\nGive step by step reasoning before you answer, and when you're ready to answer, please use the format \"The correct answer is (insert answer here)\":\n"
    prompt += "Let's think step by step:"

    return prompt

def load_examples(path: str, seed: int) -> List[Example]:
    """Load questions from csv file and return a list of Example namedtuples."""
    question_df = pd.read_csv(path)
    random.seed(seed)

    def shuffle_choices_and_create_example(row) -> Example:
        list_choices = [row['Incorrect Answer 1'], row['Incorrect Answer 2'], row['Incorrect Answer 3'], row['Correct Answer']]
        random.shuffle(list_choices)
        example = Example(row.Question, list_choices[0], list_choices[1], list_choices[2], list_choices[3],
                          list_choices.index(row['Correct Answer']), row['Explanation'])
        return example

    return [shuffle_choices_and_create_example(row) for _, row in question_df.iterrows()]

    
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
        
        
    system_prompt = "<|start_header_id|>system<|end_header_id|>\n" 
    system_prompt += "You are a very intelligent assistant, who follows instructions directly.<|eot_id|>\n\n"
    start_prompt = "<|start_header_id|>user<|end_header_id|>\n" 
    end_prompt = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    examples = load_examples("data/gpqa/gpqa_diamond.csv", seed=0)

    with open("gpqa/chain_of_thought_examples.json", 'r') as f:
        json_data = json.load(f)
        json_data = json_data['questions']
        
    for entry, example in tqdm(zip(model_outputs, examples), desc="Processing entries"):
        doc_id = entry['doc_id']

        id_exists = any(item['id'] == doc_id for item in likelihoods[0])
    
        if id_exists:
            continue

        question = example.question
        choices = dict()
        choices["A"] = example.choice1
        choices["B"] = example.choice2
        choices["C"] = example.choice3
        choices["D"] = example.choice4
        answer = entry['target']

        code_outputs = entry['resps'][0]

        for output_idx, model_output in enumerate(code_outputs):
            current_likelihoods = {
                'id': doc_id,
                'model_output': model_output,
                'answer': answer,
                'no_replace_likelihoods': [],
                'replace_likelihoods': [],
                'zero_likelihoods': [],
            }

            for replace_idx in range(5):  
                new_json_data = copy.deepcopy(json_data)

                fewshot_example = json_data[replace_idx]
                new_json_data[replace_idx]['question'] = question
                new_json_data[replace_idx]['choices'] = choices
                new_json_data[replace_idx]['explanation'] = model_output

                answer_part = fewshot_example['explanation']
        
                replace_prompt = system_prompt + chain_of_thought_prompt(new_json_data, replace_idx, fewshot_example) + end_prompt
                no_replace_prompt = system_prompt + chain_of_thought_prompt([new_json_data[replace_idx]], 0, fewshot_example) + end_prompt
                zero_prompt = system_prompt + start_prompt + zero_shot_chain_of_thought_prompt(fewshot_example) + end_prompt

                no_replace_results  = calculate_cross_entropy_loss_with_topk(no_replace_prompt, answer_part, model, tokenizer, device)
                replace_results = calculate_cross_entropy_loss_with_topk(replace_prompt, answer_part, model, tokenizer, device)
                zero_results    = calculate_cross_entropy_loss_with_topk(zero_prompt, answer_part, model, tokenizer, device)

                current_likelihoods['no_replace_likelihoods'].append(no_replace_results)
                current_likelihoods['replace_likelihoods'].append(replace_results)
                current_likelihoods['zero_likelihoods'].append(zero_results)

            likelihoods[output_idx].append(current_likelihoods)

    likelihoods_output_file = os.path.join(output_dir, "all_likelihoods.json")
    with open(likelihoods_output_file, "w") as f:
        json.dump(likelihoods, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama", help="llama")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="llama,qwen")
    parser.add_argument("--task", type=str, default="gpqa", help="Filter by problem type")
    parser.add_argument("--output_dir", type=str, default="likelihood", help="output_dir")
    parser.add_argument("--input_dir", type=str, default="result", help="input_dir")

    args = parser.parse_args()
    print("=== [Parsed Arguments] ===")
    for key, value in vars(args).items():
        print(f"{key} = {value}")
    print("\n\n")

    run(args)