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

def generate_model_output(prompt, max_new_tokens=30):
  
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

def load_data_and_fewshot(args):
    if args.task == "mmlu_pro":
        dataset, fewshot = load_mmlu_pro()
    elif args.task == "math500": 
        file_path = f"data/math500/test.jsonl"
        with open(file_path, 'r', encoding='utf-8') as f:
            dataset = [json.loads(line) for line in f]
    elif args.task == "gpqa":
        dataset = load_examples("data/gpqa/gpqa_diamond.csv", seed=0)
        
    elif args.task == "hotpotqa":
        dataset = json.load(open(f'data/hotpotqa/BM25/{args.task}-bm25.json'))
        
    elif args.task == "drop":

        dataset = pd.read_parquet("data/drop/drop_sub.parquet", engine="pyarrow")
        dataset = dataset.to_dict(orient="records")  

        dataset = convert_ndarray_to_list(dataset)
        dataset = convert_ndarray_to_list(dataset)

    elif args.task == "musr_efficiently":
        ta_path = 'data/musr/team_allocation.json'
        dataset = MuSRDataset(ta_path)
        fewshot = 1

    elif args.task == "musr_location":
        op_path = 'data/musr/object_placements.json'
        dataset = MuSRDataset(op_path)
        fewshot = 1
    else: 
        return None, None
    
    return dataset, None

def construct_prompt(args, dataset): 
    if args.task != 'mmlu_pro': 
        output_res_path = f"result/{args.task}/llama/{args.task}_{args.shot_type}.jsonl"
        res = load_model_outputs(output_res_path)

    start_prompt = "I have generated the following responses to the question: "
    end_prompt = """\n\nEvaluate these responses.\nSelect the most consistent response based on majority consensus.\nStart your answer with "The most consistent response is Response X" (without quotes)."""


    samples = []
    if args.task == "math500":
        for idx, entry in tqdm(enumerate(dataset)):             
            user_prompt = start_prompt + f"{entry['problem']}" + "\n\n"
            model_outputs = res[idx]['resps'][0]

            for i, output in enumerate(model_outputs):
                user_prompt += f"Response {i}: {output}\n"

            user_prompt += end_prompt
            user_prompt += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

            sample = {"idx": idx,"prompt": user_prompt,"entry": entry}
            samples.append(sample)

       
    elif args.task == "mmlu_pro":
        def format_example(question, options, cot_content=""):
            example = "{}\nOptions: ".format(question)
            choice_map = "ABCDEFGHIJ"
            for i, opt in enumerate(options):
                example += "{}. {}\n".format(choice_map[i], opt)
            
            return example
        
        subjects = list(dataset.keys())
        for subject in tqdm(subjects): 
            res_path = f"result/{args.task}/llama/{subject}_result.jsonl"
            res = load_model_outputs(res_path)

            random.seed(42)
            test_data = random.sample(dataset[subject], min(300, len(dataset[subject])))
            
            for idx, entry in enumerate(test_data):
                user_prompt = start_prompt + format_example(entry['question'], entry['options']) + '\n\n'
                model_outputs = res[idx]['resps'][0]
                for i, output in enumerate(model_outputs):
                    user_prompt += f"Response {i}: {output}\n"

                user_prompt += end_prompt
                user_prompt += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                sample = {"idx": entry['question_id'],"prompt": user_prompt,"entry": entry}
                samples.append(sample)
    
    elif args.task == "gpqa": 
        samples = []
        for idx, example in tqdm(enumerate(dataset)):
            user_prompt = start_prompt + f"{example.question}"
            user_prompt += f"\nChoices:\n(A) {example.choice1}\n(B) {example.choice2}\n(C) {example.choice3}\n(D) {example.choice4}\n\n"
            model_outputs = res[idx]['resps'][0]
            for i, output in enumerate(model_outputs):
                user_prompt += f"Response {i}: {output}\n"

            user_prompt += end_prompt

            user_prompt += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            sample = {"idx": idx,"prompt": user_prompt,"entry": example}
            samples.append(sample)
    

    elif args.task == "hotpotqa": 
        samples = []
       
        for idx, entry in tqdm(enumerate(dataset)):
           
            user_prompt = start_prompt + f"{entry['question']}\n\n" 
            model_outputs = res[idx]['resps'][0]
            for i, output in enumerate(model_outputs):
                user_prompt += f"Response {i}: {output}\n"
            user_prompt += end_prompt
            user_prompt += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            

            sample = {"idx": idx,"prompt": user_prompt,"entry": entry}
            samples.append(sample)

    elif args.task == "drop": 
        samples = []
        
        for idx, entry in tqdm(enumerate(dataset)):
            user_prompt = start_prompt + f"{entry['passage']} {entry['question']}\n\n" 
            model_outputs = res[idx]['resps'][0]

            for i, output in enumerate(model_outputs):
                user_prompt += f"Response {i}: {output}\n"

            user_prompt += end_prompt
            user_prompt += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            
            sample = {"idx": idx,"prompt": user_prompt,"entry": entry}
            samples.append(sample)

    elif args.task == "musr_location": 
        for idx, entry in tqdm(enumerate(dataset)):
            question = entry['question'].strip()
            context = entry['context'].strip()
            choices = entry['choices']['text']
            

            labels = ['A', 'B', 'C', 'D', 'E', 'F'][:len(choices)]
            choice_str = '\n'.join([f'{labels[idx]}: {choices[idx]}' for idx in range(len(choices))])
            original_question_part = f"{context}\n\n{question}\n\n{choice_str}"

            model_outputs = res[idx]['resps'][0]

            user_prompt = start_prompt + f"{original_question_part}\n\n" 
            for i, output in enumerate(model_outputs):
                user_prompt += f"Response {i}: {output}\n"

            user_prompt += end_prompt
            user_prompt += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            
        
            sample = {"idx": idx, "prompt": user_prompt, "entry": entry}
            samples.append(sample)
        
    elif args.task == "musr_efficiently":
        for idx, entry in tqdm(enumerate(dataset)):
            question = entry['question'].strip()
            context = entry['context'].strip()
            choices = entry['choices']['text']
            

            labels = ['A', 'B', 'C', 'D', 'E', 'F'][:len(choices)]
            choice_str = '\n'.join([f'{labels[idx]}: {choices[idx]}' for idx in range(len(choices))])
            original_question_part = f"{context}\n\n{question}\n\n{choice_str}"
                        
            user_prompt = start_prompt + f"{original_question_part}\n\n" 
            model_outputs = res[idx]['resps'][0]
            for i, output in enumerate(model_outputs):
                user_prompt += f"Response {i}: {output}\n"

            user_prompt += end_prompt
            user_prompt += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            
            sample = {"idx": idx, "prompt": user_prompt, "entry": entry}
            samples.append(sample)
    else: 
        return None

    return samples

def main(config):

    save_path = f"{config.output_dir}/{config.task}/llama/{config.task}_{config.shot_type}.jsonl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    dataset, few_shot = load_data_and_fewshot(config)

    samples = construct_prompt(config, dataset)
    print(len(samples))
    if config.num_examples != -1: 
        samples = samples[:config.num_examples] 
    if samples:
        print(f"Model: llama Task: {config.task}, Shot: {config.shot_type}")
        print(samples[0].keys())
        print("-" * 50)
        prompt = samples[-1]["prompt"]
        print(prompt)
        print("-" * 50)
    else:
        print(f"No samples found for Task: {config.task}, Shot: {config.shot_type}")

    
    if os.path.exists(save_path):
        with open(save_path, 'r', encoding='utf-8') as f:

            existing_data = {json.loads(line)['idx'] for line in f}  
    else:
        existing_data = set()  

    if samples:

        with open(save_path, "a", encoding='utf-8') as f:  
            for sample in tqdm(samples, total=len(samples)):
                if sample['idx'] in existing_data:  
                    continue
                try:

                    model_outputs = generate_model_output(sample["prompt"])
                    sample["prompt_output"] = model_outputs
                    json.dump(sample, f)
                    f.write("\n")
                except Exception as e:
                    print(f"Error processing sample {sample['idx']}: {e}")
                    break

        print(f"Results saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="model")
    parser.add_argument("--task", type=str, default=None, help="task")
    parser.add_argument("--shot_type", type=str, default="few", help="shot_type")
    parser.add_argument("--output_dir", type=str, default="llm_prompt", help="output_dir")
    parser.add_argument("--num_examples", type=int, default=-1, help="output_dir")


    args = parser.parse_args()
    main(args)
