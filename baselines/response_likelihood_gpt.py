import os
import sys
import json
from tqdm import tqdm 
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../musr')))


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gpqa.gpqa_utils import * 

from math500.math_utils import * 
from math500.parser import *
from math500.grader import * 

from mmlu_pro.mmlu_utils import * 

from hotpotqa.hotpotqa_utils import *

from drop.drop_utils import *

from musr import MuSRDataset

from utils import load_model_outputs, calculate_cross_entropy_loss_with_topk, load_llm


load_dotenv()

hf_token = os.getenv('HUGGINGFACE_TOKEN')


def load_data_and_fewshot(args):
    if args.task == "mmlu_pro":
        dataset, fewshot = load_mmlu_pro()

    elif args.task == "math500": 
        file_path = f"data/{args.task}/test.jsonl"
        with open(file_path, 'r', encoding='utf-8') as f:
            dataset = [json.loads(line) for line in f]
        fewshot = load_prompt(num_shots=5)

    elif args.task == "gpqa":
        dataset = load_examples("data/gpqa/gpqa_diamond.csv", seed=0)
        prompt_path = "gpqa/chain_of_thought_examples.json"
        with open(prompt_path, 'r') as f:
            fewshot = json.load(f)

    elif args.task == "hotpotqa":
        dataset = json.load(open(f'data/hotpotqa/{args.task}.json'))
        with open("hotpotqa/react_prompt.json", 'r') as f:
            fewshot = json.load(f)

    elif args.task == "drop":
        dataset = pd.read_parquet("data/drop/drop_sub.parquet", engine="pyarrow")
        dataset = dataset.to_dict(orient="records")  
        dataset = convert_ndarray_to_list(dataset)
        dataset = convert_ndarray_to_list(dataset)
        with open("drop/prompt.json", 'r') as f:
            fewshot = json.load(f)

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
    
    return dataset, fewshot

def construct_prompt(args, dataset, fewshot): 
    output_res_path = f"{args.input_dir}/{args.task}/{args.model}"

    samples = []

    if args.task == "math500":
        output_res_path = os.path.join(output_res_path, f"{args.task}_few.jsonl")
        res = load_model_outputs(output_res_path)

        system_prompt = "Please reason step by step, and put your final answer within \\boxed{{}}.\n\n"
        for idx, entry in tqdm(enumerate(dataset)): 
            user_prompt = ""
            if fewshot != None: 
                user_prompt = "\n\n".join([f"{q}\n\n{a}" for q, a in fewshot]) + "\n\n" 
            user_prompt += entry['problem'] + "\n"

            message = system_prompt + user_prompt
            sample = {"idx": idx,"prompt": message,"entry": entry,"model_outputs": res[idx]["model_outputs"]}
            samples.append(sample)
        
    elif args.task == "mmlu_pro":
        subjects = list(dataset.keys())
        for subject in tqdm(subjects): 
            res_path = os.path.join(output_res_path, f"{subject}_result.jsonl")
            res = load_model_outputs(res_path)
            
            user_prompt = "The following are multiple choice questions (with answers) about {}. Think step by" \
                " step and then output the answer in the format of \"The answer is (X)\" at the end.\n\n" \
            .format(subject)
            if fewshot != None:
                for each in fewshot[subject]:
                    user_prompt += format_example(each["question"], each["options"], each["cot_content"])
            
            random.seed(42)
            test_data = random.sample(dataset[subject], min(300, len(dataset[subject])))
            
            for entry, r in zip(test_data, res):
                input_text = format_example(entry['question'], entry['options'])

                message = user_prompt + input_text
                
                sample = {"idx": entry['question_id'],"prompt": message,"entry": entry,"model_outputs": r["model_outputs"]}
                samples.append(sample)
    
    elif args.task == "gpqa": 
        output_res_path = os.path.join(output_res_path, f"{args.task}_few.jsonl")
        res = load_model_outputs(output_res_path)

        system_prompt = "You are a very intelligent assistant, who follows instructions directly.\n\n"
        for example_id, example in tqdm(enumerate(dataset)):
            if fewshot != None: 
                user_prompt = chain_of_thought_prompt(fewshot, example)
            else:
                user_prompt = zero_shot_chain_of_thought_prompt(example)
            
            message = system_prompt + user_prompt

            sample = {"idx": example_id,"prompt": message,"entry": example,"model_outputs": res[example_id]['model_outputs']}
            samples.append(sample)

    elif args.task == "hotpotqa": 
        output_res_path = os.path.join(output_res_path, f"{args.task}_few.jsonl")
        res = load_model_outputs(output_res_path)

        system_prompt = ""

        if fewshot != None: 
            fewshot_prompt = ""
            for qa in fewshot:
                question = qa["Q"]
                answer = qa["A"]

                fewshot_prompt += f"Q: {question}\nA: {answer}\n\n"

        for idx, entry in tqdm(enumerate(dataset)):
            if fewshot != None: 
                user_prompt = fewshot_prompt + f"Q: {entry['question']}." + "\n\nEnd your answer with \"Answer <answer>\". Think step by step." + "\n\nA: " 
            else:
                user_prompt = f"Q: {entry['question']}." + "\n\nEnd your answer with \"Answer <answer>\". Think step by step." + "\n\nA: " 
            

            message = system_prompt + user_prompt

            sample = {"idx": idx,"prompt": message,"entry": entry,"model_outputs": res[idx]["model_outputs"]}
            samples.append(sample)

    elif args.task == "drop": 
        output_res_path = os.path.join(output_res_path, f"{args.task}_few.jsonl")
        res = load_model_outputs(output_res_path)

        system_prompt = ""

        if fewshot != None: 
            fewshot_prompt = ""
            for qa in fewshot:
                question = qa["Q"]
                answer = qa["A"]

                fewshot_prompt += f"Q: {question}\nA: {answer}\n\n"

        for idx, entry in tqdm(enumerate(dataset)):
            if fewshot != None: 
                user_prompt = fewshot_prompt + f"Q: {entry['passage']} {entry['question']}" + "\n\nEnd your answer with \"So the answer is <answer>\". Think step by step." + "\n\nA: " 
            else:
                user_prompt = f"Q: {entry['passage']} {entry['question']}" + "\n\nEnd your answer with \"So the answer is <answer>\". Think step by step." + "\n\nA: " 
            

            message = system_prompt + user_prompt

            sample = {"idx": idx,"prompt": message,"entry": entry,"model_outputs": res[idx]["model_outputs"]}
            samples.append(sample)

    elif args.task == "musr_efficiently" or args.task == "musr_location":
        output_res_path = os.path.join(output_res_path, f"{args.task}_few.jsonl")
        res = load_model_outputs(output_res_path)

        for idx, entry in tqdm(enumerate(dataset)):
            if fewshot != None:
                message = entry['fs_cot_messages'][0]['content'] + "\n\n" + entry['fs_cot_messages'][1]['content']
            else:
                message = entry['zs_cot_messages'][0]['content'] + "\n\n" + entry['zs_cot_messages'][1]['content']
            sample = {"idx": idx, "prompt": message, "entry": entry, "model_outputs": res[idx]["model_outputs"]}
            samples.append(sample)
    else: 
        return None
        
    return samples

def main(config):
    tokenizer, model, device = load_llm(config.model_name, hf_token) 

    save_path = f"{config.output_dir}/{config.task}/{config.model}/{config.task}_few_{config.shot_type}.jsonl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)


    dataset, fewshot = load_data_and_fewshot(config)
    if config.shot_type == "zero":
        fewshot = None
        

    samples = construct_prompt(config, dataset, fewshot)
    if config.num_examples != -1: 
        samples = samples[:config.num_examples] 
    if samples:
        print(f"Model: {config.model} Task: {config.task}, Shot: {config.shot_type}, Len: {len(samples)}")
        print("-" * 50)
        prompt = samples[0]["prompt"]
        print(prompt)
        print("-" * 50)
        print(samples[0]["model_outputs"][0])
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
                    model_outputs = sample["model_outputs"]
                    results = []
                    for model_output in model_outputs:
                        if len(model_output) > 5000: 
                            model_output = model_output[:5000]
                        result = calculate_cross_entropy_loss_with_topk(sample["prompt"], model_output, model, tokenizer, device)
                        results.append(result)

                    sample["results"] = results 
                    json.dump(sample, f)
                    f.write("\n")
                except Exception as e:
                    print(f"Error processing sample {sample['idx']}: {e}")
                    break

        print(f"Results saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o", help="model")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="llama,qwen")
    parser.add_argument("--task", type=str, default="math500", help="task")
    parser.add_argument("--shot_type", type=str, default="few", help="shot_type")
    parser.add_argument("--output_dir", type=str, default="baseline", help="output_dir")
    parser.add_argument("--input_dir", type=str, default="result", help="input_dir")
    parser.add_argument("--num_examples", type=str, default=-1, help="num_examples")
    parser.add_argument("--seed", type=int, default=42, help="seed")

    args = parser.parse_args()
    main(args)

    
