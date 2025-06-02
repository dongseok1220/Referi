import os
import sys
import json
from tqdm import tqdm 
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import load_model_outputs, calculate_cross_entropy_loss_with_topk, load_llm

load_dotenv()

hf_token = os.getenv('HUGGINGFACE_TOKEN')

def construct_prompt(args): 
    if args.shot_type == 'few': 
        prompt_path = f"result/mmlu_pro/{args.model}/{args.subject}_result.jsonl"
    else:
        prompt_path = f"result/mmlu_pro/{args.model}/{args.subject}_zero_result.jsonl"

    prompts=load_model_outputs(prompt_path)

    output_res_path = f"{args.input_dir}/{args.task}/llama/{args.subject}_result.jsonl"
    res = load_model_outputs(output_res_path)

    samples = []
    for r, prompt in zip(res, prompts): 
        idx = r['doc_id']
        message = prompt['doc']['input_final_prompts'][0]

        sample = {"idx": idx,"prompt": message,"entry": r['doc'],"model_outputs": r["resps"][0]}
        samples.append(sample)

    return samples

def main(config):
    tokenizer, model, device = load_llm(config.model_name, hf_token) 

    save_path = f"{config.output_dir}/{config.task}/{config.model}/{config.subject}/{config.task}_few_{config.shot_type}.jsonl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    samples = construct_prompt(config)
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
    parser.add_argument("--model", type=str, default="llama", help="model")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="llama,qwen")
    parser.add_argument("--task", type=str, default='mmlu_pro', help="task")
    parser.add_argument("--shot_type", type=str, default="few", help="shot_type")
    parser.add_argument("--input_dir", type=str, default="result", help="input_dir")
    parser.add_argument("--output_dir", type=str, default="baseline", help="output_dir")
    parser.add_argument("--num_examples", type=str, default=-1, help="num_examples")
    parser.add_argument("--subject", type=str, default="law", help="subject")

    args = parser.parse_args()
    print("=== [Parsed Arguments] ===")
    for key, value in vars(args).items():
        print(f"{key} = {value}")
    print("\n\n")

    main(args)

    
