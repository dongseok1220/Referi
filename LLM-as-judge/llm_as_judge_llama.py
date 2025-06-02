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


def load_data_and_fewshot(args):
    if args.task == "mmlu_pro":
        dataset, fewshot = load_mmlu_pro()

    elif args.task == "math500": 
        file_path = f"data/math500/test.jsonl"
        with open(file_path, 'r', encoding='utf-8') as f:
            dataset = [json.loads(line) for line in f]
        fewshot = list_fewshot_samples()
    
    elif args.task == "gpqa":
        dataset = load_examples("data/gpqa/gpqa_diamond.csv", seed=0)
        with open("gpqa/chain_of_thought_examples.json", 'r') as f:
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
    system_prompt = (
        "<|start_header_id|>system<|end_header_id|>Your job is selecting the most accurate response among multiple candidates. "
        "You will receive a question and several candidate answers labeled candidate1, candidate2, etc. "
        "Please summarize the debate very briefly and then conclude which single candidate is the most plausible. "
        "Output exactly in this format:\n"
        "Summary: <brief summary>\n"
        "Conclusion: candidate<number>\n"
        "Remember to choose only one candidate as the final answer.<|eot_id|>\n\n<|start_header_id|>user<|end_header_id|>"
    )
    before_fewshot = "The below examples are well-constructed gold question and answer pairs for the same task.\n\n"
    before_question = "Now, let’s select the most proper answer for the given question\n"

    output_res_path = f"{args.input_dir}/{args.task}/{args.model}"

    samples = []

    if args.task == "math500":
        output_res_path = os.path.join(output_res_path, f"{args.task}_few.jsonl")
        res = load_model_outputs(output_res_path)

        for idx, (entry, r) in enumerate(zip(dataset, res)):
            question = entry.get('problem', '')
            model_outputs = r.get('resps', [])[0]


            user_prompt = before_fewshot

            if fewshot != None: 
                for example in fewshot:
                        user_prompt += "Problem: " + example["problem"] + "\n\n" 
                        user_prompt += example["solution"] + "\n\n"
            user_prompt += before_question

            user_prompt += f"Question: {question}\n"

            for i, output in enumerate(model_outputs, start=1):
                user_prompt += f"candidate{i}: {output}\n"

            user_prompt += (
                "\nReminder: Your answer must follow this exact format —\n"
                "Summary: <your brief summary>\n"
                "Conclusion: candidate<number>\n"
            )

            user_prompt += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

            message = system_prompt + user_prompt

            sample = {"idx": idx, "prompt": message, "entry": entry}
            samples.append(sample)
        
    elif args.task == "mmlu_pro":
        subjects = list(dataset.keys())
        for subject in tqdm(subjects): 
            res_path = os.path.join(output_res_path, f"{subject}_result.jsonl")
            res = load_model_outputs(res_path)
            
            start_prompt = "The following are multiple choice questions (with answers) about {}. Think step by" \
                " step and then output the answer in the format of \"The answer is (X)\" at the end.\n\n" \
            .format(subject)

            start_prompt = before_fewshot

            if fewshot != None:
                for each in fewshot[subject]:
                    start_prompt += format_example(each["question"], each["options"], each["cot_content"])
            
            start_prompt += before_question
            

            random.seed(42)
            test_data = random.sample(dataset[subject], min(300, len(dataset[subject])))
            
            for idx, (entry, r) in enumerate(zip(test_data, res)):
                
                model_outputs = r.get('resps', [])[0]

                question = "Question: {}\nOptions: ".format(entry['question'])
                choice_map = "ABCDEFGHIJ"
                for i, opt in enumerate(entry['options']):
                    question += "{}. {}\n".format(choice_map[i], opt)
                            
                user_prompt = start_prompt + f"{question}\n"
                for i, output in enumerate(model_outputs, start=1):
                    user_prompt += f"candidate{i}: {output}\n"

                user_prompt += (
                    "\nReminder: Your answer must follow this exact format —\n"
                    "Summary: <your brief summary>\n"
                    "Conclusion: candidate<number>\n"
                )
                user_prompt += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

                message = system_prompt + user_prompt

                sample = {"idx": idx, "prompt": message, "entry": entry}
                samples.append(sample)
    
    elif args.task == "gpqa": 
        def chain_of_thought_prompt(json_data, example: Example) -> str:
            """Creates a chain-of-thought prompt given a single example."""

            prompt = generate_prompt_from_examples(json_data, with_explanations=True)
            prompt += before_question
            prompt += f"Question: {example.question}"
            prompt += f"\nChoices:\n(A) {example.choice1}\n(B) {example.choice2}\n(C) {example.choice3}\n(D) {example.choice4}"

            return prompt
        
        output_res_path = os.path.join(output_res_path, f"{args.task}_few.jsonl")
        res = load_model_outputs(output_res_path)



        start_prompt = before_fewshot
        for idx, (example, r) in enumerate(zip(dataset, res)):
            user_prompt = start_prompt

            if fewshot != None: 
                user_prompt += chain_of_thought_prompt(fewshot, example)
            
            entry = r.get('entry', {})
            model_outputs = r.get('resps', [])[0]

            for i, output in enumerate(model_outputs, start=1):
                user_prompt += f"candidate{i}: {output}\n"

            user_prompt += (
                "\nReminder: Your answer must follow this exact format —\n"
                "Summary: <your brief summary>\n"
                "Conclusion: candidate<number>\n"
            )
            user_prompt += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            
            message = system_prompt + user_prompt

            sample = {"idx": idx, "prompt": message, "entry": entry}
            samples.append(sample)

    elif args.task == "hotpotqa": 
        output_res_path = os.path.join(output_res_path, f"{args.task}_few.jsonl")
        res = load_model_outputs(output_res_path)


        if fewshot != None: 
            fewshot_prompt = before_fewshot
            for qa in fewshot:
                question = qa["Q"]
                answer = qa["A"]

                fewshot_prompt += f"Q: {question}\nA: {answer}\n\n"
        fewshot_prompt += before_question

        for idx, (entry, r) in enumerate(zip(dataset, res)):
            if fewshot != None: 
                user_prompt = fewshot_prompt + f"Q: {entry['question']}." + "\n\n" 
            
            entry = r.get('entry', {})
            model_outputs = r.get('resps', [])[0]

            for i, output in enumerate(model_outputs, start=1):
                user_prompt += f"candidate{i}: {output}\n"

            user_prompt += (
                "\nReminder: Your answer must follow this exact format —\n"
                "Summary: <your brief summary>\n"
                "Conclusion: candidate<number>\n"
            )
            user_prompt += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            
            message = system_prompt + user_prompt
        
            sample = {"idx": idx, "prompt": message, "entry": entry}
            samples.append(sample)

    elif args.task == "drop": 
        output_res_path = os.path.join(output_res_path, f"{args.task}_few.jsonl")
        res = load_model_outputs(output_res_path)


        if fewshot != None: 
            fewshot_prompt = before_fewshot
            for qa in fewshot:
                question = qa["Q"]
                answer = qa["A"]

                fewshot_prompt += f"Q: {question}\nA: {answer}\n\n"
        fewshot_prompt += before_question

        for idx, (entry, r) in enumerate(zip(dataset, res)):
            if fewshot != None: 
                user_prompt = fewshot_prompt + f"Q: {entry['passage']} {entry['question']}" + "\n\n"
            
            entry = r.get('entry', {})
            model_outputs = r.get('resps', [])[0]

            for i, output in enumerate(model_outputs, start=1):
                user_prompt += f"candidate{i}: {output}\n"

            user_prompt += (
                "\nReminder: Your answer must follow this exact format —\n"
                "Summary: <your brief summary>\n"
                "Conclusion: candidate<number>\n"
            )
            user_prompt += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

            message = system_prompt + user_prompt

            sample = {"idx": idx, "prompt": message, "entry": entry}
            samples.append(sample)

    elif args.task == "musr_efficiently" or args.task == "musr_location":
        from musr.op_icl_fixed import op_fewshot, few_shot_op_instruction, test_op_instruction
        from musr.ta_icl_fixed import ta_fewshot, few_shot_ta_instruction, test_ta_instruction

        if args.task == "musr_location":
            few_shot_examples = op_fewshot  
            few_instruction = few_shot_op_instruction
            test_instruction = test_op_instruction
        elif args.task == 'musr_efficiently':
            few_shot_examples = ta_fewshot
            few_instruction = few_shot_ta_instruction
            test_instruction = test_ta_instruction

        output_res_path = os.path.join(output_res_path, f"{args.task}_few.jsonl")
        res = load_model_outputs(output_res_path)

        for idx, (entry, r) in enumerate(zip(dataset, res)):
            model_outputs = r.get('resps', [])[0]

            question = entry['question'].strip()
            context = entry['context'].strip()
            choices = entry['choices']['text']
            labels = ['A', 'B', 'C', 'D', 'E', 'F'][:len(choices)]
            choice_str = '\n'.join([f'{labels[idx]}: {choices[idx]}' for idx in range(len(choices))])
            original_question_part = f"{context}\n\n{question}\n\n{choice_str}"
            start_prompt = before_fewshot
            
            user_prompt = start_prompt
            for (q, a) in few_shot_examples:
                user_prompt += q + "\n\n"
                user_prompt += a + "\n\n"

            user_prompt += before_question
            user_prompt += original_question_part + "\n\n" 

            for i, output in enumerate(model_outputs, start=1):
                user_prompt += f"candidate{i}: {output}\n"

            user_prompt += (
                "\nReminder: Your answer must follow this exact format —\n"
                "Summary: <your brief summary>\n"
                "Conclusion: candidate<number>\n"
            )
            user_prompt += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

            message = system_prompt + user_prompt

            sample = {"idx": idx, "prompt": message, "entry": entry}
            samples.append(sample)

    else: 
        return None
        
    return samples


def main(config):

    save_path = f"{config.output_dir}/{config.task}/llama/{config.task}_{config.shot_type}.jsonl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    dataset, few_shot = load_data_and_fewshot(config)

    samples = construct_prompt(config, dataset, few_shot)
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

            existing_data = {json.loads(line)['prompt'] for line in f}  
    else:
        existing_data = set()  

    if samples:

        with open(save_path, "a", encoding='utf-8') as f:  
            for sample in tqdm(samples, total=len(samples)):
                if sample['prompt'] in existing_data:  
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
    parser.add_argument("--model", type=str, default="llama", help="model")
    parser.add_argument("--task", type=str, default="math500", help="task")
    parser.add_argument("--shot_type", type=str, default="few", help="shot_type")
    parser.add_argument("--input_dir", type=str, default="result", help="input_dir")
    parser.add_argument("--output_dir", type=str, default="llm_as_judge", help="output_dir")
    parser.add_argument("--num_examples", type=int, default=-1, help="num_examples")


    args = parser.parse_args()
    print("=== [Parsed Arguments] ===")
    for key, value in vars(args).items():
        print(f"{key} = {value}")
    print("\n\n")
    
    main(args)
