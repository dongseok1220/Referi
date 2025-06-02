import os
import json 

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM



def load_llm(model_name: str, hf_token: str | None = None):
    print(f"Loading: {model_name}...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

    model = (AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token).to(device).eval())

    return tokenizer, model, device

def load_existing_likelihoods(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            existing_data = json.load(f)
            return existing_data
    return []

def load_model_outputs(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            model_outputs = [json.loads(line) for line in f]
            return model_outputs
    else:
        print(f"Error: {file_path} not found.")
        return None
    
def calculate_diffs(current_likelihoods):

    def mean_list(l):
        return (sum(l) / len(l)) if l is not None else 0.0
    
    def compute_sums(entry_list):
        ce_sums = []
        entropy_sums = []
        for e in entry_list:
            ce_sums.append(mean_list(e.get('ce_losses', [])))
            entropy_sums.append(mean_list(e.get('entropy', [])))
        
        ce_mean = sum(ce_sums) / len(ce_sums) if ce_sums else 0
        entropy_mean = sum(entropy_sums) / len(entropy_sums) if entropy_sums else 0
        
        return ce_mean, entropy_mean

    no_replace_ce, no_replace_entropy = compute_sums(current_likelihoods.get('no_replace_likelihoods', []))
    replace_ce, replace_entropy = compute_sums(current_likelihoods.get('replace_likelihoods', []))
    zero_ce, zero_entropy = compute_sums(current_likelihoods.get('zero_likelihoods', []))


    current_likelihoods['no_replace_ce_diff'] = no_replace_ce - zero_ce
    current_likelihoods['no_replace_entropy_diff'] = no_replace_entropy - zero_entropy

    current_likelihoods['replace_ce_diff'] = replace_ce - zero_ce
    current_likelihoods['replace_entropy_diff'] = replace_entropy - zero_entropy

def calculate_cross_entropy_loss_with_topk(prompt, answer, model, tokenizer, device, top_k=2):
    combined_input = prompt + answer
    combined_ids = tokenizer(combined_input, return_tensors="pt").input_ids.to(device)

    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
    prompt_len = prompt_ids.shape[1]

    labels = combined_ids.clone()
    labels[:, :prompt_len] = -100

    with torch.no_grad():
        outputs = model(input_ids=combined_ids, labels=labels)
        ce_loss = outputs.loss               
        logits = outputs.logits              # shape: [batch_size, seq_len, vocab_size]

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    shift_logits_2d = shift_logits.view(-1, model.config.vocab_size)
    shift_labels_1d = shift_labels.view(-1)

    loss_fct_none = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    token_wise_ce_loss = loss_fct_none(shift_logits_2d, shift_labels_1d)

    valid_mask = (shift_labels_1d != -100)
    valid_ce = token_wise_ce_loss[valid_mask]         # [N_valid]
    valid_logits = shift_logits_2d[valid_mask]        # [N_valid, vocab_size]
    valid_labels = shift_labels_1d[valid_mask]        # [N_valid]

    valid_probs = F.softmax(valid_logits, dim=-1)     # [N_valid, vocab_size]
    valid_entropy = -torch.sum(
        valid_probs * torch.log(valid_probs + 1e-10),
        dim=-1
    )  # [N_valid]

    summed_ce_loss = valid_ce.sum()
    valid_tokens = valid_mask.sum()
    manual_ce_loss = summed_ce_loss / valid_tokens  

    if not torch.isclose(torch.tensor(manual_ce_loss.item()), torch.tensor(ce_loss.item()), atol=1e-6):
        print("Loss mismatch!")
        print(f"Auto loss: {ce_loss.item()}, Manual loss: {manual_ce_loss.item()}")

    answer_label_probs = []
    for i in range(valid_probs.shape[0]):
        label_id = valid_labels[i].item()
        prob_for_label = valid_probs[i, label_id].item()  
        answer_label_probs.append(prob_for_label)


    topk_tokens_list = []
    topk_probs_list = []
    for i in range(valid_probs.shape[0]):
        tk_vals, tk_ids = torch.topk(valid_probs[i], k=top_k)
        tk_probs = tk_vals.tolist()       # [top_k]
        tk_ids = tk_ids.tolist()          # [top_k]
        tk_tokens = tokenizer.convert_ids_to_tokens(tk_ids)
        
        topk_tokens_list.append(tk_tokens)
        topk_probs_list.append(tk_probs)

    result = {
        "ce_losses": valid_ce.tolist(),  
        "entropy": valid_entropy.tolist(),
        "answer_labels": tokenizer.convert_ids_to_tokens(valid_labels.tolist()),
        "answer_label_probs": answer_label_probs,  
        "topk_probs": topk_probs_list # for CoT-WP
    }

    return result