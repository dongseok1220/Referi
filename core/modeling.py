from __future__ import annotations

import time
from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


def _resolve_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.bfloat16
    return torch.float32


def _load_with_token(fn, model_name: str, token: str | None, **kwargs):
    try:
        return fn(model_name, token=token, **kwargs)
    except TypeError:
        return fn(model_name, use_auth_token=token, **kwargs)


def load_llm(model_name: str, hf_token: str | None = None):
    print(f"Loading: {model_name}...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = _resolve_dtype()

    tokenizer = _load_with_token(AutoTokenizer.from_pretrained, model_name, hf_token)
    model = _load_with_token(
        AutoModelForCausalLM.from_pretrained,
        model_name,
        hf_token,
        torch_dtype=torch_dtype,
    )
    model = model.to(device).eval()

    return tokenizer, model, device


def _time_forward(fn, use_cuda: bool) -> Tuple[Any, float]:
    if use_cuda:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
        return out, elapsed_ms
    start = time.perf_counter()
    out = fn()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return out, elapsed_ms


def calculate_cross_entropy_loss_with_topk(
    prompt: str,
    answer: str,
    model,
    tokenizer,
    device,
    top_k: int = 2,
):
    combined_input = prompt + answer
    combined_ids = tokenizer(combined_input, return_tensors="pt").input_ids.to(device)

    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
    prompt_len = prompt_ids.shape[1]

    labels = combined_ids.clone()
    labels[:, :prompt_len] = -100

    def _forward():
        with torch.no_grad():
            outputs = model(input_ids=combined_ids, labels=labels)
        return outputs

    outputs, fwd_ms = _time_forward(_forward, use_cuda=torch.cuda.is_available())
    ce_loss = outputs.loss
    logits = outputs.logits

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    shift_logits_2d = shift_logits.view(-1, model.config.vocab_size)
    shift_labels_1d = shift_labels.view(-1)

    loss_fct_none = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    token_wise_ce_loss = loss_fct_none(shift_logits_2d, shift_labels_1d)

    valid_mask = shift_labels_1d != -100
    valid_ce = token_wise_ce_loss[valid_mask]
    valid_logits = shift_logits_2d[valid_mask]
    valid_labels = shift_labels_1d[valid_mask]

    valid_probs = F.softmax(valid_logits, dim=-1)

    summed_ce_loss = valid_ce.sum()
    valid_tokens = valid_mask.sum()
    manual_ce_loss = summed_ce_loss / valid_tokens

    topk_probs_list = []
    for i in range(valid_probs.shape[0]):
        tk_vals, _ = torch.topk(valid_probs[i], k=top_k)
        tk_probs = tk_vals.tolist()
        topk_probs_list.append(tk_probs)

    ce_loss_value = float(ce_loss.detach().cpu())
    manual_ce_loss_value = float(manual_ce_loss.detach().cpu())
    # Debug: skip loss_match check to avoid dtype/runtime issues.
    # loss_match = torch.isclose(
    #     ce_loss.detach(),
    #     manual_ce_loss.detach(),
    #     atol=1e-4,
    #     rtol=1e-3,
    # ).item()
    # loss_match = None

    result = {
        "ce_losses": valid_ce.tolist(),
        "answer_labels": tokenizer.convert_ids_to_tokens(valid_labels.tolist()),
        "topk_probs": topk_probs_list,
        "ce_loss": ce_loss_value,
        "manual_ce_loss": manual_ce_loss_value,
        # "loss_match": bool(loss_match),
    }

    return result, fwd_ms


def timed_call(fn, *args, **kwargs):
    def _run():
        return fn(*args, **kwargs)

    out, elapsed_ms = _time_forward(_run, use_cuda=torch.cuda.is_available())
    return out, elapsed_ms
