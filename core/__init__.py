"""Shared utilities for the project."""

from .io import read_json, write_json, read_jsonl, write_jsonl, iter_jsonl
from .records import (
    chat_to_plain,
    extract_prompt,
    extract_model_outputs,
    get_record_id,
    normalize_record,
)

try:
    from .modeling import load_llm, calculate_cross_entropy_loss_with_topk, timed_call
except ModuleNotFoundError:
    load_llm = None
    calculate_cross_entropy_loss_with_topk = None
    timed_call = None

try:
    from .likelihood import PromptSet, run_likelihood
except ModuleNotFoundError:
    PromptSet = None
    run_likelihood = None

__all__ = [
    "read_json",
    "write_json",
    "read_jsonl",
    "write_jsonl",
    "iter_jsonl",
    "chat_to_plain",
    "extract_prompt",
    "extract_model_outputs",
    "get_record_id",
    "normalize_record",
    "load_llm",
    "calculate_cross_entropy_loss_with_topk",
    "timed_call",
    "PromptSet",
    "run_likelihood",
]
