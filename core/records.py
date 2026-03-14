from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


def chat_to_plain(prompt_field: Any) -> str:
    if isinstance(prompt_field, list):
        return "\n".join(str(obj.get("content", "")) for obj in prompt_field)
    if isinstance(prompt_field, dict):
        ordered = ["system", "user", "assistant"]
        return "\n".join(str(prompt_field[r]) for r in ordered if r in prompt_field)
    if prompt_field is None:
        return ""
    return str(prompt_field)


def extract_prompt(record: Dict[str, Any]) -> str:
    doc = record.get("doc") or {}
    if isinstance(doc.get("input_final_prompts"), list) and doc["input_final_prompts"]:
        return doc["input_final_prompts"][0]
    if "prompt" in doc:
        return str(doc["prompt"])
    if "prompt" in record:
        return chat_to_plain(record.get("prompt"))
    return ""


def _coerce_outputs(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        if len(value) == 1 and isinstance(value[0], list):
            return [str(v) for v in value[0]]
        return [str(v) for v in value]
    return [str(value)]


def extract_model_outputs(record: Dict[str, Any]) -> List[str]:
    outputs = record.get("model_outputs")
    if outputs is not None:
        return _coerce_outputs(outputs)
    resps = record.get("resps") or []
    if isinstance(resps, list) and resps:
        return _coerce_outputs(resps[0])
    return []


def get_record_id(record: Dict[str, Any], fallback_idx: int) -> Any:
    return (
        record.get("id")
        or record.get("idx")
        or record.get("doc_id")
        or record.get("question_id")
        or fallback_idx
    )


@dataclass
class NormalizedRecord:
    idx: Any
    prompt: str
    entry: Dict[str, Any]
    model_outputs: List[str]
    raw: Dict[str, Any]


def normalize_record(record: Dict[str, Any], fallback_idx: int) -> NormalizedRecord:
    doc = record.get("doc") or {}
    entry = record.get("entry") or doc.get("entry") or doc or {}
    return NormalizedRecord(
        idx=get_record_id(record, fallback_idx),
        prompt=extract_prompt(record),
        entry=entry,
        model_outputs=extract_model_outputs(record),
        raw=record,
    )
