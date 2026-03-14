from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def iter_jsonl(path: str | Path, *, allow_errors: bool = True) -> Iterator[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return iter(())
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError:
                if not allow_errors:
                    raise
                continue


def read_jsonl(path: str | Path, *, allow_errors: bool = True) -> list[Dict[str, Any]]:
    return list(iter_jsonl(path, allow_errors=allow_errors))


def write_jsonl(
    path: str | Path,
    records: Iterable[Dict[str, Any]],
    *,
    mode: str = "a",
    ensure_parent: bool = True,
) -> None:
    p = Path(path)
    if ensure_parent:
        ensure_dir(p.parent)
    with p.open(mode, encoding="utf-8") as f:
        for record in records:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")


def read_json(path: str | Path, *, default: Optional[Any] = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(
    path: str | Path,
    obj: Any,
    *,
    indent: int = 2,
    ensure_parent: bool = True,
) -> None:
    p = Path(path)
    if ensure_parent:
        ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
        f.write("\n")
