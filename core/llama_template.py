from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LlamaTokens:
    begin_of_text: str = "<|begin_of_text|>"
    start_header: str = "<|start_header_id|>"
    end_header: str = "<|end_header_id|>"
    eot: str = "<|eot_id|>"

    def header(self, role: str) -> str:
        return f"{self.start_header}{role}{self.end_header}"

    def message(
        self,
        role: str,
        content: str,
        *,
        add_eot: bool = True,
        trailing_newlines: int = 0,
    ) -> str:
        text = f"{self.header(role)}\n{content}"
        if add_eot:
            text += self.eot
        if trailing_newlines:
            text += "\n" * trailing_newlines
        return text

    def assistant_prefix(self) -> str:
        return f"{self.header('assistant')}\n\n"


LLAMA = LlamaTokens()

LLAMA_SYSTEM = LLAMA.header("system")
LLAMA_USER = LLAMA.header("user")
LLAMA_ASSISTANT = LLAMA.header("assistant")
LLAMA_EOT = LLAMA.eot
