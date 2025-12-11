from typing import Literal, Optional
from dataclasses import dataclass

MemoryStyle = Literal["text", "structured", "both"]

@dataclass
class PythonExecToolConfig:
    max_new_tokens: int = 4096
    temperature: float = 0.1

    max_code_retries: int = 3
    include_code_in_answer: bool = True
    max_table_rows: int = 20   # for DataFrame/Series pretty printing
    memory_style: MemoryStyle = "text"   # "text" | "structured" | "both"
    max_context_length: int = 2000

    soc: Optional[str] = "<__code__>"
    eoc: Optional[str] = "</__code__>"

