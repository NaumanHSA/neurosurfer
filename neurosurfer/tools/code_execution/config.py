from typing import Literal, Optional
from dataclasses import dataclass

MemoryStyle = Literal["text", "structured", "both"]

@dataclass
class PythonExecToolConfig:
    max_code_retries: int = 3
    include_code_in_answer: bool = True
    max_table_rows: int = 20   # for DataFrame/Series pretty printing
    memory_style: MemoryStyle = "text"   # "text" | "structured" | "both"

    soc: Optional[str] = "<__code__>"
    eoc: Optional[str] = "</__code__>"

