from dataclasses import dataclass
from typing import Optional

@dataclass
class AgentDelimiters:
    sof: str = "<__final_answer__>"
    eof: str = "</__final_answer__>"

class BaseAgent:
    def __init__(self, sof: str = "<__final_answer__>", eof: str = "</__final_answer__>") -> None:
        self.delims = AgentDelimiters(sof=sof, eof=eof)
        self.stop_event = False

    def stop_generation(self):
        self.stop_event = True
