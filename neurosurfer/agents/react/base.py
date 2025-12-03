from dataclasses import dataclass
from typing import Optional

@dataclass
class AgentDelimiters:
    sof: str = "<__final_answer__>"
    eof: str = "</__final_answer__>"
    sot: str = "<__think__>"
    eot: str = "</__think__>"
    soa: str = "Action:"

class BaseAgent:
    def __init__(
        self, 
        sof: str = "<__final_answer__>", 
        eof: str = "</__final_answer__>",
        sot: str = "<__think__>",
        eot: str = "</__think__>",
        soa: str = "Action:",
    ) -> None:
        self.delims = AgentDelimiters(sof=sof, eof=eof, sot=sot, eot=eot, soa=soa)
        self.stop_event = False

    def stop_generation(self):
        self.stop_event = True
