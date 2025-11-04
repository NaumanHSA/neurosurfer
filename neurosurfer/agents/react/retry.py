import time
from dataclasses import dataclass

@dataclass
class RetryPolicy:
    max_parse_retries: int = 2
    max_tool_errors: int = 2
    backoff_sec: float = 0.8

    def sleep(self, attempt: int) -> None:
        # simple linear backoff; replace with exponential if you prefer
        time.sleep(self.backoff_sec * (attempt + 1))
