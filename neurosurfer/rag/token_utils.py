from __future__ import annotations
from typing import List, Tuple
import importlib


class TokenCounter:
    """
    Vendor-agnostic token utilities.
    - If llm.tokenizer is present (HF or similar), use it.
    - Else, try tiktoken if available (OpenAI-like approximation).
    - Else, fall back to char-based heuristic (chars_per_token).
    """

    def __init__(self, llm, chars_per_token: float = 4.0):
        self.llm = llm
        self.chars_per_token = max(1e-6, float(chars_per_token))
        self._tok = getattr(llm, "tokenizer", None)
        self._tiktoken = None
        if self._tok is None:
            try:
                self._tiktoken = importlib.import_module("tiktoken")
            except Exception:
                self._tiktoken = None

    # -------- counting --------
    def count(self, text: str) -> int:
        if not text:
            return 0
        # 1) HuggingFace-style
        if self._tok is not None:
            try:
                return int(self._tok(text, return_tensors="pt").input_ids.shape[1])  # type: ignore[attr-defined]
            except Exception:
                try:
                    return int(len(self._tok.encode(text)))  # type: ignore[attr-defined]
                except Exception:
                    pass
        # 2) tiktoken (best-effort)
        if self._tiktoken is not None:
            try:
                # Choose an encoding generically; you can customize per model name if desired
                enc = self._tiktoken.get_encoding("cl100k_base")
                return int(len(enc.encode(text)))
            except Exception:
                pass
        # 3) fallback heuristic
        return int(max(1, round(len(text) / self.chars_per_token)))

    # -------- trimming --------
    def trim_to_tokens(self, text: str, max_tokens: int) -> Tuple[str, int]:
        if max_tokens <= 0 or not text:
            return "", 0
        # HF fast path
        if self._tok is not None:
            try:
                ids = self._tok(text, return_tensors="pt").input_ids[0]  # type: ignore[attr-defined]
                used = min(len(ids), max_tokens)
                trimmed = self._tok.decode(ids[:used], skip_special_tokens=True)  # type: ignore[attr-defined]
                return trimmed.strip(), int(used)
            except Exception:
                pass
        # tiktoken path
        if self._tiktoken is not None:
            try:
                enc = self._tiktoken.get_encoding("cl100k_base")
                ids = enc.encode(text)
                used = min(len(ids), max_tokens)
                trimmed = enc.decode(ids[:used])
                return trimmed.strip(), int(used)
            except Exception:
                pass
        # heuristic path: binary search on char length
        lo, hi = 0, len(text)
        best = ""
        best_tokens = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = text[:mid]
            cnt = self.count(candidate)
            if cnt <= max_tokens:
                best, best_tokens = candidate, cnt
                lo = mid + 1
            else:
                hi = mid - 1
        return best.strip(), int(best_tokens)

    # -------- chat template helpers --------
    def apply_chat_template(self, messages: List[dict]) -> str:
        """
        Try llm.tokenizer.apply_chat_template; else naive concatenation.
        """
        if self._tok is not None:
            try:
                return self._tok.apply_chat_template(  # type: ignore[attr-defined]
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass
        # fallback naive concat
        buf = []
        for m in messages:
            role = (m.get("role") or "user").upper()
            buf.append(f"{role}:\n{m.get('content','')}\n")
        buf.append("ASSISTANT:\n")
        return "\n".join(buf)
