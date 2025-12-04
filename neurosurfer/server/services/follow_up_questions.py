from __future__ import annotations

import json
import re
from typing import List, Dict, Any, Optional
import logging
from neurosurfer.models.chat_models.base import BaseChatModel as BaseChatModel

logger = logging.getLogger(__name__)

# Minimal, model-agnostic system prompt (can be overridden by config)
DEFAULT_FOLLOWUPS_SYSTEM_PROMPT = """Generate exactly 3 short, crisp follow-up questions (≤12 words) 
based on the conversation so far.

Guidelines:
- Questions must relate to the prior topic
- Be intriguing, varied, and thought-provoking
- Use clear, natural language
- Avoid repetition or filler

Respond only in JSON:
{{
    "suggestions": [
        <Question 1>,
        <Question 2>,
        <Question 3>
    ]
}}

CONVERSATION:
{conversation}

""".strip()

# def _strip_code_fences(s: str) -> str:
#     # Same idea, using DOTALL so '.' matches newlines
#     return re.sub(r"``````", r"\1", s, flags=re.IGNORECASE | re.DOTALL).strip()
def _strip_code_fences(s: str) -> str:
    return re.sub(r"```[\s\S]*?```", "", s).strip()

def _between_braces(s: str) -> str:
    a, b = s.find("{"), s.rfind("}")
    return s[a:b + 1] if (a != -1 and b != -1 and b > a) else s

def _try_json(s: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(s)
    except Exception:
        return None

def _clean_suggestion_text(text: str) -> str:
    """
    Normalize one suggestion string by:
    - stripping whitespace
    - removing trailing commas
    - unescaping \" -> "
    - removing outer quotes if present
    """
    t = str(text).strip()
    # Remove trailing commas and spaces
    t = re.sub(r'[,\s]+$', '', t)
    # Unescape JSON-style quotes
    t = t.replace('\\"', '"')
    # Strip outer quotes once
    if len(t) >= 2 and t[0] == '"' and t[-1] == '"':
        t = t[1:-1].strip()
    return t.strip()

def _extract_suggestions_array_text(s: str) -> Optional[List[str]]:
    m = re.search(r'"suggestions"\s*:\s*\[([\s\S]*?)(\]|\Z)', s, flags=re.IGNORECASE)
    if not m:
        return None
    body = m.group(1)
    parts = re.split(r'\r?\n|,(?=(?:[^"]*"[^"]*")*[^"]*$)', body)
    items: List[str] = []
    for t in parts:
        t = t.strip()
        if not t:
            continue
        t = re.sub(r'^[-*\s]+', '', t)
        t = t.strip('", ')
        if len(t) >= 5:
            items.append(t)
    out: List[str] = []
    seen = set()
    for x in items:
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out or None

def _fallback_questions(s: str, k: int = 3) -> str:
    cand = re.split(r'\r?\n|(?<=\?)\s+', s)
    cand = [x.strip() for x in cand if len(x.strip()) > 6]
    with_q = [x for x in cand if x.endswith("?")]
    picked = (with_q or cand)[:k]
    out, seen = [], set()
    for x in picked:
        if x not in seen:
            out.append(x)
            seen.add(x)
    # return json array of questions
    return json.dumps({"suggestions": out})

def robust_parse_followups(raw: str) -> str:
    text = _strip_code_fences(raw)
    sliced = _between_braces(text)

    # ---------- 1) Try strict JSON paths ----------
    for candidate in (sliced, text):
        obj = _try_json(candidate)
        if obj and isinstance(obj, dict):
            arr = obj.get("suggestions")
            if isinstance(arr, list):
                cleaned: List[str] = []
                for x in arr:
                    t = _clean_suggestion_text(x)
                    if t:
                        cleaned.append(t)
                if cleaned:
                    return json.dumps({"suggestions": cleaned[:3]})
    # ---------- 2) Try approximate array extraction ----------
    approx = _extract_suggestions_array_text(sliced) or _extract_suggestions_array_text(text)
    if approx:
        cleaned: List[str] = []
        for x in approx:
            t = _clean_suggestion_text(x)
            if t:
                cleaned.append(t)
        if cleaned:
            return json.dumps({"suggestions": cleaned[:3]})
    # ---------- 3) Fallback: heuristic questions ----------
    fallback_json = _fallback_questions(sliced or text)
    # Make sure fallback also has clean strings
    try:
        obj = json.loads(fallback_json)
        if isinstance(obj, dict) and isinstance(obj.get("suggestions"), list):
            cleaned = [_clean_suggestion_text(x) for x in obj["suggestions"] if _clean_suggestion_text(x)]
            return json.dumps({"suggestions": cleaned[:3]})
    except Exception:
        pass
    # Absolute last resort
    return fallback_json


class FollowUpQuestions:
    """
    Service to generate and parse follow-up questions from an existing LLM.
    The LLM must support a 'ask(user_prompt, system_prompt, chat_history, stream=False, **kwargs)' call.
    """
    def __init__(
        self,
        llm: BaseChatModel = None,
        system_prompt: str = DEFAULT_FOLLOWUPS_SYSTEM_PROMPT,
        temperature: float = 0.7,
        max_new_tokens: int = 1024,
    ):
        self.llm = llm
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def set_llm(self, llm: BaseChatModel):
        self.llm = llm
    
    def generate(self, messages: List[Dict[str, str]]) -> str:
        """
        - messages: list of {'role': 'system'|'user'|'assistant', 'content': str}
        Returns a list of up to 3 questions.
        """ 
        # Build a concise user prompt; keep system template in system_prompt
        # Use last user message as anchor; include minimal chat history
        num_recent = 4
        chat_history = [msg for msg in messages if msg["role"] != "system"][-num_recent:]
        conversation = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
        system_prompt = self.system_prompt.format(conversation=conversation)
        
        user_query = "Generate follow-up questions based on the provided conversation. Return only JSON array of questions without any additional text."
        # Call the existing LLM in non-streaming mode, robust parse into suggestions
        response = self.llm.ask(
            user_prompt=user_query,
            system_prompt=system_prompt,
            stream=False,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
        ).choices[0].message.content

        parsed = robust_parse_followups(response)
        print("Foolow up Qustions  --: ----", parsed)
        return self.llm._final_nonstream_response(
            call_id=self.llm.call_id,
            model=self.llm.model_name,
            content=parsed,
            prompt_tokens=len(user_query),
            completion_tokens=len(parsed)
        )
