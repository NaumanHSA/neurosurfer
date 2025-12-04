from __future__ import annotations

import json
import re
from typing import List, Dict, Any, Optional
import logging
from neurosurfer.models.chat_models.base import BaseChatModel as BaseChatModel

logger = logging.getLogger(__name__)

# Minimal, model-agnostic system prompt (can be overridden by config)
TITLE_GENERATOR_SYSTEM_PROMPT = """You are a terse title generator for chat transcripts.

Rules:
- Output JSON only: {{"title":"..."}}
- 3-7 words, descriptive, no emojis.
- Do NOT include quotes, brackets, or markdown fences in the value.
- No trailing punctuation. No code.
- Use the existing messages as context. Summarize the main topic.

Examples:
User+Assistant discuss fixing a React event JSON error -> {{"title":"Fix React Circular JSON Error"}}
Building IPTV proxy with FastAPI and HLS -> {{"title":"FastAPI HLS Proxy Setup"}}

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

def robust_parse_title(raw: str) -> str:
    text = _strip_code_fences(raw)
    sliced = _between_braces(text)

    obj = _try_json(sliced) or _try_json(text)
    if obj and isinstance(obj, dict):
        title = obj.get("title")
        if isinstance(title, str):
            return title
    return None


class ThreadTitleGenerator:
    """
    Service to generate and parse follow-up questions from an existing LLM.
    The LLM must support a 'ask(user_prompt, system_prompt, chat_history, stream=False, **kwargs)' call.
    """
    def __init__(
        self,
        llm: BaseChatModel = None,
        system_prompt: str = TITLE_GENERATOR_SYSTEM_PROMPT,
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
        Returns a title for the conversation.
        """ 
        # Build a concise user prompt; keep system template in system_prompt
        # Use last user message as anchor; include minimal chat history
        num_recent = 4
        chat_history = [msg for msg in messages if msg["role"] != "system"][-num_recent:]
        default_title = "New Chat..."
        if chat_history:
            default_title = chat_history[0]['content'][:20]
        conversation = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
        system_prompt = self.system_prompt.format(conversation=conversation)
        
        user_query = "Generate a title for the provided conversation. Return only JSON object with title without any additional text."
        # Call the existing LLM in non-streaming mode, robust parse into suggestions
        response = self.llm.ask(
            user_prompt=user_query,
            system_prompt=system_prompt,
            stream=False,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
        ).choices[0].message.content

        parsed = robust_parse_title(response)
        title = parsed if parsed else default_title
        return self.llm._final_nonstream_response(
            call_id=self.llm.call_id,
            model=self.llm.model_name,
            content=json.dumps({"title": title}),
            prompt_tokens=len(user_query),
            completion_tokens=len(title)
        )
