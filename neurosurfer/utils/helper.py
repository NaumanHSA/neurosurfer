import os
from pathlib import Path
import re
import logging
import tempfile
import zipfile
import shutil
from typing import Callable, Dict, Optional, Generator, Union, Literal, List, Any, Tuple, Set
from datetime import datetime


def create_context(results):
    docs_dict = dict()
    context = ''
    for r in results:
        filename = r['metadata']['filename']
        chunk = r['text'].replace(f"{filename} — ", "")
        if filename not in docs_dict:
            docs_dict[filename] = []
        docs_dict[filename].append(chunk)
        # print(f"{r['metadata']['filename']} — Distance: {r['distance']:.3f}\n{r['text']}")
        
    for file_name, chunks in docs_dict.items():
        context += f"\n################### {file_name} ###################"
        chunks_merged = '\n\n'.join(chunks)
        context += f"\n{chunks_merged}\n"
    return context

def build_chat_context(chat_history, n_recent_chats=10):
    """
    Builds a formatted conversation context string from the last 10 chat history records.
    
    Args:
        chat_history (list): List of dicts like {"role": "user" or "assistant", "content": "..."}
    
    Returns:
        str: Formatted context string.
    """
    if not chat_history:
        return "No prior conversation."

    # Get at most the last n_recent_chats messages
    recent_history = chat_history[-n_recent_chats:]

    # Build the context string
    context_lines = []
    for msg in recent_history:
        role = msg.get("role", "").capitalize()
        content = msg.get("content", "").strip()
        context_lines.append(f"{role}: {content}")

    context_str = "\n".join(context_lines)
    return context_str

def chat_completion_wrapper(call_id: str, model_name:str, content:str):
    return {
        'id': call_id,
        'model': model_name,
        'created': str(datetime.now()),
        'object': 'chat.completion',
        'choices': [
            {
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': str(content)
                },
                'logprobs': None,
                'finish_reason': "stop"
            }
        ]
    }
    
def get_text_only_history(chat_history: List[Dict], num_recent_chats: int = 10):
    # Generic base64 data URI pattern for any mime type
    base64_pattern = re.compile(r"\n*\s*data:[a-zA-Z0-9\-\+\.\/]+;base64,[A-Za-z0-9+/=\r\n]+",  re.IGNORECASE)
    cleaned_history = []
    recent_history = chat_history[-min(len(chat_history), num_recent_chats):]
    for msg in recent_history:
        content = msg.get("content", "")
        # Remove base64 blocks
        text_only = base64_pattern.sub("", content).strip()
        if text_only:
            cleaned_history.append({
                "role": msg["role"],
                "content": text_only
            })
    return cleaned_history


def generate_folder_structure(
    root_path: Union[str, os.PathLike],
    max_depth: int = 5,
    exclude_dirs: Optional[List[str]] = None,
    supported_files: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Generate a tree-like folder structure.
    Supports both directory paths and .zip files.
    If a zip is given, it is extracted to a temporary directory,
    processed, and then deleted.
    """
    exclude_dirs = set(exclude_dirs or [])
    supported_files = set(supported_files or [])
    tmpdir = None
    is_zip = str(root_path).lower().endswith(".zip")
    # --- Handle zip extraction ---
    if is_zip:
        if not os.path.isfile(root_path):
            raise ValueError(f"Invalid zip file: {root_path}")
        tmpdir = tempfile.mkdtemp(prefix="tree_zip_")
        with zipfile.ZipFile(root_path, "r") as zf:
            zf.extractall(tmpdir)
        root_path = tmpdir  # work on extracted dir

    # --- Recursive tree builder ---
    def _tree(dir_path: str, prefix: str = "", depth: int = 0) -> Optional[str]:
        if depth > max_depth:
            return None
        try:
            entries = sorted(os.listdir(dir_path))
        except Exception:
            return None

        output = ""
        for i, entry in enumerate(entries):
            path = os.path.join(dir_path, entry)
            is_dir = os.path.isdir(path)

            # Exclude hidden/system entries and user-specified dirs
            if entry.startswith((".", "%", "_")) or (is_dir and entry in exclude_dirs):
                continue
            
            # # Exclude empty files
            # if not is_dir and os.path.splitext(entry)[1] == "":
            #     continue

            # Only show dirs and supported files
            if is_dir or (not supported_files or os.path.splitext(entry)[1] in supported_files) or os.path.splitext(entry)[1] == "":
                connector = "└── " if i == len(entries) - 1 else "├── "
                output += f"{prefix}{connector}{entry}\n"
                if is_dir:
                    extension = "    " if i == len(entries) - 1 else "│   "
                    sub_tree = _tree(path, prefix + extension, depth + 1) or ""
                    output += sub_tree
        return output

    # --- Build the tree ---
    tree__ = _tree(root_path)
    result = f"/\n{tree__}" if tree__ else None

    # --- Cleanup zip extraction ---
    if tmpdir:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return result

def is_prompt_like(text):
    return (
        "You are" in text[:200] or
        "Based on the query" in text or
        "Your task is" in text or
        "Your job is" in text or
        "Your goal is" in text or
        "{question}" in text or
        "Respond with" in text or
        "{context}" in text or
        "{query}" in text
    )
