from typing import Generator, Union, Any, Dict, Optional
from ...server.schemas import ChatCompletionChunk, ChatCompletionResponse
from json_repair import repair_json

def stream_text_from_response(resp: Generator) -> Generator[str, None, None]:
    for chunk in resp:
        if isinstance(chunk, ChatCompletionChunk):
            yield chunk.choices[0].delta.content or ""
        if isinstance(chunk, str):
            yield chunk

def nonstream_text_from_response(resp: ChatCompletionResponse) -> str:
    return (resp.choices[0].message.content or "").strip()
        
def normalize_response(results: Union[str, Generator, ChatCompletionResponse, ChatCompletionChunk, Any]) -> Union[str, Generator[str, None, None]]:
    """
    For tool responses, normalize into either a string or a generator[str].
    """
    if isinstance(results, str):
        return results
    if isinstance(results, ChatCompletionResponse):
        return nonstream_text_from_response(results)
    # streaming generator
    if isinstance(results, Generator):
        return stream_text_from_response(results)
    if isinstance(results, Dict): 
        return results
    try: 
        return str(results)
    except: 
        raise ValueError(f"Unsupported results type: {type(results)}")


def extract_and_repair_json(text: str, return_dict: bool = True) -> Optional[Union[str, Dict[str, Any]]]:
    """
    Extract and parse the first valid JSON object from arbitrary text (e.g., LLM output).
    
    Handles:
    - Markdown code fences (```json ... ```)
    - Extraneous text before/after JSON
    - Nested braces {...}
    - Incomplete or malformed chunks (best-effort parsing)

    Returns:
        dict if successful, otherwise None.
    """
    if not text or not isinstance(text, str):
        return None
    return repair_json(text, return_objects=return_dict)


# Internal helper to print messages
def rprint(msg: str, color: str = "cyan", underline: bool = False, rich: bool = True):
    try:
        if rich:
            from rich.console import Console
            console = Console(force_jupyter=False, force_terminal=True)
            if underline:
                console.print(f"[underline][bold {color}]{msg}[/bold {color}]")
            else:
                console.print(f"[bold {color}]{msg}[/bold {color}]")
        else:
            print(msg)
    except NameError:
        print(msg)