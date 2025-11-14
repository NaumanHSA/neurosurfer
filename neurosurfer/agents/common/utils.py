from typing import Generator, Union, Any, Dict
from ...server.schemas import ChatCompletionChunk, ChatCompletionResponse

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
    if isinstance(results, Dict): return results
    try: return str(results)
    except: raise ValueError(f"Unsupported results type: {type(results)}")
