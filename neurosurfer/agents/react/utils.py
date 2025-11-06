from typing import Generator, Union, Any
from ...server.schemas import ChatCompletionChunk, ChatCompletionResponse

def stream_text_from_chunk(chunk: ChatCompletionChunk) -> str:
    return chunk.choices[0].delta.content

def nonstream_text_from_response(resp: ChatCompletionResponse) -> str:
    return resp.choices[0].message.content

def normalize_tool_observation(observation: Union[str, Generator, ChatCompletionResponse, ChatCompletionChunk, Any]) -> Union[str, Generator]:
    """
    For tool responses, normalize into either a string or a generator[str].
    """
    if isinstance(observation, str):
        return observation
    if isinstance(observation, ChatCompletionResponse):
        return nonstream_text_from_response(observation)
    # streaming generator
    if isinstance(observation, Generator):
        for chunk in observation:
            if isinstance(chunk, ChatCompletionChunk):
                yield chunk.choices[0].delta.content or ""

    # single chunk (rare)
    # if isinstance(observation, ChatCompletionChunk):
    #     def gen():
    #         yield stream_text_from_chunk(observation)
    #     return gen()
    try:
        return str(observation)
    except:
        raise ValueError(f"Unsupported observation type: {type(observation)}")
