def chat_completion_response_wrapper(
    id: str, 
    model: str, 
    content: str = "",
    stream: bool = False,
    finish_reason: str = "stop"
):
    if stream:
        content_object = {
            "index": 0,
            "delta": {"content": content} if finish_reason is None else {},
            "finish_reason": finish_reason
        }
    else:
        content_object = {
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": finish_reason
        }
    return {
        "id": id,
        "object": "chat.completion",
        "model": model,
        "choices": [content_object],
        "usage": {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None
        }
    }