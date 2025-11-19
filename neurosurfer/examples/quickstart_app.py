# app.py
"""
Neurosurfer Application Entry Point

This module serves as the main application file that configures and launches the Neurosurfer
AI-powered chat application. It integrates various components including:

- FastAPI server setup via NeurosurferApp
- AI model initialization (LLM and embedding models)
- RAG (Retrieval-Augmented Generation) system setup
- Chat request handling with context management
- File upload and processing capabilities

The application provides a complete AI chat interface with:
- Multiple model support (configurable via CONFIG)
- Vector-based document retrieval and context injection
- Session-based chat management
- Real-time streaming responses
- File upload for context enhancement

Global Variables:
    BASE_DIR (str): Temporary directory for code sessions and file processing
    LLM (BaseChatModel): The primary language model instance
    EMBEDDER_MODEL (BaseEmbedder): Embedding model for vector similarity
    LOGGER (logging.Logger): Application logger instance
    RAG (RAGOrchestrator): RAG system for context retrieval

Functions:
    load_model(): Initializes AI models and RAG system on startup
    cleanup(): Cleans up temporary files on shutdown
    handler(): Processes chat requests with RAG enhancement

Usage:
    Run this file directly to start the Neurosurfer server:
        python app.py

    The server will be available at the configured host and port (see CONFIG.py).
"""
from typing import List, Generator
import os, shutil, logging

from neurosurfer.models.embedders.base import BaseEmbedder
from neurosurfer.models.chat_models import BaseChatModel as BaseChatModel
from neurosurfer.agents.rag.chunker import Chunker
from neurosurfer.agents.rag.filereader import FileReader

from neurosurfer.server.app import NeurosurferApp
from neurosurfer.server.schemas import ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChunk
from neurosurfer.server.runtime import RequestContext
from neurosurfer.server.services.rag import RAGOrchestrator

from neurosurfer.config import config
from neurosurfer import CACHE_DIR
logging.basicConfig(level=config.app.logs_level.upper())


# Create app instance
ns = NeurosurferApp(
    app_name=config.app.app_name,
    api_keys=[],
    enable_docs=config.app.enable_docs,
    cors_origins=config.app.cors_origins,
    host=config.app.host_ip,
    port=config.app.host_port,
    reload=config.app.reload,
    log_level=config.app.logs_level,
    workers=config.app.workers
)

"""
Global Application State

These variables hold the core components of the Neurosurfer application that are
initialized during startup and used throughout the application's lifecycle.
"""
LLM: BaseChatModel = None
EMBEDDER_MODEL: BaseEmbedder = None
LOGGER: logging.Logger = None
RAG: RAGOrchestrator | None = None

@ns.on_startup
async def load_model():
    """
    Initialize AI models and RAG system during application startup.

    This asynchronous startup function performs the following initialization tasks:
    1. Sets up application logging
    2. Checks for GPU availability and logs hardware info
    3. Loads the main language model (LLM) from configuration
    4. Registers the model in the application's model registry
    5. Initializes the embedding model for vector similarity
    6. Sets up the RAG (Retrieval-Augmented Generation) orchestrator
    7. Performs a warmup inference to ensure models are ready

    Global Variables Modified:
        LOGGER: Set to application logger instance
        LLM: Set to initialized TransformersModel instance
        EMBEDDER_MODEL: Set to initialized SentenceTransformerEmbedder
        RAG: Set to configured RAGOrchestrator

    Configuration Dependencies:
        - config.model.unsloth: LLM model configuration
        - config.model.embedder: Embedding model name
        - config.model.system_prompt: Default system prompt for warmup
        - config.app.database_path: RAG persistence directory

    Note:
        - This function runs automatically when the FastAPI app starts
        - GPU detection helps optimize model loading and inference
        - The warmup call ensures models are properly loaded and ready
        - RAG system includes file processing and context retrieval
    """
    global EMBEDDER_MODEL, LOGGER, LLM, RAG
    LOGGER = logging.getLogger("neurosurfer")
    try:
        import torch
        LOGGER.info(f"GPU Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            LOGGER.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
    except Exception:
        LOGGER.warning("Torch not found...")

    from neurosurfer.models.chat_models.transformers import TransformersModel
    from neurosurfer.models.embedders.sentence_transformer import SentenceTransformerEmbedder

    LLM = TransformersModel(
        model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        max_seq_length=config.base_model.max_seq_length,
        load_in_4bit=config.base_model.load_in_4bit,
        enable_thinking=config.base_model.enable_thinking,
        stop_words=config.base_model.stop_words or [],
        logger=LOGGER,
    )
    # registered models are visible in the UI. You must register a model for it to be available in the UI.
    ns.model_registry.add(
        llm=LLM,
        family="llama",
        provider="Unsloth",
        description="Proxy to Llama"
    )
    EMBEDDER_MODEL = SentenceTransformerEmbedder(
        model_name="intfloat/e5-large-v2", 
        logger=LOGGER
    )
    RAG = RAGOrchestrator(
        embedder=EMBEDDER_MODEL,
        gate_llm=LLM,
        top_k=10,
        verbose=True,
        logger=LOGGER,
    )
    # Warmup
    joke = LLM.ask(user_prompt="Say hi!", system_prompt=config.base_model.system_prompt, stream=False)
    LOGGER.info(f"LLM ready: {joke.choices[0].message.content}")

@ns.on_shutdown
def cleanup():
    """Clean up temporary files and directories on application shutdown."""
    pass

@ns.chat()
def handler(request: ChatCompletionRequest, ctx: RequestContext) -> ChatCompletionResponse | Generator[ChatCompletionChunk, None, None]:
    """
    Process chat completion requests with RAG-enhanced context.

    This is the main chat handler that processes incoming chat requests, optionally
    enhances them with relevant context from uploaded documents using RAG, and
    generates responses using the configured language model.

    Args:
        request (ChatCompletionRequest): The chat completion request containing:
            - messages: List of chat messages with roles and content
            - thread_id: Session/thread identifier for context management
            - temperature: Sampling temperature for response generation
            - max_tokens: Maximum tokens to generate in response
            - stream: Whether to stream the response
            - files: Optional list of uploaded files for context
        ctx (RequestContext): Request context containing metadata including:
            - actor_id: User/session identifier
            - Additional request metadata

    Returns:
        The response from the LLM, which can be either:
            - Complete response object (non-streaming)
            - Streaming response generator (streaming mode)

    Processing Flow:
        1. Extract user messages, system messages, and conversation history
        2. Apply RAG enhancement if files/context available for the thread
        3. Configure generation parameters (temperature, max_tokens)
        4. Call LLM with enhanced query and chat history

    RAG Enhancement:
        - Checks if RAG system is available and thread_id is provided
        - Applies document retrieval and context injection
        - Logs RAG usage statistics (similarity scores, usage decisions)
        - Falls back to original query if no relevant context found

    Configuration:
        - Uses DEFAULT_SYSTEM_PROMPT if no system message provided
        - Applies temperature/max_tokens limits with fallbacks to config defaults
        - Maintains recent chat history (last 10 messages by default)

    Note:
        - Thread-based context allows for persistent conversations
        - File uploads are processed and converted to document chunks
        - Streaming responses are supported for real-time interaction
        - RAG context injection happens before LLM generation
    """
    global LLM, RAG

    # Resolve actor/thread ids (thread id is part of JSON now)
    actor_id = (getattr(ctx, "meta", {}) or {}).get("actor_id", 0)
    thread_id = request.thread_id

    # Prepare inputs
    user_msgs: List[str] = [m["content"] for m in request.messages if m["role"] == "user"]
    system_msgs = [m["content"] for m in request.messages if m["role"] == "system"]
    system_prompt = system_msgs[0] if system_msgs else config.base_model.system_prompt    # First system message or default
    user_query = user_msgs[-1] if user_msgs else ""    # Last user message

    # Minimal chat history excluding system messages, max 10
    num_recent = 10
    conversation_messages = [msg for msg in request.messages if msg["role"] != "system"]
    chat_history = conversation_messages[-num_recent:-1]

    temperature = request.temperature if (request.temperature and 2 > request.temperature > 0) else config.base_model.temperature
    max_tokens = request.max_tokens if (request.max_tokens and request.max_tokens > 512) else config.base_model.max_new_tokens
    kwargs = {"temperature": temperature, "max_new_tokens": max_tokens}

    if RAG and thread_id is not None:
        rag_res = RAG.apply(
            actor_id=actor_id,
            thread_id=thread_id,
            user_query=user_query,
            files=[f.model_dump() for f in (request.files or [])] if request.files else None,
        )
        user_query = rag_res.augmented_query
        if rag_res.used:
            LOGGER.info(f"[RAG] used context (top_sim={rag_res.meta.get('top_similarity', 0):.3f})")
        else:
            LOGGER.info(f"[RAG] no context (reason={rag_res.meta.get('reason')})")

    # Model call (stream or non-stream handled by router)
    return LLM.ask(
        user_prompt=user_query,
        system_prompt=system_prompt,
        chat_history=chat_history,
        stream=request.stream,
        **kwargs
    )

@ns.stop_generation()
def stop_handler():
    print("Stopping generation...")
    global LLM
    LLM.stop_generation()

def create_app():
    return ns.app

def main():
    ns.run()

if __name__ == "__main__":
    main()
