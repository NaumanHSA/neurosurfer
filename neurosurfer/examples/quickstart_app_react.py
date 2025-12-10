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

from neurosurfer.agents.react import ReActAgent, ReActConfig
from neurosurfer.server.app import NeurosurferApp
from neurosurfer.server.schemas import ChatHandlerModel, ChatHandlerMessages, AppResponseModel
from neurosurfer.server.runtime import RequestContext
from neurosurfer.server.services.rag import RAGResult

from neurosurfer.tools import Toolkit
from neurosurfer.tools.code_execution.python_exec_tool import PythonExecTool

from neurosurfer.config import config
from neurosurfer import CACHE_DIR
logging.basicConfig(level=config.app.logs_level.upper())

LOGGER = logging.getLogger("neurosurfer")

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

@ns.on_startup
async def load_model():
    """
    Initialize AI models and RAG system during application startup.

    This asynchronous startup function performs the following initialization tasks:
    1. Sets up application logging
    2. Checks for GPU availability and logs hardware info
    3. Loads the main language model (LLM) from configuration
    4. Registers the model in the application's model registry
    5. Performs a warmup inference to ensure models are ready

    Global Variables Modified:
        LOGGER: Set to application logger instance
        LLM: Set to initialized TransformersModel instance

    Configuration Dependencies:
        - config.model.unsloth: LLM model configuration

    Note:
        - This function runs automatically when the FastAPI app starts
        - GPU detection helps optimize model loading and inference
        - The warmup call ensures models are properly loaded and ready
    """
    global LOGGER
    try:
        import torch
        LOGGER.info(f"GPU Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            LOGGER.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
    except Exception:
        LOGGER.warning("Torch not found...")

    from neurosurfer.models.chat_models.transformers import TransformersModel
    MODEL_SOURCE = os.getenv("NEUROSURF_MODEL_PATH", "/home/nomi/workspace/Model_Weights/Qwen3-8B-unsloth-bnb-4bit")
    llm = TransformersModel(
        # model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        model_name=MODEL_SOURCE,
        # max_seq_length=config.base_model.max_seq_length,
        max_seq_length=8000,
        load_in_4bit=config.base_model.load_in_4bit,
        enable_thinking=config.base_model.enable_thinking,
        stop_words=config.base_model.stop_words or [],
        logger=LOGGER,
    )
    # registered models are visible in the UI. You must register a model for it to be available in the UI.
    ns.register_model(
        model=llm,
        family="llama",
        provider="Unsloth",
        description="Proxy to Llama"
    )
    from neurosurfer.server.services.chat_orchestration import MainWorkflowConfig, CodeAgentConfig
    ns._init_chat_workflow(
        config=MainWorkflowConfig(  
            default_language="english",
            default_answer_length="detailed",
            enable_rag=True,
            enable_code=True,
            log_traces=True,
            max_context_chars=16000,
            max_history_chars=12000,
            max_new_tokens=8000,
            temperature=0.7
        ),
        code_agent_config=CodeAgentConfig(
            temperature=0.7,
            max_new_tokens=8000,
            mode="analysis_only"
        )
    )

    # Warmup
    joke = llm.ask(user_prompt="Say hi!", system_prompt=config.base_model.system_prompt, stream=False)
    LOGGER.info(f"LLM ready: {joke.choices[0].message.content}")

@ns.on_shutdown
def cleanup():
    """Clean up temporary files and directories on application shutdown."""
    pass

@ns.chat()
def handler(args: ChatHandlerModel) -> AppResponseModel:
    """
    Process chat completion requests with RAG-enhanced context.

    This is the main chat handler that processes incoming chat requests, optionally
    enhances them with relevant context from uploaded documents using RAG, and
    generates responses using the configured language model.

    Args:
        args (ChatHandlerModel): The chat handler model contains:
            - user_id: User/session identifier
            - thread_id: Session/thread identifier for context management
            - message_id: Message identifier for context management
            - has_files_message: Whether the message contains files
            - model: The model to use for generation, selected from the UI
            - messages: ChatHandlerMessages
                - user_query: Last user message
                - user_msgs: List of user messages
                - assistant_msgs: List of assistant messages
                - system_msgs: List of system messages
                - converstaion: List of conversation messages
            - temperature: Sampling temperature for response generation
            - max_tokens: Maximum tokens to generate in response
            - stream: Whether to stream the response
            - system_prompt: Optional system prompt for generation
            - tools: Optional list of tools to use for generation
            - tool_choice: Optional tool choice for generation
            - metadata: Optional metadata for generation
            - files: Optional list of uploaded files for context

    Returns:
        The AppResponseModel, which can be either:
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
    global LOGGER

    # Prepare inputs
    user_query = args.messages.user_query    # Last user message
    # Minimal chat history excluding system messages, max 10
    num_recent = 10
    conversation_messages = args.messages.converstaion
    chat_history = conversation_messages[-num_recent:-1]

    print(f"\nChat History:\n{chat_history}\n\n")
    # Model call (stream or non-stream handled by router)
    kwargs = {"temperature": args.temperature, "max_new_tokens": args.max_tokens}

    final_answer = ns.run_agent(
        user_id=args.user_id,
        thread_id=args.thread_id,
        message_id=args.message_id,
        user_query=user_query,
        has_files_message=args.has_files_message,
        chat_history=chat_history,
    )
    return final_answer

def create_app():
    return ns.app

def main():
    ns.run()

if __name__ == "__main__":
    main()
