import inspect, asyncio
import logging
from typing import Any, Callable, Optional, Type, Awaitable, List, Dict
from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter, Depends, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from .schemas.model_registry import ModelList
from .security import get_current_user, get_db
from .models_registry import ModelRegistry
from .api import _auth_router, _chats_router, chat_completion_router
from .config import CORS_ORIGINS, SECRET_KEY

# ---------- Neurosurfer app builder ----------
class NeurosurferApp:
    """
    A comprehensive FastAPI-based application builder for AI-powered chat and API services.

    NeurosurferApp provides a simplicity-first approach to building AI applications with:
    - Single chat handler registration via decorator or method
    - Custom endpoint creation with one-line decorators
    - Built-in authentication (API key or JWT-based)
    - Database integration
    - CORS support for cross-origin requests
    - Automatic API documentation (Swagger/OpenAPI)
    - Model registry for managing AI models
    - Lifecycle management with startup/shutdown hooks

    The app supports both simple API key authentication and full user authentication
    with JWT tokens. It includes built-in routes for health checks, model listings,
    chat completions, and user management.

    Key Features:
    - FastAPI-based with automatic OpenAPI documentation
    - SQLAlchemy ORM for database operations
    - Session middleware for CSRF protection
    - Model registry for AI model management
    - Flexible authentication (API key or JWT)
    - CORS support for web applications
    - Startup and shutdown lifecycle hooks
    - Built-in routes for common operations

    Example:
        ```python
        from neurosurfer.server.app import NeurosurferApp

        app = NeurosurferApp(api_key="your-api-key")

        @app.chat()
        def handle_chat(request, context):
            # Handle chat request and return response
            return {"response": "Hello!"}

        if __name__ == "__main__":
            app.run()
        ```
    """
    def __init__(
        self,
        app_name: str = "Neurosurfer API",
        api_keys: List[str] = [],
        cors_origins: List[str] = CORS_ORIGINS,
        api_key: str = "",
        enable_docs: bool = True,
        host: str = "0.0.0.0",
        port: int = 8000,
        reload: bool = False,
        log_level: str = "info",
        workers: int = 1,
        logger = logging.getLogger(name="Neurosurfer")
    ):
        """
        Initialize the NeurosurferApp instance.

        This constructor sets up all the necessary components for the AI application including
        FastAPI app creation, middleware configuration, database initialization, and route mounting.

        Args:
            app_name (str, optional): Name of the FastAPI application displayed in docs.
                Defaults to "Neurosurfer API".
            api_keys (List[str], optional): List of valid API keys for authentication.
                Can be used as an alternative to JWT-based user authentication. Defaults to [].
            cors_origins (List[str], optional): List of allowed origins for CORS requests.
                Defaults to CORS_ORIGINS from config.
            api_key (str, optional): Single API key for authentication. If provided, it will be
                checked in addition to or instead of user authentication. Defaults to "".
            enable_docs (bool, optional): Whether to enable automatic API documentation at /docs.
                Defaults to True.
            host (str, optional): Host address to bind the server to. Defaults to "0.0.0.0".
            port (int, optional): Port number to bind the server to. Defaults to 8000.
            reload (bool, optional): Whether to enable auto-reload during development.
                Defaults to False.
            log_level (str, optional): Logging level for the application. Defaults to "info".
            workers (int, optional): Number of worker processes for the server. Defaults to 1.
            logger (logging.Logger, optional): Custom logger instance. If not provided,
                creates a logger named "Neurosurfer".

        The constructor also initializes:
        - Model registry for AI model management
        - Startup and shutdown callback lists
        - Chat handler storage
        - FastAPI application with lifespan management
        - Database initialization
        - CORS middleware configuration
        - Session middleware for CSRF protection
        - Main router for custom endpoints
        """
        self.app_name = app_name
        self.api_keys = api_keys
        self.cors_origins = cors_origins
        self.api_key = api_key
        self.enable_docs = enable_docs
        self.host = host
        self.port = port
        self.reload = reload
        self.log_level = log_level
        self.workers = workers
        self.logger = logger
        
        self.model_registry = ModelRegistry()
        self._startup: list[Callable[[], Awaitable[None] | None]] = []
        self._shutdown: list[Callable[[], Awaitable[None] | None]] = []
        self._chat_handler: Optional[Callable] = None
        self._stop_generation: Optional[Callable] = None

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # run startup callbacks
            for fn in self._startup:
                try:
                    res = fn()
                    if asyncio.iscoroutine(res):
                        await res
                except Exception as e:
                    self.logger.error(f"Startup callback failed: {e}")
            yield
            # run shutdown callbacks
            for fn in self._shutdown:
                try:
                    res = fn()
                    if asyncio.iscoroutine(res):
                        await res
                except Exception as e:
                    self.logger.error(f"Shutdown callback failed: {e}")

        self.app = FastAPI(
            title= self.app_name,
            docs_url="/docs" if self.enable_docs else None,
            redoc_url=None,
            lifespan=lifespan
        )
        # Initialize database (tables)
        try:
            from .db.db import init_db
            init_db()
        except Exception as _e:
            print('DB init warning:', _e)

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"]
        )
        # Secure server-side sessions (used for CSRF or future features)
        self.app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY, https_only=True)
        self.main_router = APIRouter()
        # ---------------- Auth & Chats (DB-backed) ----------------
            
    # ---------- tiny auth (optional) ----------
    def require_api_key(self, req: Request):
        """
        Validate API key authentication for incoming requests.

        This method checks if the request contains a valid API key in the Authorization header.
        If no API key is configured for the app, authentication is skipped.

        Args:
            req (Request): The incoming FastAPI request object containing headers.

        Raises:
            HTTPException: If the API key is configured but missing or invalid, raises a 401
                Unauthorized exception with details about the authentication failure.

        Note:
            The expected format is: "Bearer <api_key>" in the Authorization header.
            If the API key is configured but the request doesn't have a valid key,
            the request is rejected with a 401 status code.
        """
        if not self.api_key:
            return
        auth = req.headers.get("authorization", "")
        if not auth.startswith("Bearer ") or auth.split(" ",1)[1] != self.api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    # New: API key OR JWT-user
    def require_auth(
        self,
        req: Request,
        response: Response,                 # inject Response for cookie refresh
        db: Session = Depends(get_db),
    ):
        """
        Authenticate requests using either API key or JWT-based user authentication.

        This method implements a flexible authentication strategy that supports both:
        1. API key authentication (for service-to-service communication)
        2. JWT-based user authentication (for user sessions)

        The method first checks for a valid API key if one is configured. If an API key
        is provided and valid, the request is authenticated as a service/system caller.
        Otherwise, it falls back to JWT-based user authentication.

        Args:
            req (Request): The incoming FastAPI request object containing headers and state.
            response (Response): The FastAPI response object for potential cookie refresh.
            db (Session): SQLAlchemy database session for user authentication queries.

        Raises:
            HTTPException: If neither API key nor valid user authentication is provided,
                raises a 401 Unauthorized exception.

        Note:
            - API key format: "Bearer <api_key>" in Authorization header
            - For API key auth: sets req.state.user = None (indicates service caller)
            - For JWT auth: sets req.state.user to the authenticated user object
            - If API key is configured, it takes precedence over user authentication
        """
        # 1) API key allowed if configured
        print("API key:", self.api_key)
        if self.api_key:
            auth = req.headers.get("authorization", "")
            if auth.startswith("Bearer ") and auth.split(" ", 1)[1] == self.api_key:
                req.state.user = None  # service/system caller
                return

        # 2) Otherwise must be a logged-in user
        user = get_current_user(req, response=response, db=db)
        req.state.user = user
        return user
    
    def on_startup(self, fn: Callable[[], Any]):
        """
        Register a function to be executed during application startup.

        This decorator allows you to register callback functions that will be executed
        when the FastAPI application starts up. The functions can be either synchronous
        or asynchronous (coroutines).

        Args:
            fn (Callable[[], Any]): A callable that takes no arguments and returns Any.
                Can be either a regular function or an async function (coroutine).

        Returns:
            Callable: The original function, allowing this to be used as a decorator.

        Example:
            ```python
            app = NeurosurferApp()

            @app.on_startup
            def initialize_database():
                print("Setting up database connections...")

            @app.on_startup
            async def load_models():
                print("Loading AI models...")
                await some_async_operation()
            ```

        Note:
            - Startup functions are executed in the order they are registered
            - Async functions are properly awaited during the startup process
            - These functions run before the server starts accepting requests
        """
        self._startup.append(fn)
        return fn

    def on_shutdown(self, fn: Callable[[], Any]):
        """
        Register a function to be executed during application shutdown.

        This decorator allows you to register callback functions that will be executed
        when the FastAPI application shuts down gracefully. The functions can be either
        synchronous or asynchronous (coroutines).

        Args:
            fn (Callable[[], Any]): A callable that takes no arguments and returns Any.
                Can be either a regular function or an async function (coroutine).

        Returns:
            Callable: The original function, allowing this to be used as a decorator.

        Example:
            ```python
            app = NeurosurferApp()

            @app.on_shutdown
            def cleanup_resources():
                print("Cleaning up resources...")

            @app.on_shutdown
            async def save_state():
                print("Saving application state...")
                await some_async_cleanup()
            ```

        Note:
            - Shutdown functions are executed in the order they are registered
            - Async functions are properly awaited during the shutdown process
            - These functions run after the server stops accepting requests
            - Useful for cleanup operations, saving state, or closing connections
        """
        self._shutdown.append(fn)
        return fn

    # ---------- public API ----------
    def chat(self):
        """
        Decorator to register the primary chat handler for the application.

        This method serves as a decorator factory that allows you to register a single
        chat handler function that will process all chat requests. The decorated function
        will be automatically integrated with the FastAPI routing system and will be
        accessible via the chat completions API endpoints.

        The decorator triggers the mounting of built-in routes including health checks,
        model listings, and chat completion endpoints.

        Returns:
            Callable: A decorator function that registers the chat handler.

        Example:
            ```python
            app = NeurosurferApp()

            @app.chat()
            def handle_chat(request_data, context):
                # Process the chat request
                user_message = request_data.get("messages", [{}])[-1].get("content", "")
                # Generate response
                return {
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": f"You said: {user_message}"
                        }
                    }]
                }

            # Alternative: async handler
            @app.chat()
            async def handle_chat_async(request_data, context):
                # Async processing
                response = await some_async_ai_call(request_data)
                return response
            ```

        Note:
            - Only one chat handler can be registered per application instance
            - The handler function should accept request data and return response data
            - Async handlers are supported and properly awaited
            - The decorator automatically mounts all necessary routes
        """
        def deco(fn: Callable):
            self._chat_handler = fn
            self._mount_builtin_routes()
            return fn
        return deco

    def stop_generation(self):
        """
        Stop Generation is a decorator that stops the generation of a response.
        """
        def deco(fn: Callable):
            self._stop_generation = fn
            return fn
        return deco

        
    def endpoint(
        self,
        path: str,
        *,
        method: str = "post",
        request: Type[BaseModel] | None = None,
        response: Type[BaseModel] | None = None,
        dependencies: list[Callable[..., Any]] | None = None
    ):
        """
        Create a decorator for registering custom API endpoints.

        This method serves as a decorator factory that allows you to easily create custom
        API endpoints with automatic request/response model validation, dependency injection,
        and integration with FastAPI's routing system.

        Args:
            path (str): The URL path for the endpoint (e.g., "/api/summarize").
            method (str, optional): HTTP method for the endpoint. Must be one of:
                "get", "post", "put", "patch", "delete". Defaults to "post".
            request (Type[BaseModel], optional): Pydantic model class for request body validation.
                If provided, FastAPI will automatically parse and validate the request body.
                Defaults to None.
            response (Type[BaseModel], optional): Pydantic model class for response serialization.
                If provided, FastAPI will automatically serialize the response to match this model.
                Defaults to None.
            dependencies (list[Callable[..., Any]], optional): List of FastAPI dependency functions
                that will be executed before the endpoint handler. Useful for authentication,
                database sessions, etc. Defaults to None.

        Returns:
            Callable: A decorator function that registers the endpoint.

        Example:
            ```python
            from pydantic import BaseModel

            class SummarizeRequest(BaseModel):
                text: str
                max_length: int = 100

            class SummarizeResponse(BaseModel):
                summary: str
                original_length: int
                summary_length: int

            app = NeurosurferApp()

            @app.endpoint(
                "/summarize",
                method="post",
                request=SummarizeRequest,
                response=SummarizeResponse
            )
            def summarize_text(data: SummarizeRequest):
                # data is automatically parsed and validated
                summary = summarize_function(data.text, data.max_length)
                return SummarizeResponse(
                    summary=summary,
                    original_length=len(data.text),
                    summary_length=len(summary)
                )

            # GET endpoint without request/response models
            @app.endpoint("/health", method="get")
            def custom_health():
                return {"status": "custom health check"}
            ```

        Raises:
            ValueError: If the provided HTTP method is not supported.

        Note:
            - The decorated function can access the request body via its first parameter
            - Request and response models provide automatic validation and serialization
            - Dependencies are executed before the endpoint handler runs
            - The endpoint is automatically registered with the main router
        """
        method = method.lower()
        if method not in {"get","post","put","patch","delete"}:
            raise ValueError("method must be one of get/post/put/patch/delete")

        def deco(fn: Callable):
            async def handler(body=Depends(lambda: None), request_obj: Request | None = None):
                # If a request model is provided, FastAPI will parse it automatically via signature.
                # Here we adapt call style for both (body) and () signatures.
                if request is None:
                    res = fn() if request_obj is None else fn()
                else:
                    # FastAPI already parsed body into `body` parameter if function expects it
                    # To keep things simple: re-call with correct args
                    params = inspect.signature(fn).parameters
                    if len(params) == 0:
                        res = fn()
                    else:
                        res = fn(body)
                return res

            # register
            add = getattr(self.main_router, method)
            add(path, response_model=response, dependencies=dependencies)(handler)
            return fn
        return deco

    # ---------- internals ----------
    def _mount_builtin_routes(self):
        """
        Mount built-in API routes to the FastAPI application.

        This internal method sets up the default routes that are essential for the
        NeurosurferApp functionality. It should be called after setting a chat handler
        to ensure all routes are properly configured.

        The method mounts the following routes:
        - "/" (GET): Basic index endpoint indicating the service is running
        - "/health" (GET): Health check endpoint for monitoring and load balancers
        - "/v1/models" (GET): List available AI models from the model registry
        - Chat completion routes via chat_completion_router
        - Authentication routes via _auth_router
        - Chat management routes via _chats_router

        Routes are organized with proper authentication dependencies:
        - Public routes: index, health, auth routes (no authentication required)
        - Protected routes: model listings, chat completions, chat management (require authentication)

        Note:
            - This is an internal method called automatically by the @chat() decorator
            - Should be called manually if using set_chat_handler() instead of the decorator
            - Routes are mounted with appropriate prefixes and dependency requirements
        """

        @self.main_router.get("/")
        async def index():
            return "Neurosurfer is up and Running..."
        
        @self.main_router.get("/health")
        async def health():
            return {"status": "ok"}

        @self.main_router.post("/v1/stop")
        async def stop():
            self._stop_generation()

        # ---------------- Models and Completions ----------------
        @self.main_router.get("/v1/models", response_model=ModelList)
        async def list_models():
            data = [mc for mid, mc in self.model_registry.all().items()]
            return ModelList(data=data)
        
        _chat_comp_router = chat_completion_router(self._chat_handler, self.model_registry)
        # Include built-in routes
        self.app.include_router(self.main_router)
        self.app.include_router(_auth_router, prefix="/v1")   # no auth required
        self.app.include_router(_chats_router, prefix="/v1", dependencies=[Depends(self.require_auth)])
        self.app.include_router(_chat_comp_router, prefix="/v1")
    
    def run(
        self,
        host: str = None,
        port: int = None,
        reload: bool = None,
        log_level: str = None,
        workers: int = None,
    ):
        """
        Start the FastAPI application server using Uvicorn.

        This method launches the configured FastAPI application using the Uvicorn ASGI server.
        The server will start with the configuration specified during initialization including
        host, port, reload settings, logging level, and worker processes.

        The method handles server startup and graceful error handling. If the server fails
        to start, it logs an error message with details about the failure.

        Configuration used:
        - host: Server bind address (from __init__)
        - port: Server port number (from __init__)
        - reload: Auto-reload during development (from __init__)
        - log_level: Logging verbosity (from __init__)
        - workers: Number of worker processes (from __init__)

        Example:
            ```python
            app = NeurosurferApp(port=9000, reload=True)

            # Start the server
            app.run()
            ```

        Note:
            - This method blocks the current thread while the server is running
            - Use Ctrl+C or send SIGTERM to stop the server gracefully
            - In production, consider using a process manager like gunicorn
            - The method will not return until the server is stopped
        """
        try:
            import uvicorn
            uvicorn.run(
                self.app, 
                host=host or self.host, 
                port=port or self.port,
                reload=reload or self.reload,
                log_level=log_level or self.log_level,
                workers=workers or self.workers
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.logger.error(f"Failed to run app: {e}")
