"""
Document Chunker Module
=======================

This module provides an intelligent and extensible document chunking system designed specifically
for RAG (Retrieval-Augmented Generation) applications. The chunker intelligently breaks down
large documents into smaller, semantically meaningful chunks while preserving context and structure.

The system supports multiple chunking strategies optimized for different content types:
- **Code files**: AST-based chunking for Python, structure-aware chunking for JavaScript/TypeScript
- **Text files**: Line-based chunking for code-like content, character-based for prose
- **Structured data**: JSON object/array-based chunking
- **Documentation**: Header-aware chunking for Markdown and README files

Key Features:
    - Pluggable strategy system for different file types and custom logic
    - Line-based chunking for code (preserves syntax structure and context)
    - Character-based chunking for prose and generic text content
    - Configurable overlap between chunks to maintain context continuity
    - Comment-aware filtering to exclude comment-only blocks from chunks
    - Blacklist patterns for excluding unwanted file types (binaries, configs, etc.)
    - AST-based chunking for Python code with structural awareness
    - Custom handler system for specialized chunking requirements
    - Router system for dynamic strategy selection based on content analysis
    - Comprehensive error handling and graceful fallback mechanisms

Strategy Selection Priority:
    1. Explicit custom handler (callable function or registered handler name)
    2. Router function result pointing to a registered custom handler
    3. File extension mapping to custom handlers
    4. Built-in strategy registry by file extension
    5. Heuristic fallback (line-based for code-like content, character-based for prose)

Classes:
    - ChunkerConfig: Comprehensive configuration for chunking behavior including
      chunk sizes, overlap settings, fallback modes, and safety limits
    - Chunker: Main chunker class with pluggable strategy registry and custom handler support

Example:
    >>> from neurosurfer.rag.chunker import Chunker, ChunkerConfig
    >>>
    >>> # Configure chunking behavior
    >>> config = ChunkerConfig(
    ...     fallback_chunk_size=30,
    ...     overlap_lines=5,
    ...     char_chunk_size=1000,
    ...     comment_block_threshold=4
    ... )
    >>> chunker = Chunker(config)
    >>>
    >>> # Chunk Python code with AST awareness
    >>> chunks = chunker.chunk(python_code, file_path="script.py")
    >>>
    >>> # Register custom strategy for specific file types
    >>> def my_strategy(text, file_path):
    ...     return text.split("\\n\\n")
    >>> chunker.register([".custom"], my_strategy)
    >>>
    >>> # Use custom handler for specialized processing
    >>> def custom_handler(text, file_path=None, config=None):
    ...     # Custom chunking logic here
    ...     return process_text_customly(text)
    >>> chunks = chunker.chunk(text, custom=custom_handler)
"""
# add to your imports
from typing import List, Optional, Callable, Dict, Union, Protocol, Tuple
import os, re, json
from dataclasses import dataclass
import ast
import copy
from neurosurfer.config import config, ChunkerConfig


StrategyFn = Callable[[str, Optional[str]], List[str]]

class CustomChunkHandler(Protocol):
    def __call__(
        self,
        text: str,
        *,
        file_path: Optional[str] = None,
        config: Optional[ChunkerConfig] = None
    ) -> List[str]: ...


class Chunker:
    """
    Extensible document chunker with pluggable strategy registry for different file types.

    This class provides intelligent document chunking for RAG (Retrieval-Augmented Generation)
    systems with support for multiple file formats and customizable chunking strategies.

    Key Features:
        - Pluggable strategy system for different file types (Python, JavaScript, JSON, Markdown, etc.)
        - Line-based chunking for code (preserves structure and context)
        - Character-based chunking for prose and generic text
        - Configurable overlap between chunks to maintain context
        - Comment-aware filtering to exclude comment-only blocks
        - Blacklist patterns for excluding unwanted file types
        - AST-based chunking for Python code with structural awareness
        - Custom handler system for specialized chunking logic
        - Router system for dynamic strategy selection
        - Comprehensive error handling and fallback mechanisms

    The chunker follows a priority-based strategy selection:
    1. Explicit custom handlers (callable or registered name)
    2. Router function results
    3. File extension mappings to custom handlers
    4. Built-in strategies by file extension
    5. Heuristic fallback (line-based for code, character-based for prose)

    Args:
        config: ChunkerConfig instance controlling chunking behavior.
               Uses default configuration if not provided.

    Example:
        >>> config = ChunkerConfig(fallback_chunk_size=30, overlap_lines=5)
        >>> chunker = Chunker(config)
        >>>
        >>> # Chunk Python code with structural awareness
        >>> chunks = chunker.chunk(python_code, file_path="script.py")
        >>>
        >>> # Register custom strategy for specific file types
        >>> def my_strategy(text, file_path):
        ...     return text.split("\\n\\n")
        >>> chunker.register([".custom"], my_strategy)
    """
    def __init__(self):
        """
        Initialize the Chunker with configuration settings.

        Args:
            config: Configuration object controlling chunking behavior including
                   chunk sizes, overlap settings, fallback modes, and safety limits.
                   Defaults to a new ChunkerConfig() instance if not provided.

        Sets up:
            - Blacklist patterns for excluding unwanted file types
            - Line comment markers for comment detection
            - Built-in strategy registry for common file extensions
            - Custom handler infrastructure
            - Router and logging hooks
        """
        self.cfg = config.chunker
        self.blacklist_patterns = [
            re.compile(p) for p in [
                r'.*\.lock$', r'.*\.env.*', r'.*\.git.*', r'.*node_modules.*', r'.*__pycache__.*',
                r'.*\.DS_Store$', r'.*Thumbs\.db$', r'.*\.png$', r'.*\.jpe?g$', r'.*\.svg$', r'.*\.ico$',
                r'.*\.zip$', r'.*\.tar\.gz$', r'.*\.mp4$', r'.*\.mp3$', r'.*/dist/.*', r'.*/build/.*',
                r'.*\.idea.*', r'.*\.vscode.*', r'.*\.eslintrc.*', r'.*\.prettierrc.*', r'.*\.editorconfig',
                r'.*\.gitignore$', r'.*/LICENSE$', r'.*/CODEOWNERS$', r'.*/CONTRIBUTING\.md$', r'.*/CHANGELOG\.md$'
            ]
        ]
        self._line_comment_markers = ("#", "//", "--", "%")

        # Strategy registry: ext -> function(code, file_path) -> List[str]
        self._strategies: Dict[str, StrategyFn] = {}

        # Built-in strategies (lightweight, replace as you like)
        self.register({'.py'}, self._chunk_python)
        self.register({'.js', '.ts', '.tsx', '.jsx'}, self._chunk_javascript)
        self.register({'.json'}, self._chunk_json)
        self.register({'.md', '.txt'}, self._chunk_readme)

        # custom handler infra
        self._custom_handlers: Dict[str, CustomChunkHandler] = {}
        self._ext_to_custom: Dict[str, str] = {}
        self._router: Optional[Callable[[Optional[str], str], Optional[str]]] = None
        # optional logger hook (don’t force logging impl on caller)
        self._log: Optional[Callable[[str], None]] = None

    # handlers: registration
    def register(self, exts: List[str], fn: StrategyFn):
        for ext in exts:
            self._strategies[ext] = fn

    # ---------------------------
    # Logging hook (optional)
    # ---------------------------
    def set_logger(self, logger_fn: Callable[[str], None]):
        self._log = logger_fn

    def _log_info(self, msg: str):
        if self._log:
            self._log(msg)

    def _log_warn(self, msg: str):
        if self._log:
            self._log(f"[WARN] {msg}")

    def _log_error(self, msg: str):
        if self._log:
            self._log(f"[ERROR] {msg}")

    # ---------------------------
    # Custom handlers: registry
    # ---------------------------
    def register_custom(self, name: str, handler: CustomChunkHandler):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Custom handler name must be a non-empty string.")
        if not callable(handler):
            raise TypeError(f"Handler for '{name}' must be callable.")
        if name in self._custom_handlers:
            raise ValueError(f"A handler named '{name}' is already registered.")
        self._custom_handlers[name] = handler
        self._log_info(f"Registered custom handler '{name}'.")

    def unregister_custom(self, name: str):
        if name in self._custom_handlers:
            del self._custom_handlers[name]
            # remove any ext mappings pointing to it
            for ext, mapped in list(self._ext_to_custom.items()):
                if mapped == name:
                    del self._ext_to_custom[ext]
            self._log_info(f"Unregistered custom handler '{name}'.")

    def list_custom_handlers(self) -> List[str]:
        return sorted(self._custom_handlers.keys())

    # ---------------------------
    # Routing options
    # ---------------------------
    # Map file extensions to custom handlers
    def use_custom_for_ext(self, exts: List[str], handler_name: str):
        if handler_name not in self._custom_handlers:
            raise KeyError(f"No custom handler registered with name '{handler_name}'.")
        for ext in exts:
            key = ext.lower() if ext.startswith('.') else f".{ext.lower()}"
            self._ext_to_custom[key] = handler_name
        self._log_info(f"Mapped {exts} -> '{handler_name}'.")

    def clear_custom_for_ext(self, exts: List[str]):
        for ext in exts:
            key = ext.lower() if ext.startswith('.') else f".{ext.lower()}"
            self._ext_to_custom.pop(key, None)

    def set_router(self, router: Optional[Callable[[Optional[str], str], Optional[str]]]):
        """
        router(file_path, text) -> handler_name or None
        Called before strategies and fallbacks. Return a registered name or None.
        """
        if router and not callable(router):
            raise TypeError("Router must be callable or None.")
        self._router = router
        self._log_info(f"Router set: {bool(router)}")

    def list_ext_mappings(self) -> List[Tuple[str, str]]:
        return sorted(self._ext_to_custom.items())

    
    # ---------------------------
    # Hardened custom handler execution
    # ---------------------------
    def _sanitize_chunks(self, chunks: List[str]) -> List[str]:
        cleaned: List[str] = []
        total_chars = 0
        for ch in chunks:
            if not isinstance(ch, str):
                continue
            s = ch.strip("\n")
            if len(s.strip()) < self.cfg.min_chunk_non_ws_chars:
                continue
            if total_chars + len(s) > self.cfg.max_total_output_chars:
                # cap the last allowable piece
                remaining = self.cfg.max_total_output_chars - total_chars
                if remaining > 0:
                    s = s[:remaining]
                    cleaned.append(s)
                break
            cleaned.append(s)
            total_chars += len(s)
            if len(cleaned) >= self.cfg.max_returned_chunks:
                break
        return cleaned

    def _run_custom(self, handler_name: str, text: str, file_path: Optional[str]) -> Optional[List[str]]:
        handler = self._custom_handlers.get(handler_name)
        if not handler:
            self._log_warn(f"Custom handler '{handler_name}' not found.")
            return None
        try:
            result = handler(text, file_path=file_path, config=self.cfg)
            if not isinstance(result, list):
                self._log_warn(f"Custom handler '{handler_name}' returned non-list; ignoring.")
                return None
            sanitized = self._sanitize_chunks(result)
            if not sanitized:
                self._log_warn(f"Custom handler '{handler_name}' returned no usable chunks.")
                return None
            return sanitized
        except Exception as e:
            self._log_error(f"Custom handler '{handler_name}' failed: {e}")
            return None
            
     # ---------------------------
    # Main entry: prefer custom if provided
    # ---------------------------
    def chunk(
        self,
        text: str,
        *,
        source_id: str = None,
        file_path: Optional[str] = None,
        k: int = 40,
        custom: Optional[Union[str, CustomChunkHandler]] = None
    ) -> List[str]:
        """
        Chunk text into smaller pieces using a priority-based strategy selection.

        The method uses the following priority order:
        1. Explicit custom handler (callable or registered name)
        2. Router function result pointing to registered custom handler
        3. File extension mapping to custom handler
        4. Built-in strategy registry by file extension
        5. Heuristic fallback (line-based for code-like content, char-based for prose)

        Args:
            text: The text content to be chunked
            source_id: Optional identifier for the source document (currently unused)
            file_path: Optional file path to determine file type and apply blacklist filtering
            k: Minimum word count threshold - if text has fewer words, returns as single chunk
            custom: Optional custom chunking handler (either registered name or callable)

        Returns:
            List of text chunks. Returns empty list if text is empty, file should be skipped,
            or no valid chunks can be generated.

        Raises:
            No exceptions raised - all errors are logged and handled gracefully with fallbacks.
        """
        if not text or (file_path and self._should_skip_file(file_path)):
            return []

        # if short, no chunking
        if len(text.strip().split()) <= k:
            s = text.strip()
            return [s] if len(s) >= self.cfg.min_chunk_non_ws_chars else []

        # ---- (1) explicit custom override
        if custom is not None:
            if isinstance(custom, str):
                out = self._run_custom(custom, text, file_path)
            else:
                # ad-hoc callable
                try:
                    result = custom(text, file_path=file_path, config=self.cfg)
                    out = self._sanitize_chunks(result if isinstance(result, list) else [])
                except Exception as e:
                    self._log_error(f"Ad-hoc custom handler failed: {e}")
                    out = None
            if out:
                return out  # do not continue if succeeded

        # ---- (2) router decides
        if self._router:
            chosen = None
            try:
                chosen = self._router(file_path, text)
            except Exception as e:
                self._log_warn(f"Router error: {e}")
            if isinstance(chosen, str):
                out = self._run_custom(chosen, text, file_path)
                if out:
                    return out

        # ---- (3) ext mapping to custom
        ext = os.path.splitext(file_path or '')[-1].lower()
        mapped = self._ext_to_custom.get(ext)
        if mapped:
            out = self._run_custom(mapped, text, file_path)
            if out:
                return out

        # ---- (4) built-in strategies
        strategy = self._strategies.get(ext)
        if strategy:
            try:
                chunks = strategy(text, file_path)
                if chunks:
                    return chunks
            except Exception as e:
                self._log_warn(f"Built-in strategy failed for '{ext}': {e}")

        # ---- (5) fallback heuristic
        mode = "line" if self._looks_like_code(text) else "char"
        return self._line_windows(text) if mode == "line" else self._char_windows(text, self.cfg.char_chunk_size, self.cfg.char_overlap)
            

    def _should_skip_file(self, file_path):
        return any(p.match(file_path) for p in self.blacklist_patterns)
    
    def _split_into_chunks(self, lines):
        chunks = []
        total_lines = len(lines)
        step = self.cfg.max_chunk_lines - self.cfg.overlap_lines
        for i in range(0, total_lines, step):
            chunk = lines[i:i + self.cfg.max_chunk_lines]
            if chunk:
                cleaned = self.clean_chunk_lines(chunk, self.cfg.comment_block_threshold)
                if cleaned and not self._is_fully_commented(cleaned):
                    chunks.append("\n".join(cleaned))
        return chunks

    def _chunk_python(self, code: str) -> List[str]:
        # first filter out prompt like blocks
        code = self._filter_prompt_like_blocks(code)
        chunks = []
        try:
            tree = ast.parse(code)
            lines = code.splitlines()
            used_lines = set()
            chunk_info = []

            def add_chunk(block_lines, start, end):
                sub_chunks = self._split_into_chunks(block_lines)
                for chunk in sub_chunks:
                    chunks.append(chunk)
                    chunk_info.append((chunk, start, end))

            # Collect all top-level node ranges
            node_ranges = []
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    start = node.lineno - 1
                    end = getattr(node, 'end_lineno', node.lineno)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    start = min((d.lineno for d in node.decorator_list), default=node.lineno) - 1
                    end = getattr(node, 'end_lineno', node.lineno)
                else:
                    start = node.lineno - 1
                    end = getattr(node, 'end_lineno', node.lineno)
                node_ranges.append((start, end))

            # Sort by original order
            node_ranges.sort()
            for start, end in node_ranges:
                block = lines[start:end]
                
                if self._is_prompt_like(''.join(block)):
                    continue
                
                add_chunk(block, start, end)
                used_lines.update(range(start, end))

            # Remaining lines (e.g. comments or code not in AST)
            remaining = [
                lines[i] for i in range(len(lines))
                if i not in used_lines and lines[i].strip()
            ]
            if remaining:
                sub_chunks = self._split_into_chunks(remaining)
                chunks.extend(sub_chunks)

        except Exception:
            return self._line_windows(code)
        return chunks

    def _chunk_javascript(self, code):
        chunks = []
        patterns = [r'function\s+\w+\s*\([^)]*\)\s*\{', r'class\s+\w+\s*\{']
        combined = "|".join(patterns)
        matches = list(re.finditer(combined, code))
        if not matches:
            return self._line_windows(code)

        starts = [m.start() for m in matches] + [len(code)]
        for i in range(len(matches)):
            segment = code[starts[i]:starts[i + 1]]
            cleaned = self._clean_lines(segment.splitlines(), self.cfg.comment_block_threshold)
            if cleaned and not self._is_fully_commented(cleaned):
                chunks.append("\n".join(cleaned))
        return chunks

    def _chunk_json(self, text: str) -> List[str]:
        chunks = []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                for k, v in parsed.items():
                    sub = json.dumps({k: v}, indent=2)
                    if len(sub) <= self.cfg.json_chunk_size:
                        chunks.append(sub)
            elif isinstance(parsed, list):
                for item in parsed:
                    sub = json.dumps(item, indent=2)
                    if len(sub) <= self.cfg.json_chunk_size:
                        chunks.append(sub)
        except Exception:
            return self._char_windows(text, self.cfg.json_chunk_size, int(self.cfg.json_chunk_size * 0.2))
        return chunks

    def _chunk_readme(self, code):
        lines = code.splitlines()
        chunks = []
        current = []
        for line in lines:
            if len(current) >= self.cfg.readme_max_lines or (line.strip().startswith("#") and current):
                chunks.append("\n".join(current))
                current = []
            current.append(line)

        if current:
            chunks.append("\n".join(current))
        return chunks

    def _line_windows(self, text: str, *, window: Optional[int] = None) -> List[str]:
        """
        Break text into line-based windows, removing empty lines.
        """
        # Strip empty lines before processing
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines:
            return []

        win = min(window or self.cfg.fallback_chunk_size, self.cfg.max_chunk_lines)
        step = max(1, win - self.cfg.overlap_lines)
        out: List[str] = []
        for i in range(0, len(lines), step):
            block = self._clean_lines(lines[i:i + win])
            # Also ensure no empty lines inside block
            block = [ln for ln in block if ln.strip()]
            if block and not self._is_fully_commented(block):
                out.append("\n".join(block))
        return out

    def _char_windows(self, text: str, size: int, overlap: int) -> List[str]:
        """
        Break text into char-based windows, stripping empty lines inside each chunk.
        """
        size = max(1, size)
        overlap = max(0, min(overlap, size - 1))
        if len(text) <= size:
            # Clean out empty lines
            return ["\n".join([ln for ln in text.splitlines() if ln.strip()])]

        out: List[str] = []
        start = 0
        end = size
        while start < len(text):
            chunk = text[start:end]
            # Strip empty lines inside chunk
            chunk = "\n".join([ln for ln in chunk.splitlines() if ln.strip()])
            if chunk:
                out.append(chunk)
            start = end - overlap
            end = start + size
        return out

     # ----- Heuristics / primitives -----    
    def _looks_like_code(self, text: str) -> bool:
        # Cheap heuristic: braces, semicolons, keywords, many short lines, indent patterns
        sample = text[:4000]
        signals = 0
        if re.search(r'[;{}()<>\[\]]', sample): signals += 1
        if re.search(r'^\s*(def|class|import|from|function|var|let|const|if|for|while)\b', sample, re.M): signals += 1
        if re.search(r'^\s{2,}\S', sample, re.M): signals += 1
        if sample.count('\n') > 10 and sum(len(l) < 120 for l in sample.splitlines()) / max(1, sample.count('\n')) > 0.7:
            signals += 1
        return signals >= 2

    def _is_comment_line(self, line: str) -> bool:
        s = line.strip()
        return bool(s) and any(s.startswith(m) for m in self._line_comment_markers)

    def _is_fully_commented(self, lines: List[str]) -> bool:
        if len(lines) < self.cfg.comment_block_threshold:
            return False
        return all((ln.strip() == "" or self._is_comment_line(ln)) for ln in lines)

    def _clean_lines(self, chunk_lines, comment_block_threshold=4):
        cleaned = []
        comment_block = []
        for line in chunk_lines:
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                comment_block.append(line)
            else:
                if len(comment_block) >= comment_block_threshold:
                    comment_block = []  # drop
                else:
                    cleaned.extend(comment_block)
                    comment_block = []
                cleaned.append(line)
        if len(comment_block) < comment_block_threshold:
            cleaned.extend(comment_block)
        return cleaned

    def _is_prompt_like(self, text: str) -> bool:
        return (
            "you are" in text[:200].lower() or
            "based on the query" in text.lower() or
            "your task is" in text.lower() or
            "your job is" in text.lower() or
            "your goal is" in text.lower() or
            "{question}" in text.lower() or
            "respond with" in text.lower() or
            "{context}" in text.lower() or
            "{query}" in text.lower()
        )

    def _filter_prompt_like_blocks(self, code: str) -> str:
        """
        Finds all triple-quoted string blocks in the code and returns their (start_line, end_line)
        if they contain prompt-like content.
        """
        pattern = re.compile(r'(?:[rRuUbBfF]{0,2})([\'"]{3})(.*?)\1', re.DOTALL)
        matches = list(pattern.finditer(code))

        code_to_return = copy.deepcopy(code)
        for match in matches:
            full_text = match.group(0)
            # Estimate line range
            start_pos = match.start()
            end_pos = match.end()
            start_line = code[:start_pos].count("\n")
            end_line = code[:end_pos].count("\n")
            # prompt_block = ''.join(code.splitlines()[start_line:end_line])
            # print(full_text)
            if self._is_prompt_like(full_text):
                code_to_return = code_to_return.replace(full_text, "")
        return code_to_return