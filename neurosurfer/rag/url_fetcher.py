from __future__ import annotations

import hashlib
import ipaddress
import json
import logging
import mimetypes
import os
import re
import socket
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from html.parser import HTMLParser


def _sha256(s: str) -> str:
    h = hashlib.sha256()
    h.update(s.encode("utf-8"))
    return h.hexdigest()


def _now_ts() -> float:
    return time.time()


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _is_http_url(url: str) -> bool:
    try:
        u = urlparse(url)
        return u.scheme in ("http", "https") and bool(u.netloc)
    except Exception:
        return False


def _normalize_url(url: str) -> str:
    """
    Normalize URL for caching/fingerprinting.
    - Lowercase scheme/host
    - Strip default ports
    - Keep path/query/fragment as-is
    """
    u = urlparse(url.strip())
    host = (u.hostname or "").lower()
    scheme = (u.scheme or "").lower()

    # Remove default ports
    netloc = host
    if u.port:
        if not ((scheme == "http" and u.port == 80) or (scheme == "https" and u.port == 443)):
            netloc = f"{host}:{u.port}"

    # Preserve username/password if present (rare; generally avoid for security)
    if u.username or u.password:
        # If you want, you can block auth URLs entirely; for now we preserve.
        auth = u.username or ""
        if u.password:
            auth += f":{u.password}"
        netloc = f"{auth}@{netloc}"

    return urlunparse((scheme, netloc, u.path or "", u.params or "", u.query or "", u.fragment or ""))


def _resolve_host_ips(hostname: str) -> set[str]:
    """
    Resolve hostname to a set of IP strings (v4/v6). If resolution fails, returns empty set.
    """
    ips: set[str] = set()
    try:
        infos = socket.getaddrinfo(hostname, None)
        for family, _, _, _, sockaddr in infos:
            if family == socket.AF_INET:
                ips.add(sockaddr[0])
            elif family == socket.AF_INET6:
                ips.add(sockaddr[0])
    except Exception:
        return set()
    return ips


def _is_private_or_local_ip(ip_str: str) -> bool:
    """
    Blocks:
    - private ranges
    - loopback
    - link-local
    - multicast
    - reserved / unspecified
    """
    try:
        ip = ipaddress.ip_address(ip_str)
    except Exception:
        return True
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _content_type_main(ct: Optional[str]) -> str:
    if not ct:
        return ""
    return ct.split(";")[0].strip().lower()


def _parse_content_disposition_filename(cd: Optional[str]) -> Optional[str]:
    if not cd:
        return None
    # Very lightweight parsing for filename=
    m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^\";]+)"?', cd, flags=re.IGNORECASE)
    if not m:
        return None
    return m.group(1).strip()


def _guess_ext(url: str, content_type: Optional[str], content_disposition: Optional[str]) -> str:
    """
    Guess a file extension for saving downloads so FileReader can route properly.
    Priority:
    1) Content-Disposition filename extension
    2) URL path suffix
    3) Content-Type mapping via mimetypes
    """
    # 1) Content-Disposition
    fname = _parse_content_disposition_filename(content_disposition)
    if fname:
        suf = Path(fname).suffix.lower()
        if suf:
            return suf

    # 2) URL path suffix
    try:
        u = urlparse(url)
        suf = Path(u.path).suffix.lower()
        if suf:
            return suf
    except Exception:
        pass

    # 3) Content-Type -> extension
    ct = _content_type_main(content_type)
    if ct:
        ext = mimetypes.guess_extension(ct) or ""
        return (ext or "").lower()

    return ""

@dataclass
class URLFetcherConfig:
    # Network
    user_agent: str = "NeurosurferRAG"
    connect_timeout_s: float = 8.0
    read_timeout_s: float = 20.0
    max_redirects: int = 5  # urllib follows redirects; we still sanity-check final URL

    # SSRF protection
    allow_private_network: bool = False

    # Size limits
    max_html_bytes: int = 10_000_000      # 10 MB
    max_file_bytes: int = 25_000_000     # 25 MB
    max_probe_bytes: int = 64_000        # for fallback probes when HEAD fails

    # Caching
    enable_cache: bool = True
    cache_dir: str = "./tmp/url_cache"

    # HTML extraction
    # If bs4 is installed it will be used; otherwise fallback parser.
    collapse_whitespace: bool = True


@dataclass
class URLFetchResult:
    url: str
    final_url: str
    source_type: str  # "html" | "file" | "unknown"
    content_type: Optional[str]
    bytes_downloaded: int
    cached: bool
    text: Optional[str]
    warning: Optional[str] = None
    error: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


class _FallbackHTMLTextExtractor(HTMLParser):
    """
    Stdlib HTML to text extraction:
    - Skips script/style/noscript/svg/canvas
    - Adds newlines around block-ish elements
    - Tries to ignore nav/header/footer/aside content somewhat by tag
    """

    SKIP_TAGS = {"script", "style", "noscript", "svg", "canvas"}
    NOISE_TAGS = {"nav", "header", "footer", "aside"}
    BLOCK_TAGS = {
        "p", "div", "br", "hr",
        "section", "article", "main",
        "h1", "h2", "h3", "h4", "h5", "h6",
        "ul", "ol", "li",
        "table", "tr", "td", "th",
        "pre", "code", "blockquote",
    }

    def __init__(self):
        super().__init__()
        self._chunks: list[str] = []
        self._skip_stack: list[str] = []
        self._noise_stack: list[str] = []

    def handle_starttag(self, tag: str, attrs):
        t = tag.lower()
        if t in self.SKIP_TAGS:
            self._skip_stack.append(t)
            return
        if t in self.NOISE_TAGS:
            self._noise_stack.append(t)
        if t in self.BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str):
        t = tag.lower()
        if self._skip_stack and self._skip_stack[-1] == t:
            self._skip_stack.pop()
            return
        if self._noise_stack and self._noise_stack[-1] == t:
            self._noise_stack.pop()
        if t in self.BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_data(self, data: str):
        if self._skip_stack:
            return
        if self._noise_stack:
            # drop noisy areas entirely (basic but effective)
            return
        txt = data.strip()
        if not txt:
            return
        self._chunks.append(txt)
        self._chunks.append(" ")

    def text(self) -> str:
        return "".join(self._chunks)


class URLFetcher:
    """
    Production-friendly URL fetcher:
    - SSRF protection
    - HEAD probing (fallback to small GET if needed)
    - HTML extraction -> plain text
    - File download (size-limited) -> FileReader.read -> plain text
    - Optional ETag/Last-Modified caching

    Usage:
        fetcher = URLFetcher(config=URLFetcherConfig(), file_reader=FileReader())
        text = fetcher.fetch_text("https://example.com/page")
    """

    def __init__(self, *, config: Optional[URLFetcherConfig] = None, file_reader=None, logger: Optional[logging.Logger] = None):
        self.config = config or URLFetcherConfig()
        self.log = logger or logging.getLogger(__name__)
        self.file_reader = file_reader  # expects .read(Path) -> str

        self.cache_dir = Path(self.config.cache_dir)
        if self.config.enable_cache:
            _safe_mkdir(self.cache_dir)

    # ---------------------------
    # Public API
    # ---------------------------
    def fetch_text(self, url: str) -> Optional[str]:
        r = self.fetch(url)
        return r.text if r and r.text else None

    def fetch(self, url: str) -> URLFetchResult:
        url_in = url.strip()
        if not _is_http_url(url_in):
            return URLFetchResult(
                url=url_in,
                final_url=url_in,
                source_type="unknown",
                content_type=None,
                bytes_downloaded=0,
                cached=False,
                text=None,
                error="Invalid URL (must be http/https).",
            )

        norm = _normalize_url(url_in)
        ok, err = self._validate_url(norm)
        if not ok:
            return URLFetchResult(
                url=url_in,
                final_url=norm,
                source_type="unknown",
                content_type=None,
                bytes_downloaded=0,
                cached=False,
                text=None,
                error=err,
            )

        # Probe headers (HEAD; fallback to small GET)
        probe = self._probe(norm)
        if probe.get("error"):
            return URLFetchResult(
                url=url_in,
                final_url=probe.get("final_url") or norm,
                source_type="unknown",
                content_type=probe.get("content_type"),
                bytes_downloaded=0,
                cached=False,
                text=None,
                error=probe["error"],
                meta=probe,
            )

        final_url = probe.get("final_url") or norm
        content_type = probe.get("content_type")
        content_len = probe.get("content_length")  # may be None
        cd = probe.get("content_disposition")

        # Re-validate final URL after redirects (SSRF safety)
        ok2, err2 = self._validate_url(final_url)
        if not ok2:
            return URLFetchResult(
                url=url_in,
                final_url=final_url,
                source_type="unknown",
                content_type=content_type,
                bytes_downloaded=0,
                cached=False,
                text=None,
                error=f"Final URL blocked: {err2}",
                meta=probe,
            )

        ct_main = _content_type_main(content_type)

        # Decide HTML vs file
        is_html = (ct_main.startswith("text/html") or ct_main == "application/xhtml+xml" or ct_main == "")
        ext_guess = _guess_ext(final_url, content_type, cd)

        probe_prefix = probe.get("probe_bytes")
        # Strong signals for "this is a PDF/file even if headers are confusing"
        if _looks_like_pdf(probe_prefix) or _is_arxiv_pdf_like(final_url):
            # Force pdf extension if we can’t guess it
            # if not ext_guess:
            ext_guess = ".pdf"
            return self._fetch_as_file(url_in, final_url, content_type, content_len, cd, ext_guess, probe)

        # If content-type clearly indicates a file (pdf/docx/etc), treat as file
        if not is_html:
            return self._fetch_as_file(url_in, final_url, content_type, content_len, cd, ext_guess, probe)

        # If looks like a file by suffix even when content-type is weak, treat as file
        if ext_guess and ext_guess != ".html" and ext_guess != ".htm":
            return self._fetch_as_file(url_in, final_url, content_type, content_len, cd, ext_guess, probe)

        # Otherwise treat as HTML
        return self._fetch_as_html(url_in, final_url, content_type, content_len, probe)

    # ---------------------------
    # Validation (SSRF)
    # ---------------------------
    def _validate_url(self, url: str) -> Tuple[bool, Optional[str]]:
        try:
            u = urlparse(url)
            if u.scheme not in ("http", "https"):
                return False, "Only http/https URLs are allowed."
            if not u.hostname:
                return False, "URL hostname is missing."

            # Block local/private networks unless explicitly allowed
            if not self.config.allow_private_network:
                ips = _resolve_host_ips(u.hostname)
                if not ips:
                    # If we cannot resolve, be conservative
                    return False, f"Could not resolve hostname: {u.hostname}"
                for ip in ips:
                    if _is_private_or_local_ip(ip):
                        return False, f"Blocked private/local address: {u.hostname} -> {ip}"

            return True, None
        except Exception as e:
            return False, f"URL validation failed: {e}"

    # ---------------------------
    # Probing
    # ---------------------------
    def _probe(self, url: str) -> Dict[str, Any]:
        headers = {"User-Agent": self.config.user_agent, "Accept": "*/*"}
        timeout = self.config.connect_timeout_s

        # HEAD
        try:
            req = Request(url, headers=headers, method="HEAD")
            with urlopen(req, timeout=timeout) as resp:
                final_url = resp.geturl()
                h = dict(resp.headers.items())
                return {
                    "final_url": final_url,
                    "status": getattr(resp, "status", None) or 200,
                    "content_type": h.get("Content-Type"),
                    "content_length": self._safe_int(h.get("Content-Length")),
                    "etag": h.get("ETag"),
                    "last_modified": h.get("Last-Modified"),
                    "content_disposition": h.get("Content-Disposition"),
                    "probe_bytes": None,   # <-- add
                }
        except Exception as e:
            # Fallback probe via GET reading small prefix
            try:
                req = Request(
                    url,
                    headers={**headers, "Range": f"bytes=0-{self.config.max_probe_bytes - 1}"},
                    method="GET",
                )
                with urlopen(req, timeout=timeout) as resp:
                    final_url = resp.geturl()
                    h = dict(resp.headers.items())
                    prefix = resp.read(self.config.max_probe_bytes)  # <-- keep it

                    return {
                        "final_url": final_url,
                        "status": getattr(resp, "status", None) or 200,
                        "content_type": h.get("Content-Type"),
                        "content_length": self._safe_int(h.get("Content-Length")),
                        "etag": h.get("ETag"),
                        "last_modified": h.get("Last-Modified"),
                        "content_disposition": h.get("Content-Disposition"),
                        "probe_bytes": prefix,  # <-- add
                        "head_error": str(e),
                    }
            except Exception as e2:
                return {
                    "final_url": url,
                    "status": None,
                    "content_type": None,
                    "content_length": None,
                    "etag": None,
                    "last_modified": None,
                    "content_disposition": None,
                    "probe_bytes": None,   # <-- add
                    "error": f"Probe failed (HEAD+GET): {e2}",
                    "head_error": str(e),
                }

    @staticmethod
    def _safe_int(x: Optional[str]) -> Optional[int]:
        if not x:
            return None
        try:
            return int(x)
        except Exception:
            return None

    # ---------------------------
    # Fetch as HTML
    # ---------------------------
    def _fetch_as_html(
        self,
        url_in: str,
        final_url: str,
        content_type: Optional[str],
        content_len: Optional[int],
        probe_meta: Dict[str, Any],
    ) -> URLFetchResult:
        # Size guard (if Content-Length known)
        if content_len is not None and content_len > self.config.max_html_bytes:
            w = f"HTML too large ({content_len} bytes > {self.config.max_html_bytes}); skipping."
            self.log.warning(w)
            return URLFetchResult(
                url=url_in,
                final_url=final_url,
                source_type="html",
                content_type=content_type,
                bytes_downloaded=0,
                cached=False,
                text=None,
                warning=w,
                meta=probe_meta,
            )

        # Download (optionally cached)
        html_bytes, cached, dl_bytes, warn = self._download_file(
            final_url,
            max_bytes=self.config.max_html_bytes,
            use_cache=False,  # caching raw HTML is often not worth it; keep it simple
            cache_ext=".html",
            probe_meta=probe_meta,
        )
        if html_bytes is None:
            return URLFetchResult(
                url=url_in,
                final_url=final_url,
                source_type="html",
                content_type=content_type,
                bytes_downloaded=dl_bytes,
                cached=cached,
                text=None,
                error=warn or "Failed to download HTML.",
                meta=probe_meta,
            )

        text = self._extract_html_text(html_bytes)
        text = self._postprocess_text(text)
        if not text.strip():
            w2 = "HTML fetched but extracted text is empty."
            self.log.warning(w2)
            return URLFetchResult(
                url=url_in,
                final_url=final_url,
                source_type="html",
                content_type=content_type,
                bytes_downloaded=dl_bytes,
                cached=cached,
                text=None,
                warning=w2,
                meta=probe_meta,
            )

        return URLFetchResult(
            url=url_in,
            final_url=final_url,
            source_type="html",
            content_type=content_type,
            bytes_downloaded=dl_bytes,
            cached=cached,
            text=text,
            warning=warn,
            meta=probe_meta,
        )

    def _extract_html_text(self, html_bytes: bytes) -> str:
        # Try bs4 if installed for better extraction
        try:
            from bs4 import BeautifulSoup  # type: ignore

            soup = BeautifulSoup(html_bytes, "html.parser")

            # Remove obvious junk
            for tag in soup(["script", "style", "noscript", "svg", "canvas"]):
                tag.decompose()

            # Prefer main/article when available
            main = soup.find("main") or soup.find("article") or soup.body or soup
            text = main.get_text(separator="\n", strip=True)
            return text
        except Exception:
            # Fallback stdlib parser
            parser = _FallbackHTMLTextExtractor()
            try:
                html_str = html_bytes.decode("utf-8", errors="ignore")
            except Exception:
                html_str = str(html_bytes)
            parser.feed(html_str)
            return parser.text()

    # ---------------------------
    # Fetch as file -> FileReader.read -> text
    # ---------------------------
    def _fetch_as_file(
        self,
        url_in: str,
        final_url: str,
        content_type: Optional[str],
        content_len: Optional[int],
        content_disposition: Optional[str],
        ext_guess: str,
        probe_meta: Dict[str, Any],
    ) -> URLFetchResult:
        if self.file_reader is None:
            return URLFetchResult(
                url=url_in,
                final_url=final_url,
                source_type="file",
                content_type=content_type,
                bytes_downloaded=0,
                cached=False,
                text=None,
                error="FileReader is not set; cannot parse downloaded files.",
                meta=probe_meta,
            )

        # If size known, enforce file cap
        if content_len is not None and content_len > self.config.max_file_bytes:
            w = f"Remote file too large ({content_len} bytes > {self.config.max_file_bytes}); skipping."
            self.log.warning(w)
            return URLFetchResult(
                url=url_in,
                final_url=final_url,
                source_type="file",
                content_type=content_type,
                bytes_downloaded=0,
                cached=False,
                text=None,
                warning=w,
                meta=probe_meta,
            )

        # Determine extension (ensure we have something reasonable)
        ext = (ext_guess or "").lower()
        if not ext:
            # fallback: if content-type says pdf/docx etc, mimetypes may handle
            ext = _guess_ext(final_url, content_type, content_disposition) or ""
        if not ext:
            # unknown file type, still can try but FileReader might fail
            ext = ".bin"

        # Download to cache (cache is valuable for big files)
        data, cached, dl_bytes, warn = self._download_file(
            url=final_url,
            max_bytes=self.config.max_file_bytes,
            use_cache=self.config.enable_cache,
            cache_ext=ext,
            probe_meta=probe_meta,
        )
        if data is None:
            return URLFetchResult(
                url=url_in,
                final_url=final_url,
                source_type="file",
                content_type=content_type,
                bytes_downloaded=dl_bytes,
                cached=cached,
                text=None,
                error=warn or "Failed to download file.",
                meta=probe_meta,
            )

        # Write to temp file (or cached file path when caching enabled)
        file_path = self._write_cached_bytes(final_url, data, ext) if self.config.enable_cache else self._write_temp_bytes(data, ext)

        # Parse with FileReader
        try:
            text = self.file_reader.read(Path(file_path))
        except Exception as e:
            return URLFetchResult(
                url=url_in,
                final_url=final_url,
                source_type="file",
                content_type=content_type,
                bytes_downloaded=dl_bytes,
                cached=cached,
                text=None,
                error=f"FileReader failed: {e}",
                warning=warn,
                meta=probe_meta,
            )

        text = self._postprocess_text(text)
        if not text.strip():
            w2 = "File downloaded but extracted text is empty."
            self.log.warning(w2)
            return URLFetchResult(
                url=url_in,
                final_url=final_url,
                source_type="file",
                content_type=content_type,
                bytes_downloaded=dl_bytes,
                cached=cached,
                text=None,
                warning=w2,
                meta=probe_meta,
            )

        return URLFetchResult(
            url=url_in,
            final_url=final_url,
            source_type="file",
            content_type=content_type,
            bytes_downloaded=dl_bytes,
            cached=cached,
            text=text,
            warning=warn,
            meta=probe_meta,
        )

    # ---------------------------
    # Download helpers + caching
    # ---------------------------
    def _cache_key(self, url: str) -> str:
        return _sha256(_normalize_url(url))

    def _cache_paths(self, url: str, ext: str) -> Tuple[Path, Path]:
        key = self._cache_key(url)
        data_path = self.cache_dir / f"{key}{ext}"
        meta_path = self.cache_dir / f"{key}.json"
        return data_path, meta_path

    def _load_cache_meta(self, meta_path: Path) -> Dict[str, Any]:
        try:
            if meta_path.is_file():
                return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass
        return {}

    def _save_cache_meta(self, meta_path: Path, meta: Dict[str, Any]) -> None:
        try:
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        except Exception as e:
            self.log.debug(f"Failed to save cache meta: {e}")

    def _download_file(
        self,
        url: str,
        *,
        max_bytes: int,
        use_cache: bool,
        cache_ext: str,
        probe_meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[bytes], bool, int, Optional[str]]:
        """
        Download URL content (bytes) with optional local caching.

        Signature matches your current usage:
            content_bytes, cached, dl_bytes, warn = _download_file(
                final_url,
                max_bytes=...,
                use_cache=...,
                cache_ext="....",
                probe_meta=probe_meta,
            )

        Returns:
            (content_bytes, cached, bytes_downloaded, warning)

        Notes:
        - Handles HTTP 304 correctly.
        - Enforces max_bytes (hard cap). If exceeded, returns (None, False, bytes_so_far, warning).
        - Caching:
            - If use_cache=True: stores bytes to disk and can re-use them later.
            - If server returns 304 and cache exists: reads cached bytes from disk.
        """
        from pathlib import Path
        from urllib.request import Request, urlopen
        from urllib.error import HTTPError, URLError

        probe_meta = probe_meta or {}

        # ----------------------------
        # Cache paths & validators
        # ----------------------------
        cached = False
        warning: Optional[str] = None
        bytes_downloaded = 0

        cache_dir = Path(getattr(self.config, "cache_dir", "./.cache/url_fetcher")).resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)

        # stable-ish filename based on url (or final_url)
        # (if you already have a helper for this, use that)
        import hashlib
        key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]
        cache_path = cache_dir / f"{key}{cache_ext}"
        meta_path = cache_dir / f"{key}.meta.json"

        def _read_cache_meta() -> Dict[str, Any]:
            if not meta_path.exists():
                return {}
            try:
                import json
                return json.loads(meta_path.read_text(encoding="utf-8")) or {}
            except Exception:
                return {}

        def _write_cache_meta(meta: Dict[str, Any]) -> None:
            try:
                import json
                meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            except Exception:
                pass

        cache_meta = _read_cache_meta() if use_cache else {}
        has_cached_file = use_cache and cache_path.exists() and cache_path.is_file()

        cached_etag = cache_meta.get("etag")
        cached_last_modified = cache_meta.get("last_modified")

        # ----------------------------
        # Request headers
        # ----------------------------
        headers = {
            "User-Agent": getattr(self.config, "user_agent", "neurosurfer-url-fetcher/1.0"),
            "Accept": "application/pdf, text/html;q=0.9, */*;q=0.8",
        }

        # Only add conditional headers if we truly have a cached file
        if has_cached_file:
            if cached_etag:
                headers["If-None-Match"] = cached_etag
            if cached_last_modified:
                headers["If-Modified-Since"] = cached_last_modified

        timeout = getattr(self.config, "connect_timeout_s", 20)

        def _safe_int(x: Optional[str]) -> Optional[int]:
            try:
                return int(x) if x is not None else None
            except Exception:
                return None

        def _read_stream_to_bytes(resp) -> Tuple[Optional[bytes], int, Optional[str]]:
            total = 0
            buf = bytearray()
            while True:
                chunk = resp.read(1024 * 1024)  # 1MB
                if not chunk:
                    break
                buf.extend(chunk)
                total += len(chunk)
                if total > max_bytes:
                    return None, total, f"Download exceeded max_bytes={max_bytes}. Aborted."
            return bytes(buf), total, None

        def _read_cached_bytes() -> Tuple[Optional[bytes], int, Optional[str]]:
            try:
                b = cache_path.read_bytes()
                if len(b) > max_bytes:
                    return None, 0, f"Cached file exceeds max_bytes={max_bytes}. Ignoring cache."
                return b, 0, None
            except Exception as e:
                return None, 0, f"Failed to read cached file: {e}"

        def _do_get(hdrs: Dict[str, str]) -> Tuple[Optional[bytes], bool, int, Optional[str]]:
            req = Request(url, headers=hdrs, method="GET")
            with urlopen(req, timeout=timeout) as resp:
                final_url = resp.geturl()
                resp_headers = dict(resp.headers.items())

                # Optional early size check if server provides Content-Length
                content_length = _safe_int(resp_headers.get("Content-Length"))
                if content_length is not None and content_length > max_bytes:
                    return None, False, 0, f"Remote content too large ({content_length} bytes) > max_bytes={max_bytes}. Skipped."

                body, dl, warn = _read_stream_to_bytes(resp)
                if body is None:
                    return None, False, dl, warn

                # If caching enabled, write bytes + validators
                if use_cache:
                    try:
                        cache_path.write_bytes(body)
                        _write_cache_meta({
                            "url": url,
                            "final_url": final_url,
                            "etag": resp_headers.get("ETag"),
                            "last_modified": resp_headers.get("Last-Modified"),
                            "content_type": resp_headers.get("Content-Type"),
                            "content_length": content_length,
                        })
                    except Exception:
                        # caching failure shouldn't fail download
                        pass

                # Let caller update probe_meta if they want
                probe_meta.setdefault("final_url", final_url)
                probe_meta.setdefault("content_type", resp_headers.get("Content-Type"))
                probe_meta.setdefault("content_length", content_length)
                probe_meta.setdefault("etag", resp_headers.get("ETag"))
                probe_meta.setdefault("last_modified", resp_headers.get("Last-Modified"))

                return body, False, dl, None

        # ----------------------------
        # Download attempt
        # ----------------------------
        try:
            body, cached, bytes_downloaded, warning = _do_get(headers)
            return body, cached, bytes_downloaded, warning

        except HTTPError as e:
            # urllib raises for 304 too
            if e.code == 304:
                # If we have cached bytes, use them
                if has_cached_file:
                    body, _, warn = _read_cached_bytes()
                    if body is not None:
                        return body, True, 0, warn
                    # cache exists but unreadable -> fall through to unconditional retry

                # 304 but no usable cache -> unconditional retry
                retry_headers = dict(headers)
                retry_headers.pop("If-None-Match", None)
                retry_headers.pop("If-Modified-Since", None)
                retry_headers["Cache-Control"] = "no-cache"
                try:
                    body, cached, bytes_downloaded, warning = _do_get(retry_headers)
                    return body, cached, bytes_downloaded, warning
                except Exception as e2:
                    return None, False, 0, f"HTTP 304 received but no usable cache; retry failed: {e2}"

            # Other HTTP error
            return None, False, 0, f"HTTP error: {e.code} {getattr(e, 'reason', '')}".strip()

        except URLError as e:
            return None, False, 0, f"URL error: {e}"

        except Exception as e:
            return None, False, 0, f"Unexpected error: {e}"



    def _write_temp_bytes(self, data: bytes, ext: str) -> str:
        # temp file in system temp dir
        fd, path = tempfile.mkstemp(prefix="rag_url_", suffix=ext)
        os.close(fd)
        Path(path).write_bytes(data)
        return path

    def _write_cached_bytes(self, url: str, data: bytes, ext: str) -> str:
        # ensure cached location exists and write; return path
        _safe_mkdir(self.cache_dir)
        data_path, meta_path = self._cache_paths(url, ext)
        try:
            data_path.write_bytes(data)
            # minimal meta update (optional)
            meta = self._load_cache_meta(meta_path)
            meta.setdefault("url", url)
            meta["saved_at"] = _now_ts()
            self._save_cache_meta(meta_path, meta)
        except Exception as e:
            self.log.debug(f"Failed to write cached bytes: {e}")
            return self._write_temp_bytes(data, ext)
        return str(data_path)

    # ---------------------------
    # Text post-processing
    # ---------------------------
    def _postprocess_text(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        if self.config.collapse_whitespace:
            # Keep paragraph breaks while collapsing noisy spacing
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            # collapse >2 newlines to 2
            text = re.sub(r"\n{3,}", "\n\n", text)
            # collapse spaces/tabs
            text = re.sub(r"[ \t]{2,}", " ", text)
            # strip each line
            text = "\n".join(line.strip() for line in text.splitlines())
            text = text.strip()
        return text


def _looks_like_pdf(prefix: Optional[bytes]) -> bool:
    return bool(prefix) and prefix.lstrip().startswith(b"%PDF-")

def _looks_like_html(prefix: Optional[bytes]) -> bool:
    if not prefix:
        return False
    p = prefix.lstrip().lower()
    return p.startswith(b"<!doctype html") or p.startswith(b"<html") or b"<html" in p[:5000]

def _is_arxiv_pdf_like(final_url: str) -> bool:
    try:
        u = urlparse(final_url)
        host = (u.hostname or "").lower()
        path = (u.path or "")
        # arXiv PDF endpoints commonly look like /pdf/1706.03762 or /pdf/1706.03762.pdf
        return host.endswith("arxiv.org") and path.startswith("/pdf/")
    except Exception:
        return False
