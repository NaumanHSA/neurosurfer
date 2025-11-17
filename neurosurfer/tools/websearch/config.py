from __future__ import annotations
from typing import Any, Dict, Optional, Sequence
from pydantic import BaseModel
from typing import Literal


USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"

# Content limit strategy.
# Applicable when content_words_limit is set and the crawled content exceeds the limit.
LimitStrategy = Literal[
    "first",            # Keep the first N words
    "last",             # Keep the last N words
    "middle",           # Keep a centered window of N words
    "head_tail",        # Split budget between start and end, join with "...".
    "distributive"      # Pick several short chunks distributed across the text, joined with "...".
]


class WebSearchConfig(BaseModel):
    # Engine selection
    engine: str = "serpapi"                                  # Which backend to use (e.g. SerpAPI)
    engine_kwargs: Optional[Dict[str, Any]] = None            # Engine-specific config (api_key, endpoint, ...)

    # Generic knobs
    content_words_limit: Optional[int] = 1500                # Effective per-call limit for crawled page text length (words); -1 means no limit
    content_limit_strategy: LimitStrategy = "distributive"   # Strategy to use when limiting content length, must be one of ["first", "last", "middle", "head_tail", "distributive"]
    max_results: int = 10                                    # Maximum number of SERP results to return
    include_raw: bool = False                                # Whether to include raw SERP results in the output

    # SERP-level knobs  
    location: Optional[str] = None                           # Location for SERP results
    gl: str = "us"                                           # gl parameter for SERP results

    # Crawling configuration
    enable_crawl: bool = True                                # Whether to enable crawling of SERP results
    max_crawl_results: Optional[int] = None                  # Maximum number of SERP results to crawl
    crawl_timeout: int = 10                                  # Timeout for each crawl request
    max_concurrent_crawls: int = 4                           # Maximum number of concurrent crawl requests
    allowed_content_types: Optional[Sequence[str]] = [
        "text/html", "text/plain", "application/json", 
        "application/xml", "application/xhtml+xml"
    ]                                                        # Allowed content types for crawled pages
    user_agent: Optional[str] = USER_AGENT                   # User agent to use for crawl requests

    # Domain + extraction configuration
    domain_content_config: Optional[Dict[str, Dict[str, Any]]] = {}      # Domain content configuration
    preferred_domains: Optional[Sequence[str]] = None        # Preferred domains to crawl
    
    # LLM configuration
    summarize: bool = False                                  # Whether to summarize the results with an LLM
    stream_summary: bool = False                             # Whether to stream the LLM summary internally

    def __init__(self, **kwargs):
        super().__init__(**kwargs)



# --------------------------------------------------------------------------- #
# Built-in domain content configuration
# --------------------------------------------------------------------------- #
# Keys are domain suffixes; matching is done via "endswith" on netloc
# (e.g. "en.wikipedia.org" will match "wikipedia.org").
# Each entry can define:
#   - "prefer": list of CSS selectors to prefer
#   - "exclude": list of CSS selectors to remove
# --------------------------------------------------------------------------- #
DOMAIN_CONTENT_CONFIG_DEFAULT: Dict[str, Dict[str, Any]] = {
    # === Encyclopedic / docs style ===
    "wikipedia.org": {
        "prefer": ["div#bodyContent", "div#mw-content-text"],
        "exclude": [
            "table.infobox",
            "div#toc",
            "div#mw-navigation",
            "div#footer",
            "div.mw-panel",
        ],
    },
    "simple.wikipedia.org": {
        "prefer": ["div#bodyContent", "div#mw-content-text"],
        "exclude": ["table.infobox", "div#toc"],
    },
    "docs.python.org": {
        "prefer": ["div#content", "div.body"],
        "exclude": ["div.related", "div.sphinxsidebar"],
    },
    "readthedocs.io": {
        "prefer": ["div.wy-nav-content", "div.document"],
        "exclude": ["nav", "footer", "div.wy-side-scroll", "div.toctree-wrapper"],
    },
    "developer.mozilla.org": {
        "prefer": ["main#content", "article"],
        "exclude": ["nav", "header", "footer", "aside"],
    },

    # === Blogs / articles ===
    "medium.com": {
        "prefer": ["article"],
        "exclude": ["header", "footer", "aside", "nav"],
    },
    "towardsdatascience.com": {
        "prefer": ["article"],
        "exclude": ["header", "footer", "aside", "nav"],
    },
    "dev.to": {
        "prefer": ["div.crayons-article__main"],
        "exclude": ["header", "footer", "aside", "nav"],
    },
    "substack.com": {
        "prefer": ["main", "article"],
        "exclude": ["header", "footer", "nav"],
    },

    # === Q&A / forums ===
    "stackoverflow.com": {
        "prefer": ["div#question", "div.answer"],
        "exclude": ["div#sidebar", "div#hot-network-questions", "header", "footer"],
    },
    "superuser.com": {
        "prefer": ["div#question", "div.answer"],
        "exclude": ["div#sidebar", "header", "footer"],
    },
    "serverfault.com": {
        "prefer": ["div#question", "div.answer"],
        "exclude": ["div#sidebar", "header", "footer"],
    },
    "reddit.com": {
        "prefer": ["div[data-test-id='post-content']", "div.Post"],
        "exclude": ["header", "footer", "nav", "aside"],
    },

    # === Code / repos ===
    "github.com": {
        "prefer": ["article.markdown-body", "div#readme"],
        "exclude": ["header", "footer", "nav", "aside"],
    },
    "gitlab.com": {
        "prefer": ["div.file-content", "article.md"],
        "exclude": ["header", "footer", "nav", "aside"],
    },
    "pypi.org": {
        "prefer": ["div.project-description"],
        "exclude": ["header", "footer", "nav", "aside"],
    },

    # === Papers / scientific ===
    "arxiv.org": {
        "prefer": ["blockquote.abstract", "div#content"],
        "exclude": ["header", "footer", "nav"],
    },
    "acm.org": {
        "prefer": ["div.abstractSection", "section.abstract"],
        "exclude": ["header", "footer", "nav", "aside"],
    },
    "springer.com": {
        "prefer": ["section#Abs1", "div#Abs1-content", "section.Abstract"],
        "exclude": ["header", "footer", "nav", "aside"],
    },
    "nature.com": {
        "prefer": ["div#content", "section.c-article-body"],
        "exclude": ["header", "footer", "nav", "aside"],
    },

    # === News / magazines ===
    "bbc.com": {
        "prefer": ["main", "article"],
        "exclude": ["header", "footer", "nav", "aside"],
    },
    "nytimes.com": {
        "prefer": ["section[name='articleBody']", "main", "article"],
        "exclude": ["header", "footer", "nav", "aside"],
    },
    "theguardian.com": {
        "prefer": ["div.article-body-viewer-selector", "main", "article"],
        "exclude": ["header", "footer", "nav", "aside"],
    },
}

