from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None  # checked at runtime when used

from urllib.parse import urlparse


def get_domain(url: str) -> str:
    """
    Extract normalized domain from URL (netloc without leading 'www.').
    """
    try:
        netloc = urlparse(url).netloc.lower()
    except Exception:
        return ""
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc


def _lookup_domain_config(
    url: str,
    domain_content_config: Mapping[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Return the best-matching domain configuration for a URL, if any.

    Matching is done by:
      1) exact domain match
      2) suffix match: domain.endswith(config_key)
    """
    domain = get_domain(url)
    if not domain:
        return None

    if domain in domain_content_config:
        return domain_content_config[domain]

    for key, cfg in domain_content_config.items():
        if domain.endswith(key):
            return cfg

    return None


def html_to_text(
    html: str,
    url: Optional[str],
    domain_content_config: Mapping[str, Dict[str, Any]],
) -> str:
    """
    Convert HTML to cleaned plain text using BeautifulSoup.

    Uses domain-specific selectors from `domain_content_config` when possible,
    falling back to generic extraction otherwise.
    """
    if BeautifulSoup is None:
        raise RuntimeError(
            "BeautifulSoup (beautifulsoup4) is required for HTML parsing. "
            "Install it with `pip install beautifulsoup4`."
        )

    soup = BeautifulSoup(html, "html.parser")

    # Domain-specific handling, if available
    domain_cfg = _lookup_domain_config(url or "", domain_content_config)
    if domain_cfg:
        prefer_selectors = domain_cfg.get("prefer", []) or []
        exclude_selectors = domain_cfg.get("exclude", []) or []

        # Prefer specific content containers
        for selector in prefer_selectors:
            node = soup.select_one(selector)
            if node is not None:
                soup = node
                break

        # Exclude noisy regions
        for selector in exclude_selectors:
            for tag in soup.select(selector):
                tag.decompose()

    # Generic stripping of scripts/styles/etc.
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)
    text = " ".join(text.split())
    return text
