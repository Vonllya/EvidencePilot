"""Search and document retrieval public API."""

from .fetch import FetchedPage, WebFetcher, extract_main_text, extract_pdf_text, is_safe_url
from .search import (
    MockSearchProvider,
    SearchProvider,
    TavilySearchProvider,
    deduplicate_queries,
    deduplicate_results,
)

__all__ = [
    "FetchedPage", "MockSearchProvider", "SearchProvider", "TavilySearchProvider",
    "WebFetcher", "deduplicate_queries", "deduplicate_results", "extract_main_text",
    "extract_pdf_text", "is_safe_url",
]
