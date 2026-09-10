"""Backward-compatible search imports."""

from .retrieval.search import (
    MockSearchProvider,
    SearchProvider,
    TavilySearchProvider,
    deduplicate_queries,
    deduplicate_results,
)

__all__ = [
    "MockSearchProvider", "SearchProvider", "TavilySearchProvider",
    "deduplicate_queries", "deduplicate_results",
]
