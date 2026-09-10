"""Backward-compatible retrieval imports."""

from .retrieval.fetch import (
    FetchedPage,
    WebFetcher,
    extract_main_text,
    extract_pdf_text,
    is_safe_url,
)

__all__ = ["FetchedPage", "WebFetcher", "extract_main_text", "extract_pdf_text", "is_safe_url"]
