from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import socket
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader


def is_safe_url(url: str, resolve_dns: bool = True) -> bool:
    parsed = urlparse(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        return False
    hostname = parsed.hostname.casefold().rstrip(".")
    blocked_suffixes = (".localhost", ".local", ".internal", ".home.arpa")
    if hostname in {"localhost", "metadata.google.internal"} or hostname.endswith(blocked_suffixes):
        return False
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError:
        return False
    if port not in {80, 443}:
        return False
    try:
        addresses = [ipaddress.ip_address(hostname)]
    except ValueError:
        if not resolve_dns:
            return True
        try:
            addresses = [
                ipaddress.ip_address(item[4][0])
                for item in socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
            ]
        except (socket.gaierror, UnicodeError, ValueError):
            return False
    return bool(addresses) and all(ip.is_global for ip in addresses)


def extract_main_text(html: str, max_chars: int = 20_000) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "svg"]):
        tag.decompose()
    root = soup.find("main") or soup.find("article") or soup.body or soup
    return "\n".join(line for line in (s.strip() for s in root.get_text("\n").splitlines()) if line)[:max_chars]


def extract_pdf_text(content: bytes, max_chars: int = 20_000) -> str:
    """Extract bounded text from a PDF without executing embedded content."""
    reader = PdfReader(BytesIO(content), strict=False)
    pages: list[str] = []
    size = 0
    for page in reader.pages:
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        remaining = max_chars - size
        if remaining <= 0:
            break
        pages.append(text[:remaining])
        size += len(pages[-1])
    return "\n\n".join(pages)[:max_chars]


class WebFetcher:
    def __init__(
        self, timeout: float = 12, max_bytes: int = 2_000_000,
        max_chars: int = 20_000, max_redirects: int = 5,
        transport: httpx.AsyncBaseTransport | None = None,
        max_concurrency: int = 8,
        per_domain_concurrency: int = 2,
    ) -> None:
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.max_chars = max_chars
        self.max_redirects = max_redirects
        self.transport = transport
        self._global_limit = asyncio.Semaphore(max_concurrency)
        self._per_domain_concurrency = per_domain_concurrency
        self._domain_limits: dict[str, asyncio.Semaphore] = {}
        self._cache: dict[str, FetchedPage] = {}

    async def fetch(self, url: str) -> str:
        return (await self.fetch_page(url)).content

    async def fetch_page(self, url: str) -> FetchedPage:
        if url in self._cache:
            return self._cache[url]
        hostname = urlparse(url).hostname or ""
        domain_limit = self._domain_limits.setdefault(
            hostname, asyncio.Semaphore(self._per_domain_concurrency)
        )
        async with self._global_limit, domain_limit:
            page = await self._fetch_page(url)
        self._cache[url] = page
        return page

    async def _fetch_page(self, url: str) -> FetchedPage:
        headers = {"User-Agent": "EvidencePilot/0.1 (+https://github.com/Vonllya/EvidencePilot)"}
        current_url = url
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout), follow_redirects=False, headers=headers,
            transport=self.transport, trust_env=False,
        ) as client:
            for redirect_count in range(self.max_redirects + 1):
                if not is_safe_url(current_url):
                    raise ValueError("unsafe or unresolvable URL")
                async with client.stream("GET", current_url) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location:
                            raise ValueError("redirect response has no Location header")
                        if redirect_count >= self.max_redirects:
                            raise ValueError("too many redirects")
                        next_url = urljoin(str(response.url), location)
                        if not is_safe_url(next_url):
                            raise ValueError("redirect target is unsafe or unresolvable")
                        if urlparse(current_url).scheme == "https" and urlparse(next_url).scheme != "https":
                            raise ValueError("HTTPS redirect downgrade is not allowed")
                        current_url = next_url
                        continue
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().casefold()
                    if content_type not in {
                        "text/html", "text/plain", "application/xhtml+xml", "application/pdf"
                    }:
                        raise ValueError(f"unsupported content type: {content_type or 'missing'}")
                    declared_length = response.headers.get("content-length")
                    if declared_length is not None:
                        try:
                            if int(declared_length) < 0 or int(declared_length) > self.max_bytes:
                                raise ValueError("response exceeds size limit")
                        except ValueError as exc:
                            if str(exc) == "response exceeds size limit":
                                raise
                            raise ValueError("invalid Content-Length header") from exc
                    chunks, size = [], 0
                    async for chunk in response.aiter_bytes():
                        size += len(chunk)
                        if size > self.max_bytes:
                            raise ValueError("response exceeds size limit")
                        chunks.append(chunk)
                    raw_content = b"".join(chunks)
                    parser = "pypdf" if content_type == "application/pdf" else "beautifulsoup"
                    content = (
                        extract_pdf_text(raw_content, self.max_chars)
                        if content_type == "application/pdf"
                        else extract_main_text(
                            raw_content.decode(response.encoding or "utf-8", errors="replace"),
                            self.max_chars,
                        )
                    )
                    if not content.strip():
                        raise ValueError(f"parsed {content_type} body was empty")
                    return FetchedPage(
                        content=content, final_url=str(response.url),
                        http_status=response.status_code, content_type=content_type,
                        retrieved_at=datetime.now(UTC).isoformat(),
                        content_hash=hashlib.sha256(content.encode()).hexdigest(),
                        parser=parser,
                    )
        raise ValueError("too many redirects")


@dataclass(frozen=True, slots=True)
class FetchedPage:
    content: str
    final_url: str
    http_status: int
    content_type: str
    retrieved_at: str
    content_hash: str
    parser: str = "beautifulsoup"
