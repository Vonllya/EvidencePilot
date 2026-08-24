from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import socket
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup


def is_safe_url(url: str, resolve_dns: bool = True) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    hostname = parsed.hostname.casefold().rstrip(".")
    if hostname in {"localhost", "metadata.google.internal"} or hostname.endswith(".localhost"):
        return False
    try:
        addresses = [ipaddress.ip_address(hostname)]
    except ValueError:
        if not resolve_dns:
            return True
        try:
            addresses = [ipaddress.ip_address(item[4][0]) for item in socket.getaddrinfo(hostname, parsed.port or 443)]
        except (socket.gaierror, ValueError):
            return False
    return all(ip.is_global for ip in addresses)


def extract_main_text(html: str, max_chars: int = 20_000) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "svg"]):
        tag.decompose()
    root = soup.find("main") or soup.find("article") or soup.body or soup
    return "\n".join(line for line in (s.strip() for s in root.get_text("\n").splitlines()) if line)[:max_chars]


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
        headers = {"User-Agent": "EvidencePilot/0.1 (+https://github.com/example/evidencepilot)"}
        current_url = url
        async with httpx.AsyncClient(
            timeout=self.timeout, follow_redirects=False, headers=headers, transport=self.transport
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
                        current_url = next_url
                        continue
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "text/html").split(";", 1)[0]
                    if content_type not in {"text/html", "text/plain", "application/xhtml+xml"}:
                        raise ValueError(f"unsupported content type: {content_type}")
                    if int(response.headers.get("content-length", "0")) > self.max_bytes:
                        raise ValueError("response exceeds size limit")
                    chunks, size = [], 0
                    async for chunk in response.aiter_bytes():
                        size += len(chunk)
                        if size > self.max_bytes:
                            raise ValueError("response exceeds size limit")
                        chunks.append(chunk)
                    content = extract_main_text(
                        b"".join(chunks).decode(response.encoding or "utf-8", errors="replace"),
                        self.max_chars,
                    )
                    return FetchedPage(
                        content=content, final_url=str(response.url),
                        http_status=response.status_code, content_type=content_type,
                        retrieved_at=datetime.now(UTC).isoformat(),
                        content_hash=hashlib.sha256(content.encode()).hexdigest(),
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
