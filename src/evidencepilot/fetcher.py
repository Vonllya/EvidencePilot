from __future__ import annotations

import ipaddress
import socket
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
    ) -> None:
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.max_chars = max_chars
        self.max_redirects = max_redirects
        self.transport = transport

    async def fetch(self, url: str) -> str:
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
                    if int(response.headers.get("content-length", "0")) > self.max_bytes:
                        raise ValueError("response exceeds size limit")
                    chunks, size = [], 0
                    async for chunk in response.aiter_bytes():
                        size += len(chunk)
                        if size > self.max_bytes:
                            raise ValueError("response exceeds size limit")
                        chunks.append(chunk)
                    return extract_main_text(
                        b"".join(chunks).decode(response.encoding or "utf-8", errors="replace"),
                        self.max_chars,
                    )
        raise ValueError("too many redirects")
