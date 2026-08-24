import httpx
import pytest

from evidencepilot.fetcher import WebFetcher, extract_main_text, is_safe_url


def test_url_safety():
    for url in ["http://127.0.0.1/x", "http://10.0.0.2", "http://169.254.169.254/latest", "file:///etc/passwd", "http://localhost"]:
        assert not is_safe_url(url, resolve_dns=False)
    assert is_safe_url("https://example.com/path", resolve_dns=False)


def test_extract_main_text_removes_noise():
    html = "<html><nav>menu</nav><main><h1>Title</h1><p>Useful evidence.</p><script>bad()</script></main><footer>legal</footer></html>"
    text = extract_main_text(html)
    assert "Title" in text and "Useful evidence" in text
    assert "menu" not in text and "bad()" not in text and "legal" not in text


@pytest.mark.asyncio
async def test_fetcher_revalidates_and_blocks_unsafe_redirect(monkeypatch):
    calls = []

    def handler(request: httpx.Request):
        calls.append(str(request.url))
        return httpx.Response(302, headers={"Location": "http://127.0.0.1/private"})

    monkeypatch.setattr(
        "evidencepilot.fetcher.is_safe_url", lambda url: "127.0.0.1" not in url
    )
    fetcher = WebFetcher(transport=httpx.MockTransport(handler))
    with pytest.raises(ValueError, match="redirect target is unsafe"):
        await fetcher.fetch("https://example.com/start")
    assert calls == ["https://example.com/start"]


@pytest.mark.asyncio
async def test_fetcher_allows_safe_redirect_and_limits_hops(monkeypatch):
    def handler(request: httpx.Request):
        if request.url.path == "/start":
            return httpx.Response(302, headers={"Location": "/final"})
        return httpx.Response(200, text="<main>Public evidence.</main>")

    monkeypatch.setattr("evidencepilot.fetcher.is_safe_url", lambda url: True)
    fetcher = WebFetcher(transport=httpx.MockTransport(handler), max_redirects=1)
    assert await fetcher.fetch("https://public.example/start") == "Public evidence."


@pytest.mark.asyncio
async def test_fetch_page_records_metadata_and_uses_cache(monkeypatch):
    calls = 0

    def handler(request: httpx.Request):
        nonlocal calls
        calls += 1
        return httpx.Response(
            200, text="<main>Cached evidence.</main>",
            headers={"Content-Type": "text/html; charset=utf-8"},
        )

    monkeypatch.setattr("evidencepilot.fetcher.is_safe_url", lambda url: True)
    fetcher = WebFetcher(transport=httpx.MockTransport(handler))
    first = await fetcher.fetch_page("https://public.example/page")
    second = await fetcher.fetch_page("https://public.example/page")
    assert first == second and calls == 1
    assert first.http_status == 200 and first.content_type == "text/html"
    assert len(first.content_hash) == 64 and first.parser == "beautifulsoup"


@pytest.mark.asyncio
async def test_fetcher_rejects_unsupported_content_type(monkeypatch):
    monkeypatch.setattr("evidencepilot.fetcher.is_safe_url", lambda url: True)
    fetcher = WebFetcher(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, content=b"binary", headers={"Content-Type": "image/png"})
    ))
    with pytest.raises(ValueError, match="unsupported content type"):
        await fetcher.fetch("https://public.example/image")
