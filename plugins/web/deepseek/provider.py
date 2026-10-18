"""DeepSeek Search + Fetch (``/v1/tools/{search,fetch}``; sync httpx).

DeepSeek runs its own index here — this is not a proxy in front of Google/Bing. Two properties
shape this provider:

* **Billing is per successful request**, not per token: ¥5 per 1,000 calls for each endpoint,
  and a call that errors or returns no results is free. So a thin wrapper is correct; there is
  nothing to batch or cache locally to save money.
* **The vendor already ranked and graded the results** — each hit carries ``rerank_score``
  (0–3, ≥1.5 = high quality) and ``authority`` (0–4, ≥3 = highly authoritative). We keep the
  vendor's order and hand the model its long ``summary`` (written to be read by a model,
  measured 166–3797 chars) instead of a raw snippet.

Auth reuses the model key ``DEEPSEEK_API_KEY`` — DeepSeek issues no separate search key, so a
working chat key already pays for search.

Rate limit: 5 QPS per account, per endpoint (a 429 means exactly that, not a quota error).
Docs: https://api-docs.deepseek.com/zh-cn/guides/search/
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import httpx

from plugins.web._common import (
    SEARCH_LIMIT_CAP,
    BaseWebSearchProvider,
    document,
    extract_fail,
    http_status_detail,
    page_error,
    provider_env,
    run_extract,
    run_search,
    search_fail,
    search_ok,
    setup_schema,
    title_hit,
)

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.deepseek.com"

# The API rejects queries longer than 128 Unicode characters outright (400), and search intent
# never needs more than that; truncating keeps a long agent query working instead of failing.
_QUERY_LIMIT = 128

# One hit's summary can be ~3.8k chars on its own; capped so five hits cannot swallow a turn.
_DESCRIPTION_LIMIT = 1200


def _deepseek_request(endpoint: str, payload: dict[str, Any], *, api_key: str) -> dict[str, Any]:
    """POST to ``/v1/tools/<endpoint>`` and return parsed JSON. Non-2xx raises ``ValueError``
    with the body so DeepSeek's own message (402 top-up / 429 QPS / 403 fetch-policy) reaches
    the model verbatim instead of being flattened into "request failed"."""
    base_url = provider_env("DEEPSEEK_BASE_URL") or _DEFAULT_BASE_URL
    url = f"{base_url.rstrip('/')}/v1/tools/{endpoint}"
    logger.info("DeepSeek %s request to %s", endpoint, url)
    response = httpx.post(
        url,
        json=payload,
        timeout=60,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    if response.status_code >= 400:
        raise ValueError(http_status_detail(response))
    return response.json()


def _describe(result: dict[str, Any]) -> str:
    """Model-facing blurb for one hit: the long summary, falling back to the snippet."""
    text = (result.get("summary") or "").strip() or (result.get("snippet") or "").strip()
    return text[:_DESCRIPTION_LIMIT]


def _normalize_search(raw: dict[str, Any]) -> dict[str, Any]:
    return search_ok([
        title_hit(
            r.get("title", ""),
            r.get("url", ""),
            _describe(r),
            i + 1,
        )
        for i, r in enumerate(raw.get("results") or [])
    ])


class DeepSeekWebSearchProvider(BaseWebSearchProvider):
    """DeepSeek search + fetch provider (keyed only — the model key doubles as the search key)."""

    NAME = "deepseek"
    DISPLAY_NAME = "DeepSeek"
    KEY_ENV = "DEEPSEEK_API_KEY"
    EXTRACT = True

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        def _body() -> Dict[str, Any]:
            api_key = provider_env(self.KEY_ENV)
            if not api_key:
                return search_fail("DEEPSEEK_API_KEY is not set — DeepSeek Search uses the platform key")
            logger.info("DeepSeek search: '%s' (limit=%d)", query, limit)
            payload = {"query": query[:_QUERY_LIMIT], "count": min(limit, SEARCH_LIMIT_CAP)}
            return _normalize_search(_deepseek_request("search", payload, api_key=api_key))

        return run_search("DeepSeek", logger, _body)

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        def _body() -> List[Dict[str, Any]]:
            api_key = provider_env(self.KEY_ENV)
            if not api_key:
                return extract_fail(urls, "DEEPSEEK_API_KEY is not set — DeepSeek Fetch uses the platform key")
            logger.info("DeepSeek fetch: %d URL(s)", len(urls))
            documents: List[Dict[str, Any]] = []
            for url in urls:
                try:
                    raw = _deepseek_request("fetch", {"url": url}, api_key=api_key)
                    documents.append(document(raw.get("url") or url, raw.get("title", ""), raw.get("content", "")))
                except Exception as exc:  # 403 policy / 404 no extractable body are per-URL, not per-batch
                    documents.append(page_error(url, str(exc)[:300]))
            return documents

        return run_extract("DeepSeek", logger, urls, _body)

    def get_setup_schema(self) -> Dict[str, Any]:
        return setup_schema(
            "DeepSeek",
            "paid · per-request",
            "Search + fetch on DeepSeek's own index (¥5 per 1,000 successful calls each).",
            "DEEPSEEK_API_KEY",
            "DeepSeek API key (the model key works — no separate search key exists)",
            "https://platform.deepseek.com/api_keys",
        )
