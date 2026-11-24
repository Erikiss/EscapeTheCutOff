"""External event sources (Wikidata, Wikipedia) and the polite HTTP helper they share.

Wikimedia API etiquette, applied by every fetcher here: a descriptive User-Agent with a project URL and no personal
data (:data:`cutoffmix.bench.sources.wikidata.USER_AGENT`), one request at a time, ``Retry-After`` honoured on
429/503, exponential back-off on transient failures, and ``maxlag`` on MediaWiki API calls.
"""

from __future__ import annotations

import email.utils
import logging
import time
from datetime import datetime, timezone
from typing import Any

import httpx

log = logging.getLogger(__name__)

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
TIMEOUT_STATUSES = frozenset({502, 504})


class SourceError(RuntimeError):
    """A request to an external source failed for good.

    ``timeout`` is True when the failure looks like a server-side query timeout (WDQS answers those with 500 and a
    ``TimeoutException`` body, with 502/504 from its proxy, or with a 200 whose JSON body is cut off), which callers
    handle by narrowing the request rather than repeating it.
    """

    def __init__(self, message: str, *, status: int | None = None, timeout: bool = False):
        super().__init__(message)
        self.status = status
        self.timeout = timeout


def pause(seconds: float) -> None:
    """Wait ``seconds``. Every source module waits through this function (looked up on this module at call time),
    so tests can replace it with a no-op."""
    time.sleep(seconds)


def retry_after_seconds(value: str | None) -> float | None:
    """Parse a ``Retry-After`` header (delta-seconds or HTTP-date); None if absent or unparseable."""
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


def backoff_delay(attempt: int, retry_after: float | None, *, base: float = 2.0, cap: float = 120.0) -> float:
    """Seconds to wait before retry number ``attempt + 1``: ``Retry-After`` if given, else ``base * 2**attempt``."""
    delay = retry_after if retry_after is not None else base * (2**attempt)
    return min(cap, delay)


def polite_request(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    retries: int = 4,
    base_delay: float = 2.0,
    max_delay: float = 120.0,
    **kwargs: Any,
) -> httpx.Response:
    """Send one request, retrying transient failures; returns the first response with status < 400.

    Retries (at most ``retries`` times) on 429/500/502/503/504, on timeouts and on transport errors, waiting
    ``Retry-After`` when the server sends it and exponential back-off otherwise. Other 4xx/5xx statuses raise
    :class:`SourceError` at once. After the last retry the last failure is raised as :class:`SourceError`.
    """
    attempt = 0
    while True:
        retry_after: float | None = None
        try:
            resp = client.request(method, url, **kwargs)
        except httpx.TimeoutException as e:
            err = SourceError(f"{method} {url}: timed out ({type(e).__name__})", timeout=True)
        except httpx.TransportError as e:
            err = SourceError(f"{method} {url}: {type(e).__name__}: {e}")
        else:
            status = resp.status_code
            if status < 400:
                return resp
            body = resp.text[:300].replace("\n", " ")
            timeout = status in TIMEOUT_STATUSES or (status == 500 and "TimeoutException" in resp.text)
            err = SourceError(f"{method} {url}: HTTP {status}: {body}", status=status, timeout=timeout)
            if status not in RETRY_STATUSES:
                raise err
            retry_after = retry_after_seconds(resp.headers.get("Retry-After"))
        if attempt >= retries:
            raise err
        delay = backoff_delay(attempt, retry_after, base=base_delay, cap=max_delay)
        log.info("%s; retry %d/%d in %.1fs", err, attempt + 1, retries, delay)
        pause(delay)
        attempt += 1
