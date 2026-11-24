"""OpenRouter chat-completions backend: streamed relay segments and non-streaming chat calls (DESIGN.md §3.3).

Stream facts this module relies on (OpenRouter docs, checked 2026-11-24):

- SSE framing: ``data: {json}`` lines, blank line between events, ``data: [DONE]`` at the end, and comment lines
  such as ``: OPENROUTER PROCESSING`` that must be skipped.
- ``choices[0].delta`` carries ``content`` (visible text), ``reasoning`` (summary text) and ``refusal``.
- The last chunk before ``[DONE]`` carries ``usage``; it has one choice with an empty delta that *repeats*
  ``finish_reason``, so it is accounting, not a second terminal event.
- After the HTTP 200 is committed, errors arrive as a chunk with a top-level ``error`` and ``finish_reason: "error"``.
- Closing the connection stops generation (and billing) on Anthropic/OpenAI/Azure, not on Google.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import datetime as _dt
import email.utils
import json
import os
import random
import time
from collections.abc import AsyncGenerator, AsyncIterator, Iterable, Iterator, Mapping
from typing import Any

import httpx

from cutoffmix.backends.base import BackendError, continuation_max_tokens, is_retryable_status
from cutoffmix.prompts import answer_system_prompt, answer_user_blocks, relay_system_prompt, relay_user_blocks
from cutoffmix.types import ChatRequest, ChatResult, ContinuationRequest, ModelSpec, StreamEvent

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
USER_AGENT = "cutoffmix/0.1"
CUT_FACTOR = 1.5  # the stream is closed once visible text exceeds hard_cut_chars * CUT_FACTOR
MAX_RETRY_AFTER_S = 60.0
_CONNECT_TIMEOUT_S = 10.0


# --------------------------------------------------------------------------------------
# SSE parsing
# --------------------------------------------------------------------------------------


class _SSEBuffer:
    """Collects ``data:`` lines of one server-sent event until the blank line that ends it."""

    def __init__(self) -> None:
        self._data: list[str] = []

    def feed(self, line: str) -> str | None:
        """Consume one line; return the event's data when this line completes an event."""
        line = line.rstrip("\r\n")
        if not line:
            return self.flush()
        if line.startswith(":"):  # comment, e.g. ": OPENROUTER PROCESSING"
            return None
        name, _, value = line.partition(":")
        if name == "data":
            self._data.append(value[1:] if value.startswith(" ") else value)
        return None  # other fields (event:, id:, retry:) carry nothing we use

    def flush(self) -> str | None:
        """Return the pending event's data (if any) and reset."""
        if not self._data:
            return None
        data, self._data = "\n".join(self._data), []
        return data


def _is_done(data: str) -> bool:
    return data.strip() == "[DONE]"


def _decode(data: str) -> dict[str, Any]:
    try:
        obj = json.loads(data)
    except json.JSONDecodeError as e:
        raise ValueError(f"malformed SSE data payload: {data[:200]!r}") from e
    if not isinstance(obj, dict):
        raise ValueError(f"SSE data payload is not a JSON object: {data[:200]!r}")
    return obj


def parse_sse(lines: Iterable[str]) -> Iterator[dict]:
    """Decode SSE text lines into JSON payloads; skips ``:`` comments and blank lines and stops at ``[DONE]``.

    Multi-line ``data:`` fields are joined with newlines. A malformed payload raises ValueError.
    """
    buf = _SSEBuffer()
    for line in lines:
        data = buf.feed(line)
        if data is None:
            continue
        if _is_done(data):
            return
        yield _decode(data)
    data = buf.flush()
    if data is not None and not _is_done(data):
        yield _decode(data)


# --------------------------------------------------------------------------------------
# Errors and helpers
# --------------------------------------------------------------------------------------


def _error_meta(err: Any) -> Mapping[str, Any]:
    """``error.metadata`` of an OpenRouter error object (empty if absent)."""
    meta = err.get("metadata") if isinstance(err, Mapping) else None
    return meta if isinstance(meta, Mapping) else {}


def _error_message(err: Any) -> str:
    if not isinstance(err, Mapping):
        return str(err)
    meta = _error_meta(err)
    label = " ".join(str(p) for p in (err.get("code"), meta.get("error_type")) if p is not None)
    message = str(err.get("message") or "unknown error")[:500]
    return f"{label}: {message}" if label else message


def _retry_after(value: str | None) -> float | None:
    """Parse a Retry-After header (seconds or HTTP date) into seconds, clamped to [0, MAX_RETRY_AFTER_S]."""
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=_dt.timezone.utc)
        seconds = (when - _dt.datetime.now(_dt.timezone.utc)).total_seconds()
    return min(max(seconds, 0.0), MAX_RETRY_AFTER_S)


def _http_error(response: httpx.Response, model: ModelSpec) -> BackendError:
    """Turn a non-200 OpenRouter response (already read) into a BackendError."""
    status = response.status_code
    err: Any = None
    try:
        payload = response.json()
        err = payload.get("error") if isinstance(payload, dict) else None
    except ValueError:
        pass
    message = str(err.get("message")) if isinstance(err, Mapping) and err.get("message") else response.text[:500]
    retry_after = _retry_after(response.headers.get("retry-after"))
    # 402 with Retry-After is OpenRouter's in-flight budget limit, which clears; plain 402 means no credits
    retryable = is_retryable_status(status) or (status == 402 and retry_after is not None)
    return BackendError(f"OpenRouter HTTP {status} for {model.provider_id}: {message}", status=status,
                        error_type=_error_meta(err).get("error_type"), retryable=retryable, retry_after=retry_after)


def _transport_error(exc: httpx.HTTPError, model: ModelSpec) -> BackendError:
    kind = "timeout" if isinstance(exc, httpx.TimeoutException) else "transport"
    return BackendError(f"request to {model.provider_id} failed: {exc!r}", error_type=kind, retryable=True)


def _deep_update(target: dict[str, Any], extra: Mapping[str, Any]) -> None:
    for k, v in extra.items():
        if isinstance(v, Mapping) and isinstance(target.get(k), dict):
            _deep_update(target[k], v)
        else:
            target[k] = copy.deepcopy(v)


def _message_text(content: Any) -> str:
    """Message content as text; some providers return a list of content parts."""
    if isinstance(content, list):
        return "".join(str(p.get("text", "")) for p in content if isinstance(p, Mapping))
    return content or ""


# --------------------------------------------------------------------------------------
# Stream state
# --------------------------------------------------------------------------------------


class _StreamState:
    """Accumulates one streamed segment and decides when to stop reading."""

    def __init__(self, sentinel: str, cut_chars: float | None) -> None:
        self.sentinel = sentinel
        self.cut_chars = cut_chars
        self.text = ""
        self.refusal = ""
        self.usage: dict[str, Any] | None = None
        self.finish: str | None = None
        self.done_seen = False

    def on_chunk(self, chunk: Mapping[str, Any]) -> tuple[list[StreamEvent], bool]:
        """Process one decoded chunk; returns the events to emit and whether to stop reading."""
        if chunk.get("usage"):
            self.usage = dict(chunk["usage"])
        if chunk.get("error"):
            self.finish = "error"
            return [StreamEvent(kind="error", error=_error_message(chunk["error"]), finish_reason="error")], True
        choices = chunk.get("choices") or []
        if not choices:
            return [], False
        choice = choices[0]
        delta = choice.get("delta") or {}
        events: list[StreamEvent] = []
        stop = False
        if delta.get("reasoning"):
            events.append(StreamEvent(kind="reasoning", text=delta["reasoning"]))
        if delta.get("refusal"):
            self.refusal += delta["refusal"]
        if delta.get("content"):
            event, stop = self._on_text(delta["content"])
            events.append(event)
        reason = choice.get("finish_reason")
        if reason and self.finish is None:
            self.finish = reason
            if reason == "error":
                events.append(StreamEvent(kind="error", error="provider reported finish_reason=error",
                                          finish_reason="error"))
                stop = True
        return events, stop

    def _on_text(self, piece: str) -> tuple[StreamEvent, bool]:
        before = len(self.text)
        self.text += piece
        idx = self.text.find(self.sentinel, max(0, before - len(self.sentinel) + 1))
        if idx >= 0:  # emit up to the end of the sentinel and nothing after it
            end = idx + len(self.sentinel)
            self.text = self.text[:end]
            self.finish = "sentinel"
            return StreamEvent(kind="text", text=piece[: end - before]), True
        if self.cut_chars is not None and len(self.text) > self.cut_chars:
            self.finish = "cut"
            return StreamEvent(kind="text", text=piece), True
        return StreamEvent(kind="text", text=piece), False

    def final_events(self) -> list[StreamEvent]:
        """The closing ``done`` event, preceded by an error event if the stream ended without completing."""
        events: list[StreamEvent] = []
        finish = self.finish
        if finish is None:
            if self.done_seen:
                finish = "stop"
            else:
                finish = "error"
                events.append(StreamEvent(kind="error", error="stream ended before a finish_reason or [DONE]",
                                          finish_reason="error"))
        note = f"refused: {self.refusal}" if finish == "content_filter" and self.refusal else None
        events.append(StreamEvent(kind="done", usage=self.usage, finish_reason=finish, error=note))
        return events


# --------------------------------------------------------------------------------------
# Backend
# --------------------------------------------------------------------------------------


class OpenRouterBackend:
    """``types.Backend`` implementation for OpenRouter's ``/chat/completions``.

    The API key is read from ``api_key`` or, at the first request, from the environment variable ``api_key_env``;
    a missing key raises :class:`BackendError` then, never at construction or import.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 120.0,
        app_title: str = "cutoffmix",
        referer: str | None = None,
        client: httpx.AsyncClient | None = None,
        max_retries: int = 3,
        api_key_env: str = "OPENROUTER_API_KEY",
        retry_base_delay: float = 1.0,
    ) -> None:
        self._api_key = api_key
        self.api_key_env = api_key_env
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.app_title = app_title
        self.referer = referer
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self._client = client
        self._owns_client = client is None

    async def __aenter__(self) -> OpenRouterBackend:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # ---- plumbing -------------------------------------------------------------------

    @property
    def url(self) -> str:
        """The chat-completions endpoint."""
        return f"{self.base_url}/chat/completions"

    def _headers(self) -> dict[str, str]:
        key = self._api_key or os.environ.get(self.api_key_env)
        if not key:
            raise BackendError(f"no OpenRouter API key: pass api_key or set the {self.api_key_env} environment "
                               "variable", error_type="missing_api_key")
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "X-Title": self.app_title,
            "User-Agent": USER_AGENT,
        }
        if self.referer:
            headers["HTTP-Referer"] = self.referer
        return headers

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:  # only reachable when no client was injected
            self._client = httpx.AsyncClient()
        return self._client

    def _timeout(self) -> httpx.Timeout:
        return httpx.Timeout(self.timeout, connect=min(self.timeout, _CONNECT_TIMEOUT_S))

    @staticmethod
    def _apply_model_params(body: dict[str, Any], model: ModelSpec, *, reasoning: Mapping[str, Any] | None,
                            temperature: float | None, run_id: str) -> None:
        if reasoning is not None:
            body["reasoning"] = copy.deepcopy(dict(reasoning))
        if model.provider_routing:
            body["provider"] = copy.deepcopy(dict(model.provider_routing))
        if model.supports_temperature and temperature is not None:
            body["temperature"] = temperature
        if model.cache_style == "openai" and run_id:
            key = f"{run_id}:{model.key}"[:256]
            body["prompt_cache_key"] = key
            body["session_id"] = key
        extra = copy.deepcopy(dict(model.extra_body))
        if model.provider_id.startswith("anthropic/"):
            extra.pop("verbosity", None)  # OpenRouter would map it to Claude's effort setting
        _deep_update(body, extra)

    # ---- request bodies ---------------------------------------------------------------

    def build_continuation_body(self, model: ModelSpec, req: ContinuationRequest) -> dict:
        """The streaming request body for one relay segment (``phase="think"``) or the final answer.

        ``stop`` is sent only for think segments on models that support it; temperature (``req.metadata
        ["temperature"]``) only if the model accepts it. Anthropic cache markers go on the system block and the last
        segment block of think requests (an answer request is a one-off prefix, so a cache write would be wasted).
        """
        think = req.phase == "think"
        if think:
            system = relay_system_prompt(req.target_tokens, req.today, req.sentinel, req.final_marker)
            blocks = relay_user_blocks(req.question, req.segments)
        else:
            system = answer_system_prompt(req.today)
            blocks = answer_user_blocks(req.question, req.segments)
        system_block: dict[str, Any] = {"type": "text", "text": system}
        if think and model.cache_style == "anthropic":
            system_block["cache_control"] = {"type": "ephemeral"}
            if req.segments:
                blocks[-2]["cache_control"] = {"type": "ephemeral"}  # last segment block, never the tail
        body: dict[str, Any] = {
            "model": model.provider_id,
            "messages": [{"role": "system", "content": [system_block]}, {"role": "user", "content": blocks}],
            "stream": True,
            "max_tokens": continuation_max_tokens(model, req),
        }
        if think and model.supports_stop:
            body["stop"] = [req.sentinel]
        self._apply_model_params(body, model, reasoning=model.reasoning, temperature=req.metadata.get("temperature"),
                                 run_id=req.run_id)
        return body

    def build_chat_body(self, model: ModelSpec, req: ChatRequest) -> dict:
        """The non-streaming request body for a plain chat call (benchmark probes, judges, probe generation)."""
        body: dict[str, Any] = {
            "model": model.provider_id,
            "messages": copy.deepcopy(req.messages),
            "stream": False,
            "max_tokens": req.max_tokens,
        }
        if req.response_format is not None:
            body["response_format"] = copy.deepcopy(req.response_format)
        reasoning = req.reasoning if req.reasoning is not None else model.reasoning
        self._apply_model_params(body, model, reasoning=reasoning, temperature=req.temperature, run_id="")
        return body

    # ---- streaming ----------------------------------------------------------------------

    async def stream_continuation(self, model: ModelSpec, req: ContinuationRequest) -> AsyncIterator[StreamEvent]:
        """Stream one segment as raw ``text``/``reasoning`` deltas followed by exactly one ``done`` event.

        Reading stops, and the HTTP response is closed, as soon as the accumulated text contains ``req.sentinel``
        (the delta is truncated right after it; finish_reason ``"sentinel"``) or, for think segments, exceeds
        ``req.hard_cut_chars * 1.5`` characters (``"cut"``). Leaving the ``async for`` early also closes the response.
        HTTP and connection errors raise :class:`BackendError` before the first event; afterwards failures are
        reported as an ``error`` event followed by ``done`` with finish_reason ``"error"``.
        """
        body = self.build_continuation_body(model, req)
        client = self._http()
        request = client.build_request("POST", self.url, json=body, headers=self._headers(), timeout=self._timeout())
        try:
            response = await client.send(request, stream=True)
        except httpx.HTTPError as e:
            raise _transport_error(e, model) from e
        try:
            if response.status_code != 200:
                try:
                    await response.aread()
                except httpx.HTTPError as e:
                    raise _transport_error(e, model) from e
                raise _http_error(response, model)
            cut = req.hard_cut_chars * CUT_FACTOR if req.phase == "think" else None
            state = _StreamState(req.sentinel, cut)
            try:
                async with contextlib.aclosing(self._read_events(response, state)) as events:
                    async for event in events:
                        yield event
            except (httpx.HTTPError, ValueError) as e:
                state.finish = "error"
                yield StreamEvent(kind="error", error=f"stream from {model.provider_id} failed: {e!r}",
                                  finish_reason="error")
            for event in state.final_events():
                yield event
        finally:
            await response.aclose()

    @staticmethod
    async def _read_events(response: httpx.Response, state: _StreamState) -> AsyncGenerator[StreamEvent, None]:
        buf = _SSEBuffer()
        async for line in response.aiter_lines():
            data = buf.feed(line)
            if data is None:
                continue
            if _is_done(data):
                state.done_seen = True
                return
            events, stop = state.on_chunk(_decode(data))
            for event in events:
                yield event
            if stop:
                return
        data = buf.flush()  # an event not terminated by a blank line before EOF
        if data is None:
            return
        if _is_done(data):
            state.done_seen = True
            return
        for event in state.on_chunk(_decode(data))[0]:
            yield event

    # ---- non-streaming ------------------------------------------------------------------

    async def chat(self, model: ModelSpec, req: ChatRequest) -> ChatResult:
        """One non-streaming chat call. Retries (up to ``max_retries``) on 408/429/5xx, retryable 402s, timeouts and
        connection errors, honouring Retry-After and otherwise backing off exponentially with jitter."""
        body = self.build_chat_body(model, req)
        headers = self._headers()
        attempt = 0
        while True:
            try:
                return await self._chat_once(model, body, headers)
            except BackendError as e:
                if not e.retryable or attempt >= self.max_retries:
                    raise
                delay = e.retry_after
                if delay is None:
                    delay = self.retry_base_delay * (2 ** attempt) * random.uniform(0.5, 1.0)
                attempt += 1
                await asyncio.sleep(delay)

    async def _chat_once(self, model: ModelSpec, body: dict[str, Any], headers: dict[str, str]) -> ChatResult:
        t0 = time.monotonic()
        try:
            response = await self._http().post(self.url, json=body, headers=headers, timeout=self._timeout())
        except httpx.HTTPError as e:
            raise _transport_error(e, model) from e
        if response.status_code != 200:
            raise _http_error(response, model)
        try:
            payload = response.json()
        except ValueError as e:
            raise BackendError(f"invalid JSON from {model.provider_id}: {response.text[:200]!r}",
                               status=response.status_code, error_type="bad_response", retryable=True) from e
        if not isinstance(payload, dict):
            raise BackendError(f"unexpected response from {model.provider_id}: {payload!r:.200}",
                               error_type="bad_response", retryable=True)
        err = payload.get("error")
        if err:
            code = err.get("code") if isinstance(err, Mapping) else None
            status = code if isinstance(code, int) else None
            raise BackendError(f"{model.provider_id}: {_error_message(err)}", status=status,
                               error_type=_error_meta(err).get("error_type"),
                               retryable=status is None or is_retryable_status(status))
        choices = payload.get("choices") or []
        if not choices:
            raise BackendError(f"response from {model.provider_id} has no choices", error_type="bad_response",
                               retryable=True)
        choice = choices[0]
        message = choice.get("message") or {}
        return ChatResult(
            text=_message_text(message.get("content")),
            reasoning=message.get("reasoning") or "",
            usage=payload.get("usage"),
            finish_reason=choice.get("finish_reason"),
            latency_s=time.monotonic() - t0,
            raw=payload,
        )

    async def aclose(self) -> None:
        """Close the HTTP client if this backend created it."""
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None
