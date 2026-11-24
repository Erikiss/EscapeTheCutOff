"""HTTP server: the web UI, rich relay events over SSE and an OpenAI-compatible endpoint (DESIGN.md §6).

Endpoints of :func:`create_app`:

- ``GET /`` -- the single-file web UI (``cutoffmix/web/index.html``).
- ``GET /api/config`` -- models with colours, topic names, defaults and the selectable strategies/policies.
- ``GET /api/index[?probe=]`` -- the cutoff index as table rows (topic, model, tau, tau_label, sd, n, hazard, p_best).
- ``POST /api/relay`` -- ``{"question", "policy"?, "strategy"?, "segment_tokens"?}``; answers with server-sent
  events ``data: RelayEvent.to_dict()``.
- ``GET /v1/models`` and ``POST /v1/chat/completions`` -- OpenAI-compatible, streaming and non-streaming. Model ids
  ``cutoffmix``, ``cutoffmix-speculative`` and ``cutoffmix-fixed-<model key>``. The conversation is flattened into
  the PROBLEM text; the shared notes go to ``delta.reasoning`` (plus ``delta.cutoffmix`` metadata), the answer to
  ``delta.content``.

FastAPI is imported inside :func:`create_app`, so this module imports without the optional server dependencies.
Each SSE stream polls for a client disconnect and stops the relay (closing the in-flight model stream) when the
client goes away; idle streams get ``: ping`` comments so proxies keep them open.

(No ``from __future__ import annotations`` here: FastAPI must see the real ``Request`` class in the endpoint
signatures defined inside :func:`create_app`; names imported only for type checking are quoted.)
"""

import asyncio
import contextlib
import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from importlib import resources
from typing import TYPE_CHECKING, Any

from cutoffmix import __version__
from cutoffmix.relay import RelayRuntime, to_jsonable
from cutoffmix.types import Backend, RelayEvent

if TYPE_CHECKING:
    from fastapi import FastAPI, Request

    from cutoffmix.config import AppConfig
    from cutoffmix.router.index import CutoffIndex

__all__ = ["MODEL_ID", "SSE_HEADERS", "create_app", "flatten_messages", "index_payload", "model_ids"]

log = logging.getLogger(__name__)

MODEL_ID = "cutoffmix"
SPECULATIVE_MODEL_ID = "cutoffmix-speculative"
FIXED_MODEL_PREFIX = "cutoffmix-fixed-"
STRATEGIES: tuple[str, ...] = ("sequential", "speculative")
POLICIES: tuple[str, ...] = ("softmax", "thompson", "argmax", "round_robin", "random")
SSE_HEADERS: dict[str, str] = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
HEARTBEAT_S = 15.0  # ": ping" comment after this much silence
POLL_S = 0.5  # how often an idle stream checks for a client disconnect
MAX_QUESTION_CHARS = 20_000
MAX_SEGMENT_TOKENS = 4096
_CREATED = 1_764_000_000  # fixed "created" timestamp of the virtual models
_END = object()


class RequestError(ValueError):
    """A client error: invalid JSON, a missing field or an invalid override (HTTP 400)."""


# --------------------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------------------


def _dumps(obj: Any) -> str:
    return json.dumps(to_jsonable(obj), ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def model_ids(model_keys: Sequence[str]) -> list[str]:
    """The OpenAI-compatible model ids served for these relay models."""
    return [MODEL_ID, SPECULATIVE_MODEL_ID, *(FIXED_MODEL_PREFIX + k for k in model_keys)]


def _model_overrides(model_id: str, model_keys: Sequence[str]) -> dict[str, Any]:
    """Per-run overrides implied by an OpenAI model id; ``LookupError`` for an unknown id."""
    if model_id == MODEL_ID:
        return {}
    if model_id == SPECULATIVE_MODEL_ID:
        return {"strategy": "speculative"}
    if model_id.startswith(FIXED_MODEL_PREFIX) and model_id[len(FIXED_MODEL_PREFIX):] in model_keys:
        return {"policy": f"fixed:{model_id[len(FIXED_MODEL_PREFIX):]}"}
    raise LookupError(f"model {model_id!r} does not exist; available: {', '.join(model_ids(model_keys))}")


def _run_overrides(body: Mapping[str, Any]) -> dict[str, Any]:
    """The per-request overrides ``policy``, ``strategy`` and ``segment_tokens`` (type-checked)."""
    out: dict[str, Any] = {}
    for key in ("policy", "strategy"):
        value = body.get(key)
        if value is not None:
            if not isinstance(value, str) or not value.strip():
                raise RequestError(f"{key} must be a non-empty string")
            out[key] = value.strip()
    tokens = body.get("segment_tokens")
    if tokens is not None:
        if isinstance(tokens, bool) or not isinstance(tokens, (int, float)) or int(tokens) != tokens:
            raise RequestError("segment_tokens must be an integer")
        if not 1 <= int(tokens) <= MAX_SEGMENT_TOKENS:
            raise RequestError(f"segment_tokens must be between 1 and {MAX_SEGMENT_TOKENS}")
        out["segment_tokens"] = int(tokens)
    return out


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(p.get("text", "")) for p in content
                         if isinstance(p, Mapping) and p.get("type") in ("text", "input_text"))
    return ""


_ROLE_LABELS = {"system": "Instructions", "developer": "Instructions", "user": "User", "assistant": "Assistant",
                "tool": "Tool result"}


def flatten_messages(messages: Any) -> str:
    """Flatten an OpenAI ``messages`` list into the relay's PROBLEM text.

    A conversation of one user message becomes that message; anything longer becomes ``Role: text`` paragraphs
    (system/developer messages as "Instructions"). Raises :class:`RequestError` when there is no text.
    """
    if not isinstance(messages, list) or not messages:
        raise RequestError("messages must be a non-empty list")
    parts: list[tuple[str, str]] = []
    for m in messages:
        if not isinstance(m, Mapping):
            raise RequestError("every message must be an object with role and content")
        text = _content_text(m.get("content")).strip()
        if text:
            parts.append((str(m.get("role", "user")), text))
    if not parts:
        raise RequestError("messages contain no text")
    if len(parts) == 1 and parts[0][0] == "user":
        return parts[0][1]
    return "\n\n".join(f"{_ROLE_LABELS.get(role, role.title())}: {text}" for role, text in parts)


def index_payload(index: "CutoffIndex | None", *, probe: str | None = None, source: str | None = None,
                  preference: Sequence[str] = ("current_d", "current", "recall", "mcq")) -> dict[str, Any]:
    """The ``/api/index`` body: table rows of one probe (default: the first of ``preference`` that exists)."""
    if index is None:
        return {"available": False, "probe": None, "probes": [], "models": [], "topics": [], "rows": [],
                "meta": {}, "source": None}
    name = probe or index.primary(tuple(preference)).probe
    if name not in index.probes:
        raise RequestError(f"probe {name!r} not in the index; available: {', '.join(index.probes)}")
    return {"available": True, "probe": name, "probes": list(index.probes), "models": list(index.models),
            "topics": list(index.topics), "rows": index.table(name), "meta": index.meta, "source": source}


def _openai_usage(done: Mapping[str, Any] | None) -> dict[str, Any]:
    total = dict((done or {}).get("usage_total") or {})
    total.pop("calls", None)
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        total.setdefault(key, 0)
    return total


def _thinking(done: Mapping[str, Any] | None) -> str:
    segments = (done or {}).get("segments") or []
    return "\n\n".join(s["text"] for s in segments if s.get("phase") == "think" and s.get("text"))


def _ui_html() -> str:
    return resources.files("cutoffmix").joinpath("web").joinpath("index.html").read_text(encoding="utf-8")


# --------------------------------------------------------------------------------------
# Streaming plumbing
# --------------------------------------------------------------------------------------


async def _pump_events(events: AsyncGenerator[RelayEvent, None],
                       request: "Request") -> AsyncIterator[RelayEvent | None]:
    """Run the relay in its own task and hand its events over (``None`` = heartbeat due).

    Stops when the relay ends or the client disconnects; leaving early cancels the relay task, which closes the
    relay generator and with it the in-flight model stream.
    """
    queue: asyncio.Queue[Any] = asyncio.Queue()

    async def pump() -> None:
        try:
            async with contextlib.aclosing(events) as it:
                async for event in it:
                    queue.put_nowait(event)
        except Exception as e:  # a relay bug must still end the stream cleanly
            log.exception("relay failed")
            queue.put_nowait(RelayEvent(type="error", text=f"internal error: {type(e).__name__}: {e}"))
        finally:
            queue.put_nowait(_END)

    task = asyncio.create_task(pump())
    idle = 0.0
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=POLL_S)
            except asyncio.TimeoutError:  # the builtin TimeoutError on Python >= 3.11
                if await request.is_disconnected():
                    return
                idle += POLL_S
                if idle >= HEARTBEAT_S:
                    idle = 0.0
                    yield None
                continue
            if item is _END:
                return
            idle = 0.0
            yield item
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def _sse_relay(events: AsyncIterator[RelayEvent | None]) -> AsyncIterator[str]:
    async for event in events:
        yield ": ping\n\n" if event is None else f"data: {_dumps(event.to_dict())}\n\n"


async def _sse_openai(events: AsyncIterator[RelayEvent | None], *, completion_id: str, created: int,
                      model_id: str) -> AsyncIterator[str]:
    """OpenAI ``chat.completion.chunk`` stream: notes in ``delta.reasoning``, the answer in ``delta.content``."""

    def chunk(delta: dict[str, Any], finish: str | None = None, **extra: Any) -> str:
        body = {"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model_id,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], **extra}
        return f"data: {_dumps(body)}\n\n"

    yield chunk({"role": "assistant", "content": ""})
    done: dict[str, Any] | None = None
    error: str | None = None
    need_separator = False  # a committed notes segment was streamed: separate the next one by a blank line
    async for event in events:
        if event is None:
            yield ": ping\n\n"
            continue
        if event.type == "delta":
            if event.phase == "answer":
                yield chunk({"content": event.text})
            else:
                text = ("\n\n" if need_separator else "") + event.text
                need_separator = False
                yield chunk({"reasoning": text,
                             "cutoffmix": {"type": "delta", "segment": event.segment, "model": event.model}})
            continue
        if event.type == "done":
            done = event.data or {}
            continue
        if event.type == "error":
            error = event.text
        if (event.type == "segment_end" and event.phase == "think" and event.text
                and not (event.data or {}).get("dropped")):
            need_separator = True
        meta = event.to_dict()
        if event.type == "segment_end":
            meta.pop("data", None)  # keep chunks small; the full trace arrives with the final chunk
        yield chunk({"cutoffmix": meta})
    failed = error is not None and not (done or {}).get("answer")
    extra = {"error": {"message": error, "code": 502}} if failed else {}
    yield chunk({}, "error" if failed else "stop", **extra)
    final = {"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": model_id,
             "choices": [], "usage": _openai_usage(done), "cutoffmix": done}
    yield f"data: {_dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


# --------------------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------------------


def create_app(cfg: "AppConfig | None" = None, *, backend: Backend | None = None, mock: bool = False) -> "FastAPI":
    """Build the FastAPI app. ``cfg`` defaults to the packaged config; ``backend`` defaults to OpenRouter (or the
    mock backend with ``mock=True``). A backend created here is closed when the app shuts down."""
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import HTMLResponse, Response, StreamingResponse
    except ImportError as e:  # pragma: no cover - depends on the installation
        raise ImportError("the cutoffmix server needs FastAPI: pip install 'cutoffmix[server]'") from e
    from cutoffmix.config import load_config

    config = cfg if cfg is not None else load_config()
    runtime = RelayRuntime(config, backend=backend, mock=mock)
    index_cache: dict[str, dict[str, Any]] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        await runtime.aclose()

    app = FastAPI(title="cutoffmix", version=__version__, lifespan=lifespan)
    app.state.runtime = runtime

    def json_response(body: Any, status: int = 200) -> Response:
        return Response(_dumps(body), status_code=status, media_type="application/json")

    def bad_request(message: str) -> Response:
        return json_response({"detail": message}, 400)

    def openai_error(status: int, message: str, kind: str, code: str | int | None = None) -> Response:
        return json_response({"error": {"message": message, "type": kind, "code": code}}, status)

    async def json_body(request: Request) -> dict[str, Any]:
        try:
            body = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise RequestError(f"the request body must be JSON: {e}") from e
        if not isinstance(body, dict):
            raise RequestError("the request body must be a JSON object")
        return body

    def run_config(overrides: Mapping[str, Any]) -> "AppConfig":
        try:
            return runtime.configure(**overrides)
        except (ValueError, TypeError) as e:
            raise RequestError(str(e)) from e

    def sse(body: AsyncIterator[str]) -> StreamingResponse:
        return StreamingResponse(body, media_type="text/event-stream", headers=SSE_HEADERS)

    @app.get("/", response_class=HTMLResponse)
    async def ui() -> HTMLResponse:
        return HTMLResponse(_ui_html(), headers={"Cache-Control": "no-cache"})

    @app.get("/api/config")
    async def api_config() -> Response:
        from cutoffmix.taxonomy import TOPICS

        rc, ro = config.relay, config.router
        return json_response({
            "app": config.app_title, "version": __version__, "mock": runtime.mock, "today": config.today_str(),
            "models": [{"key": m.key, "display": m.label(), "color": m.color, "provider_id": m.provider_id}
                       for m in config.models],
            "topics": [{"id": t.id, "name": t.name} for t in TOPICS],
            "defaults": {"strategy": rc.strategy, "policy": ro.policy, "segment_tokens": rc.segment_tokens,
                         "max_segments": rc.max_segments, "answer_mode": rc.answer_mode,
                         "speculative_gate": rc.speculative_gate},
            "strategies": list(STRATEGIES),
            "policies": [*POLICIES, *(f"fixed:{k}" for k in config.model_keys())],
            "model_ids": model_ids(config.model_keys()),
        })

    @app.get("/api/index")
    async def api_index(probe: str | None = None) -> Response:
        key = probe or ""
        if key not in index_cache:
            try:
                index_cache[key] = index_payload(runtime.index, probe=probe, source=runtime.index_source,
                                                 preference=config.router.probe_preference)
            except RequestError as e:
                return bad_request(str(e))
        return json_response(index_cache[key])

    @app.post("/api/relay")
    async def api_relay(request: Request) -> Response:
        try:
            body = await json_body(request)
            question = body.get("question")
            if not isinstance(question, str) or not question.strip():
                raise RequestError("question must be a non-empty string")
            if len(question) > MAX_QUESTION_CHARS:
                raise RequestError(f"question is longer than {MAX_QUESTION_CHARS} characters")
            relay = runtime.relay(run_config(_run_overrides(body)))
        except RequestError as e:
            return bad_request(str(e))
        events = relay.run(question, run_id=uuid.uuid4().hex[:16])
        return sse(_sse_relay(_pump_events(events, request)))

    @app.get("/v1/models")
    async def v1_models() -> Response:
        return json_response({"object": "list", "data": [
            {"id": mid, "object": "model", "created": _CREATED, "owned_by": "cutoffmix"}
            for mid in model_ids(config.model_keys())]})

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        try:
            body = await json_body(request)
            model_id = body.get("model") or MODEL_ID
            if not isinstance(model_id, str):
                raise RequestError("model must be a string")
            try:
                overrides = _model_overrides(model_id, config.model_keys())
            except LookupError as e:
                return openai_error(404, str(e), "invalid_request_error", "model_not_found")
            overrides |= _run_overrides(body)
            question = flatten_messages(body.get("messages"))
            relay = runtime.relay(run_config(overrides))
        except RequestError as e:
            return openai_error(400, str(e), "invalid_request_error")
        run_id = uuid.uuid4().hex[:24]
        completion_id, created = f"chatcmpl-{run_id}", int(time.time())
        events = relay.run(question, run_id=run_id)
        if body.get("stream"):
            return sse(_sse_openai(_pump_events(events, request), completion_id=completion_id, created=created,
                                   model_id=model_id))
        done: dict[str, Any] | None = None
        error: str | None = None
        async with contextlib.aclosing(events) as it:
            async for event in it:
                if event.type == "done":
                    done = event.data
                elif event.type == "error":
                    error = event.text
        answer = (done or {}).get("answer") or ""
        if error is not None and not answer:
            return json_response({"error": {"message": error, "type": "upstream_error", "code": 502},
                                  "cutoffmix": done}, 502)
        return json_response({
            "id": completion_id, "object": "chat.completion", "created": created, "model": model_id,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": answer, "reasoning": _thinking(done)}}],
            "usage": _openai_usage(done),
            "cutoffmix": done,
        })

    return app
