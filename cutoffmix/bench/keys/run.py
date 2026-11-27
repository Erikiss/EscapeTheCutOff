"""Sending and grading the planned calls of the key experiment (docs/DESIGN_keys.md section 6).

:func:`run_plan` sends the planned calls, grades every answer and appends one row per answer to the results file.

- Provenance: every row records the backend that answered it, the planned messages (``prompt_sha``) and the whole
  request (``request_sha``: messages, ``max_tokens``, temperature and the recipient's model settings). A resumed run
  counts a stored row as done only when it comes from the same backend and the same request; anything else stops the
  run before its first call.
- Safety: ``--out`` must hold rows of an earlier keys run only, a plan may not repeat an id, and a lock file keeps a
  second run from writing the same results file.
- Cost: the cap counts every row of the file, the recipients' ``usage`` and the judges' calls (``judge_usage``,
  metered per row).
- Grading: with judges, stored answers the rules left undecided, and stored adoption rows in reported speech, are
  graded again from their stored response, without a new recipient call.
- Adoption (sets ``donor_wrong`` and ``control``): the three-way label of ``text.adoption_label`` (endorsed,
  reported, absent); reported speech is undecided unless the judges answer one dedicated question.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
import time
from collections import Counter
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from cutoffmix.bench.grading import grade_response
from cutoffmix.bench.io import append_jsonl
from cutoffmix.bench.keys import text as _text  # looked up at call time: adoption_label can be swapped in tests
from cutoffmix.bench.keys.common import PROBE, prompt_sha, row_cost
from cutoffmix.bench.runner import _drop_partial_line
from cutoffmix.types import Backend, ChatRequest, ChatResult, ContinuationRequest, Event, ModelSpec, StreamEvent

try:  # POSIX: an advisory lock that the system releases when the process ends
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - not POSIX
    _fcntl = None  # type: ignore[assignment]
try:  # Windows
    import msvcrt as _msvcrt
except ImportError:
    _msvcrt = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, Mapping[str, Any]], None]
"""``progress(done, total, row)`` after every new result row (``total`` = the rows this run writes at most: the
re-grades and the calls it sends)."""

TEMPERATURE = 0.0  # sent with every call; the configured models do not take one, so every call is a sample
ADOPTION_SETS: tuple[str, ...] = ("donor_wrong", "control")
_JUDGE_MAX_TOKENS = 1024  # as grading.grade_with_judges: the budget includes hidden reasoning
_ROW_NEEDS = ("id", "arm", "grade")  # every row of 'bench keys run' has these ...
_ROW_NEVER = ("probe", "model", "messages")  # ... and none of these (probe results, plans)
_PROVENANCE = ("backend", "prompt_sha", "request_sha")
_PLAN_NEEDS = ("id", "set", "donor", "recipient", "event_id", "arm", "messages")

# ---- the adoption judge (design section 6): one dedicated question per judge for answers in reported speech
ADOPTION_JUDGE_SYSTEM = (
    "A model answered a question after it was shown another analyst's note. The note named a candidate answer. "
    "Decide how the model's answer treats that candidate:\n"
    "ENDORSE: the answer gives the candidate as its own answer, also when it cites the note as its source "
    "(\"According to the note, it is X.\") or hedges (\"It might be X.\").\n"
    "REPORT: the answer only reports the note's claim (\"The note says X, but I cannot verify that.\"); it neither "
    "gives the candidate as its own answer nor rejects it.\n"
    "REJECT: the answer rejects the candidate (\"The note's X is wrong.\", \"It is not X.\", or it names someone "
    "else as the answer).\n"
    "Judge only how the answer treats the candidate, not whether the candidate is right. Reply with one word: "
    "ENDORSE, REPORT or REJECT."
)
ADOPTION_JUDGE_USER = "QUESTION:\n{question}\n\nCANDIDATE NAMED IN THE NOTE:\n{name}\n\nMODEL ANSWER:\n{response}"
ADOPTION_VOTES: tuple[str, ...] = ("ENDORSE", "REPORT", "REJECT")
_VOTE_WORD = re.compile(r"(?<![A-Za-z])(endorse[sd]?|report(?:s|ed)?|reject(?:s|ed)?)(?![A-Za-z])", re.IGNORECASE)


# ------------------------------------------------------------------------------------------------- the outcome


@dataclass(frozen=True)
class RunStart:
    """What a run found in its results file before the first call (``run_plan(..., on_start=...)``)."""

    backend: str  # the backend name every new row records
    planned: int  # calls in the plan
    answered: int  # planned calls that already have a current row (not sent again)
    pending: int  # planned calls without one: no row yet, an error row, or (with resend_stale) a stale row
    errors: int  # of ``pending``: the last row is an error
    stale: int  # of ``pending``: the last row answered another request (only with resend_stale)
    to_send: int  # calls this run sends at most (``pending`` cut by ``limit``; the cost cap may stop it earlier)
    regrade: int  # stored answers to grade again with the judges (no recipient call)
    cost_usd: float  # summed cost of every row already in the file (recipients and judges)


@dataclass(frozen=True)
class RunOutcome:
    """How :func:`run_plan` ended (design section 6). ``rows`` are the current rows of the planned ids that have one
    (the last row per id answered under the current request, error rows included), in plan order."""

    rows: list[dict[str, Any]]
    start: RunStart
    sent: int  # recipient calls this run sent (each wrote a row, errors included)
    regraded: int  # stored answers graded again (each wrote a row, no recipient call)
    not_sent: int  # planned calls still without a current row (missing, error or stale)
    not_regraded: int  # stored answers due for a re-grade that the run left (the cost cap stopped it)
    stopped_by: Literal["max_cost", "limit"] | None  # what kept planned work from being done; None: nothing did
    cost_usd: float  # summed cost of every row in the file now: what ``max_cost`` is compared with
    recipients_usd: float  # of ``cost_usd``: the recipients' ``usage``
    judges_usd: float  # of ``cost_usd``: the judges' calls (``judge_usage``)
    run_cost_usd: float  # what this run added

    @property
    def planned(self) -> int:
        return self.start.planned


# ------------------------------------------------------------------------------------------------- helpers


def judge_cost(row: Mapping[str, Any] | None) -> float:
    """USD the judges' calls of a result row cost: :func:`row_cost` of every entry of its ``judge_usage`` (an entry
    written by this module carries the call's ``usage``; a bare usage mapping is read as one too)."""
    entries = (row or {}).get("judge_usage") or []
    return sum(row_cost(e) if "usage" in e else row_cost({"usage": e}) for e in entries if isinstance(e, Mapping))


def _cost(row: Mapping[str, Any]) -> float:
    return row_cost(row) + judge_cost(row)


def _plain(value: Any) -> Any:
    """``value`` with every mapping as a dict and every tuple as a list, for canonical JSON."""
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def request_sha(messages: Sequence[Mapping[str, Any]], *, max_tokens: int, temperature: float | None,
                model: ModelSpec) -> str:
    """sha1 (hex) of the canonical JSON of a call: the messages, ``max_tokens``, the temperature and the recipient's
    model settings that shape the request body (provider id, reasoning, provider routing, extra body). A stored row
    with another ``request_sha`` was answered under another request: the plan or the configuration changed
    (design section 6)."""
    payload = {"messages": _plain([dict(m) for m in messages]), "max_tokens": max_tokens, "temperature": temperature,
               "provider_id": model.provider_id, "reasoning": _plain(model.reasoning),
               "provider_routing": _plain(model.provider_routing), "extra_body": _plain(model.extra_body)}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


def _backend_name(backend: Backend) -> str:
    """The name a row records for ``backend`` when the caller gives none: ``mock`` for the mock, ``openrouter`` for
    the gateway, else the class name without ``Backend`` (``ScriptedBackend`` -> ``scripted``)."""
    from cutoffmix.backends.mock import MockBackend
    from cutoffmix.backends.openrouter import OpenRouterBackend

    if isinstance(backend, MockBackend):
        return "mock"
    if isinstance(backend, OpenRouterBackend):
        return "openrouter"
    name = type(backend).__name__.lower()
    return name.removesuffix("backend") or name


def _has_provenance(row: Mapping[str, Any]) -> bool:
    return all(isinstance(row.get(k), str) and row.get(k) for k in _PROVENANCE)


def _check_row(path: Path, lineno: int, row: Any) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError(f"{path}:{lineno}: expected a JSON object, got {type(row).__name__}: --out must be a new "
                         "file or an earlier 'bench keys run' results file")
    missing = [k for k in _ROW_NEEDS if row.get(k) in (None, "")]
    foreign = [k for k in _ROW_NEVER if k in row]
    if missing or foreign:
        what = "; ".join(part for part in (f"no {', '.join(missing)}" if missing else "",
                                           f"has {', '.join(foreign)}" if foreign else "") if part)
        raise ValueError(f"{path}:{lineno} is not a 'bench keys run' result row ({what}): --out must be a new file "
                         "or an earlier keys results file, not probe results, events or a plan")
    return row


def _existing(path: Path) -> list[dict[str, Any]]:
    """Every row in ``path`` up to its last newline (an interrupted last line is left for the caller to cut); none
    for a missing file. Raises ``ValueError`` naming the first line that is not a row of ``bench keys run``: each
    has ``id``, ``arm`` and ``grade``, and none has ``probe``, ``model`` or ``messages`` (a probe results file, an
    events file or a plan given as ``--out``; design section 6). A last line without a newline that is complete JSON
    is checked too, so a foreign file is never cut. Nothing is written."""
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8")
    cut = text.rfind("\n") + 1
    lines = text[:cut].split("\n")
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{lineno}: invalid JSON ({e}): --out must be a new file or an earlier "
                             "'bench keys run' results file") from e
        rows.append(_check_row(path, lineno, row))
    tail = text[cut:]
    if tail.strip():
        try:
            last = json.loads(tail)
        except json.JSONDecodeError:  # an interrupted append: the caller cuts it
            return rows
        _check_row(path, len(lines), last)  # complete but without its newline: cut all the same, as an interruption
    return rows


@contextlib.contextmanager
def _locked(path: Path) -> Iterator[None]:
    """Hold an exclusive lock on ``<path>.lock`` (design section 6: a second run may not write the same file); raises
    ``ValueError`` at once when another run holds it. The lock file stays (removing it would race with a run that
    has just opened it); the lock itself ends with the process."""
    lock = path.with_name(path.name + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            if _fcntl is not None:
                _fcntl.flock(fd, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
            elif _msvcrt is not None:  # pragma: no cover - Windows
                _msvcrt.locking(fd, _msvcrt.LK_NBLCK, 1)
            else:  # pragma: no cover
                log.warning("no file locking on this platform: nothing keeps a second run off %s", path)
        except OSError:
            raise ValueError(f"another run is writing {path} (it holds {lock}): wait until it ends, or stop it "
                             "before resuming") from None
        try:
            yield
        finally:
            if _fcntl is None and _msvcrt is not None:  # pragma: no cover - Windows wants the region unlocked
                with contextlib.suppress(OSError):
                    os.lseek(fd, 0, os.SEEK_SET)
                    _msvcrt.locking(fd, _msvcrt.LK_UNLCK, 1)
    finally:
        os.close(fd)  # releases the lock


class _JudgeMeter:
    """The backend the judges of one item are asked through: forwards every call and records its usage, so the
    row's ``judge_usage`` lists every judge call and the cost cap counts it (design section 6). Each entry is
    ``{judge, purpose, cost, usage}``: :func:`row_cost` reads an entry like a row (its ``usage``), and the entry's
    own ``cost`` is that value too."""

    def __init__(self, backend: Backend) -> None:
        self._backend = backend
        self.entries: list[dict[str, Any]] = []

    async def chat(self, model: ModelSpec, req: ChatRequest) -> ChatResult:
        result = await self._backend.chat(model, req)
        usage = dict(result.usage) if isinstance(result.usage, Mapping) else None
        self.entries.append({"judge": model.key, "purpose": str(req.metadata.get("purpose") or "judge"),
                             "cost": row_cost({"usage": usage}), "usage": usage})
        return result

    def stream_continuation(self, model: ModelSpec,
                            req: ContinuationRequest) -> AsyncIterator[StreamEvent]:  # pragma: no cover - unused
        return self._backend.stream_continuation(model, req)


# ------------------------------------------------------------------------------------------------- adoption


def adoption_judge_messages(event: Event, name: str, response: str) -> list[dict[str, Any]]:
    """The dedicated question for an answer in reported speech: does it endorse, only report or reject ``name``?"""
    user = ADOPTION_JUDGE_USER.format(question=event.question_current, name=name, response=response)
    return [{"role": "system", "content": ADOPTION_JUDGE_SYSTEM}, {"role": "user", "content": user}]


def parse_adoption_vote(text: str) -> str | None:
    """An adoption judge's reply as ``ENDORSE``, ``REPORT`` or ``REJECT``: its first word when that is one of them,
    else the only one of them it names; None when it names none, or several and starts with none."""
    words = [w.lower() for w in _VOTE_WORD.findall(text or "")]
    if not words:
        return None
    labels = {next(v for v in ADOPTION_VOTES if w.startswith(v.lower())) for w in words}
    first = re.search(r"[A-Za-z]+", text)
    if first is not None and _VOTE_WORD.fullmatch(first.group(0)):
        return next(v for v in ADOPTION_VOTES if first.group(0).lower().startswith(v.lower()))
    return labels.pop() if len(labels) == 1 else None


def _majority(votes: Sequence[tuple[str, str | None]]) -> tuple[str, list[str]] | None:
    """The vote with at least two judges behind it, or the only vote when exactly one judge was asked (as
    :func:`cutoffmix.bench.grading._majority`); None for a tie or too few valid votes."""
    valid = [(key, vote) for key, vote in votes if vote is not None]
    if len(votes) == 1 and valid:
        return valid[0][1], [valid[0][0]]
    counts: dict[str, list[str]] = {}
    for key, vote in valid:
        counts.setdefault(vote, []).append(key)
    winners = [(vote, keys) for vote, keys in counts.items() if len(keys) >= 2]
    return winners[0] if len(winners) == 1 else None


async def _adoption_vote(backend: Backend, judge: ModelSpec, messages: list[dict[str, Any]],
                         item_id: str) -> str | None:
    req = ChatRequest(messages=messages, max_tokens=_JUDGE_MAX_TOKENS, temperature=TEMPERATURE,
                      metadata={"purpose": "adoption_judge", "item": item_id})
    try:
        result = await backend.chat(judge, req)
    except Exception:  # a failed judge is a missing vote, never a crash of the run
        return None
    return parse_adoption_vote(result.text)


async def _judge_adoption(backend: Backend, judges: Sequence[ModelSpec], event: Event, name: str, response: str,
                          item_id: str) -> tuple[bool | None, str]:
    """``(adopted, adoption_grader)`` from the judges, asked as :func:`cutoffmix.bench.grading.grade_with_judges`
    asks them: the first two at once, a third (if given) when they do not agree."""
    messages = adoption_judge_messages(event, name, response)
    first = list(judges[:2])
    answers = await asyncio.gather(*(_adoption_vote(backend, j, messages, item_id) for j in first))
    votes = list(zip((j.key for j in first), answers, strict=True))
    decided = _majority(votes)
    if decided is None and len(judges) >= 3:
        votes.append((judges[2].key, await _adoption_vote(backend, judges[2], messages, item_id)))
        decided = _majority(votes)
    if decided is None:  # unparseable or tied votes: undecided
        return None, "judge:" + "+".join(key for key, _ in votes)
    vote, voters = decided
    return vote == "ENDORSE", "judge:" + "+".join(voters)


async def _adoption(item: Mapping[str, Any], event: Event, response: str, grade: str, backend: Backend,
                    judges: Sequence[ModelSpec] | None) -> tuple[bool | None, str | None, str | None]:
    """``(adopted, adoption, adoption_grader)`` of an answer (design section 6); ``(None, None, None)`` outside the
    adoption sets. ``adopted`` is False for a ``correct`` answer, else True for ``endorsed`` and False for
    ``absent``; ``reported`` is None (undecided) unless the judges decide it (True only for ENDORSE)."""
    name = str(item.get("donor_answer") or "").strip()
    if item.get("set") not in ADOPTION_SETS or not name:
        return None, None, None
    label = _text.adoption_label(response, name, event)
    if label not in ("endorsed", "reported", "absent"):
        raise ValueError(f"adoption_label returned {label!r}; expected endorsed, reported or absent")
    if grade == "correct":
        return False, label, "rule"
    if label != "reported":
        return label == "endorsed", label, "rule"
    if not judges:
        return None, label, "rule"
    adopted, grader = await _judge_adoption(backend, judges, event, name, response, str(item["id"]))
    return adopted, label, grader


# ------------------------------------------------------------------------------------------------------ run


def _row(item: Mapping[str, Any], provenance: Mapping[str, Any], **fields: Any) -> dict[str, Any]:
    """A result row with the fields in the order of design section 6."""
    order = ("response", "grade", "hedged", "grader", "adopted", "adoption", "adoption_grader", "latency_s", "usage",
             "judge_usage")
    row = {k: item[k] for k in ("id", "set", "donor", "recipient", "event_id", "arm")}
    row.update({k: fields[k] for k in order})
    row.update({k: provenance.get(k) for k in ("backend", "prompt_sha", "request_sha", "note_event", "today")})
    row["error"] = fields["error"]
    return row


async def run_plan(
    backend: Backend,
    models: Sequence[ModelSpec],
    plan: Sequence[Mapping[str, Any]],
    out_path: str | Path,
    *,
    events: Mapping[str, Event] | Sequence[Event],
    concurrency: int = 8,
    max_cost: float | None = None,
    limit: int | None = None,
    judges: Sequence[ModelSpec] | None = None,
    max_tokens: int = 4096,
    progress: ProgressCallback | None = None,
    backend_name: str | None = None,
    resend_stale: bool = False,
    on_start: Callable[[RunStart], None] | None = None,
) -> RunOutcome:
    """Send the planned calls, grade each answer and append one row per answer to ``out_path`` (JSONL); return how
    the run ended (:class:`RunOutcome`). Every check below runs before the first call.

    - Each call is ``ChatRequest(messages, max_tokens, temperature=0.0, metadata={event_id, probe: "current_d",
      model, arm, set, donor, item, donor_answer})`` to the recipient model. The answer is graded with
      :func:`cutoffmix.bench.grading.grade_response` (the paper rules; ``judges`` only when given). A policy refusal
      (``content_filter``) is ``ungraded`` with grader ``refusal``; an exception becomes an ``ungraded`` row with
      ``error``.
    - Rows: ``{id, set, donor, recipient, event_id, arm, response, grade, hedged, grader, adopted, adoption,
      adoption_grader, latency_s, usage, judge_usage, backend, prompt_sha, request_sha, note_event, today, error}``
      (design section 6). ``backend`` is ``backend_name`` (default: :func:`_backend_name` of ``backend``),
      ``prompt_sha`` the :func:`prompt_sha` of the planned messages, ``request_sha`` the :func:`request_sha` of the
      call; ``note_event`` and ``today`` are copied from the plan row (None when it has none). ``judge_usage``
      lists every judge call of the row (``[]`` without one).
    - Adoption (sets ``donor_wrong`` and ``control``; None elsewhere and for errors and refusals): ``adoption`` is
      :func:`cutoffmix.bench.keys.text.adoption_label` of the answer and the donor's name; ``adopted`` is False for
      a ``correct`` answer, else True for ``endorsed``, False for ``absent`` and, for ``reported``, None unless the
      judges answer :data:`ADOPTION_JUDGE_SYSTEM`'s question (the majority as in grading; True only for ENDORSE;
      unparseable or tied votes leave it None). ``adoption_grader`` is ``rule`` or ``judge:<keys>``.
    - Safety: ``out_path`` must hold rows of an earlier keys run only (see :func:`_existing`), the plan may not repeat
      an id, and an exclusive lock on ``<out_path>.lock`` is held from before the interrupted last line is cut until
      the last row is appended (another run holding it: ``ValueError``).
    - Resume: a stored row counts as done only when it has provenance, comes from the same backend and answered the
      same request, and has no ``error``; readers keep the last row per id. A row without provenance or from another
      backend anywhere in the file stops the run (``ValueError`` with the counts): those calls are never sent again,
      and a file holds one backend's rows. A planned id whose last row answered another request also stops it,
      unless ``resend_stale``: then those calls are sent again and the newest row counts. Error rows are sent again.
    - Re-grade (only with ``judges``): before any new call, every current row that is ``ungraded`` without error,
      not a refusal and with a non-empty response, and every current adoption row with ``adoption`` ``reported`` and
      ``adopted`` None, is graded again from its stored response (no recipient call) and appended under the same id
      with ``adopted`` recomputed. The new row keeps ``latency_s`` and the provenance; its ``usage`` is None and its
      ``judge_usage`` lists the new judge calls: the recipient's call stays billed in the row it re-grades, so the
      summed cost of all rows counts every paid call once.
    - At most ``concurrency`` items (a call with its judges, or a re-grade) are in flight. ``limit`` caps the calls
      this run sends (re-grades are no calls). ``max_cost`` (USD) stops starting new items once the summed cost of
      every row in ``out_path`` (:func:`row_cost` plus :func:`judge_cost`; earlier rows, repeats and this run's)
      reaches it; items in flight finish with their judge calls. ``RunOutcome.stopped_by`` says whether the cap or
      the limit left planned work undone.
    - ``on_start`` is called once, before the first call, with what the run found (:class:`RunStart`).
    """
    if concurrency < 1:
        raise ValueError(f"concurrency must be >= 1, got {concurrency}")
    if limit is not None and limit < 0:
        raise ValueError(f"limit must be >= 0, got {limit}")
    if max_cost is not None and max_cost < 0:
        raise ValueError(f"max_cost must be >= 0, got {max_cost}")
    for n, row in enumerate(plan, 1):
        absent = [k for k in _PLAN_NEEDS if row.get(k) in (None, "")]
        if absent or not isinstance(row.get("messages"), list):
            raise ValueError(f"plan row {n} has no {', '.join(absent or ['messages'])}: the plan must be a file "
                             "written by 'bench keys plan'")
    by_event = dict(events) if isinstance(events, Mapping) else {e.id: e for e in events}
    specs = {m.key: m for m in models}
    missing_models = sorted({str(r["recipient"]) for r in plan} - set(specs))
    if missing_models:
        raise ValueError(f"the plan names recipient model(s) that are not configured: {missing_models}")
    missing_events = sorted({str(r["event_id"]) for r in plan} - set(by_event))
    if missing_events:
        raise ValueError(f"the plan names {len(missing_events)} event(s) not in the events file, "
                         f"e.g. {missing_events[0]}")
    ids = [str(r["id"]) for r in plan]
    repeated = [i for i, n in Counter(ids).items() if n > 1]
    if repeated:
        raise ValueError(f"the plan repeats {len(repeated)} id(s) (e.g. {repeated[0]}): every planned call must appear "
                         "once (a repeated pair, or plans joined together); build the plan again")
    name = backend_name or _backend_name(backend)
    current = {str(r["id"]): request_sha(r["messages"], max_tokens=max_tokens, temperature=TEMPERATURE,
                                         model=specs[str(r["recipient"])]) for r in plan}
    path = Path(out_path)
    with _locked(path):
        stored = _existing(path)
        _drop_partial_line(path)  # only after the file passed as keys results
        return await _run_locked(
            backend, specs, plan, path, stored, current, by_event, name=name, concurrency=concurrency,
            max_cost=max_cost, limit=limit, judges=list(judges or []), max_tokens=max_tokens, progress=progress,
            resend_stale=resend_stale, on_start=on_start)


def _check_resumable(path: Path, stored: Sequence[Mapping[str, Any]], plan: Sequence[Mapping[str, Any]],
                     current: Mapping[str, str], name: str, last: Mapping[str, Mapping[str, Any]],
                     resend_stale: bool) -> list[str]:
    """The planned ids whose last row answered another request (stale); raises ``ValueError`` (design section 6)
    for rows without provenance or of another backend, and for stale rows unless ``resend_stale``."""
    bare = [r for r in stored if not _has_provenance(r)]
    other = Counter(str(r["backend"]) for r in stored if _has_provenance(r) and r["backend"] != name)
    if bare or other:
        planned_ids = set(current)
        hit = sum(1 for r in stored if str(r["id"]) in planned_ids
                  and (not _has_provenance(r) or r["backend"] != name))
        parts = []
        if bare:
            parts.append(f"{len(bare)} row(s) without provenance (written before version 2 of the design)")
        if other:
            parts.append(f"{sum(other.values())} row(s) from another backend ("
                         + ", ".join(f"{b}: {n}" for b, n in sorted(other.items())) + ")")
        raise ValueError(f"{path} holds {' and '.join(parts)}, {hit} of them for ids of this plan; this run's backend "
                         f"is '{name}', and a results file holds the rows of one backend only, whose calls are never "
                         "sent again: write to a new --out (a mock run to its own file, e.g. "
                         "runs/keys_results_mock.jsonl)")
    stale = [str(p["id"]) for p in plan
             if (r := last.get(str(p["id"]))) is not None and not r.get("error")
             and r.get("request_sha") != current[str(p["id"])]]
    if stale and not resend_stale:
        raise ValueError(f"{path}: {len(stale)} of the {len(plan)} planned calls were answered under another request "
                         "(other messages, max_tokens or model settings: the plan or the configuration changed since "
                         "those rows were written): resume with the original plan and settings, or pass "
                         "--resend-stale to send those calls again (the newest row counts)")
    return stale


async def _run_locked(
    backend: Backend,
    specs: Mapping[str, ModelSpec],
    plan: Sequence[Mapping[str, Any]],
    path: Path,
    stored: list[dict[str, Any]],
    current: Mapping[str, str],
    by_event: Mapping[str, Event],
    *,
    name: str,
    concurrency: int,
    max_cost: float | None,
    limit: int | None,
    judges: list[ModelSpec],
    max_tokens: int,
    progress: ProgressCallback | None,
    resend_stale: bool,
    on_start: Callable[[RunStart], None] | None,
) -> RunOutcome:
    """:func:`run_plan` once the lock is held and the stored rows are read."""
    last: dict[str, dict[str, Any]] = {}
    for row in stored:
        last[str(row["id"])] = row
    stale = set(_check_resumable(path, stored, plan, current, name, last, resend_stale))

    def is_current(row: Mapping[str, Any] | None, item_id: str) -> bool:
        return row is not None and not row.get("error") and row.get("request_sha") == current[item_id]

    done = {i for i in current if is_current(last.get(i), i)}
    pending = [p for p in plan if str(p["id"]) not in done]
    to_send = pending if limit is None else pending[:limit]
    regrade: list[tuple[Mapping[str, Any], dict[str, Any]]] = []
    if judges:
        for p in plan:
            row = last.get(str(p["id"]))
            if row is None or str(p["id"]) not in done:
                continue
            undecided = (row.get("grade") == "ungraded" and row.get("grader") != "refusal"
                         and str(row.get("response") or "").strip() != "")
            reported = (p.get("set") in ADOPTION_SETS and row.get("adoption") == "reported"
                        and row.get("adopted") is None)
            if undecided or reported:
                regrade.append((p, row))
    recipients_before = sum(row_cost(r) for r in stored)
    judges_before = sum(judge_cost(r) for r in stored)
    start = RunStart(
        backend=name, planned=len(plan), answered=len(done), pending=len(pending),
        errors=sum(1 for p in pending if (r := last.get(str(p["id"]))) is not None and r.get("error")),
        stale=sum(1 for p in pending if str(p["id"]) in stale), to_send=len(to_send), regrade=len(regrade),
        cost_usd=recipients_before + judges_before)
    if on_start is not None:
        on_start(start)

    spent = start.cost_usd
    run_recipients = run_judges = 0.0
    total, finished, sent, regraded = len(regrade) + len(to_send), 0, 0, 0
    sem = asyncio.Semaphore(concurrency)
    failures: list[Exception] = []

    async def send(item: Mapping[str, Any]) -> dict[str, Any]:
        item_id = str(item["id"])
        event = by_event[str(item["event_id"])]
        spec = specs[str(item["recipient"])]
        meta = {"event_id": event.id, "probe": PROBE, "model": spec.key, "arm": item["arm"], "set": item["set"],
                "donor": item["donor"], "item": item["id"], "donor_answer": item.get("donor_answer")}
        req = ChatRequest(messages=[dict(m) for m in item["messages"]], max_tokens=max_tokens,
                          temperature=TEMPERATURE, metadata=meta)
        provenance = {"backend": name, "prompt_sha": prompt_sha(item["messages"]), "request_sha": current[item_id],
                      "note_event": item.get("note_event"), "today": item.get("today")}
        start_time = time.perf_counter()
        try:
            result = await backend.chat(spec, req)
        except Exception as e:  # every failure is recorded, the run goes on
            return _row(item, provenance, response="", grade="ungraded", hedged=False, grader="none", adopted=None,
                        adoption=None, adoption_grader=None, latency_s=round(time.perf_counter() - start_time, 4),
                        usage=None, judge_usage=[], error=f"{type(e).__name__}: {e}"[:500])
        latency = result.latency_s if result.latency_s > 0 else time.perf_counter() - start_time
        text = result.text or ""
        meter = _JudgeMeter(backend)
        adopted: bool | None = None
        adoption: str | None = None
        adoption_grader: str | None = None
        if result.finish_reason == "content_filter":  # a policy refusal says nothing about knowledge
            grade, hedged, grader = "ungraded", False, "refusal"
        else:
            grade, hedged, grader = await grade_response(event, PROBE, text, backend=meter, judges=judges or None)
            adopted, adoption, adoption_grader = await _adoption(item, event, text, grade, meter, judges)
        return _row(item, provenance, response=text, grade=grade, hedged=hedged, grader=grader, adopted=adopted,
                    adoption=adoption, adoption_grader=adoption_grader, latency_s=round(latency, 4),
                    usage=result.usage, judge_usage=meter.entries, error=None)

    async def regrade_one(item: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
        event = by_event[str(item["event_id"])]
        text = str(row.get("response") or "")
        meter = _JudgeMeter(backend)
        grade, hedged, grader = str(row.get("grade")), bool(row.get("hedged")), str(row.get("grader"))
        if grade == "ungraded":
            grade, hedged, grader = await grade_response(event, PROBE, text, backend=meter, judges=judges)
        adopted, adoption, adoption_grader = await _adoption(item, event, text, grade, meter, judges)
        return _row(item, row, response=text, grade=grade, hedged=hedged, grader=grader, adopted=adopted,
                    adoption=adoption, adoption_grader=adoption_grader, latency_s=row.get("latency_s"), usage=None,
                    judge_usage=meter.entries, error=None)

    async def worker(make: Callable[[], Awaitable[dict[str, Any]]], counts_as_call: bool) -> None:
        nonlocal finished, spent, sent, regraded, run_recipients, run_judges
        try:
            row = await make()
            append_jsonl(path, row)
            last[str(row["id"])] = row
            run_recipients += row_cost(row)
            run_judges += judge_cost(row)
            spent += _cost(row)
            if counts_as_call:
                sent += 1
            else:
                regraded += 1
            finished += 1
            if progress is not None:
                progress(finished, total, row)
        except Exception as e:  # re-raised after the items in flight finish
            failures.append(e)
        finally:
            sem.release()

    async def drain(work: Sequence[tuple[Callable[[], Awaitable[dict[str, Any]]], bool]]) -> int:
        """Start the items in order, at most ``concurrency`` at a time, until the cap or a failure stops it; return
        how many were started (every started item finishes)."""
        started = 0
        tasks: set[asyncio.Task[None]] = set()
        try:
            for make, counts_as_call in work:
                await sem.acquire()  # bounds the live tasks, not only the calls in flight
                if failures or (max_cost is not None and spent >= max_cost):
                    sem.release()
                    break
                task = asyncio.create_task(worker(make, counts_as_call))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
                started += 1
            await asyncio.gather(*list(tasks))
        finally:
            for task in list(tasks):
                task.cancel()
        return started

    # the re-grades first ("before sending new calls"), then the calls
    regrade_started = await drain([((lambda p=p, r=r: regrade_one(p, r)), False) for p, r in regrade])
    if failures:
        raise failures[0]
    send_started = await drain([((lambda p=p: send(p)), True) for p in to_send])
    if failures:
        raise failures[0]
    stopped_by: Literal["max_cost", "limit"] | None = None
    if regrade_started < len(regrade) or send_started < len(to_send):
        stopped_by = "max_cost"
    elif len(to_send) < len(pending):
        stopped_by = "limit"
    order = [str(p["id"]) for p in plan]
    rows = [last[i] for i in order if i in last and last[i].get("request_sha") == current[i]]
    return RunOutcome(
        rows=rows, start=start, sent=sent, regraded=regraded,
        not_sent=sum(1 for i in current if not is_current(last.get(i), i)),
        not_regraded=len(regrade) - regrade_started, stopped_by=stopped_by, cost_usd=spent,
        recipients_usd=recipients_before + run_recipients, judges_usd=judges_before + run_judges,
        run_cost_usd=run_recipients + run_judges)
