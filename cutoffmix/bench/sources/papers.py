"""Probe events from recent AI/ML research papers (topic ``ai_ml``): Hugging Face daily papers select, arXiv tells.

The routing index's ``ai_ml`` cell needs dated events that a model either knows or does not. A paper's *first
author* is such a fact: it cannot be guessed from the title, it is fixed on the day the paper appears, and arXiv
metadata makes it rule-gradable. The pipeline:

1. **Selection** (:func:`collect_month`): the Hugging Face daily-papers listing
   (``https://huggingface.co/api/daily_papers?date=YYYY-MM-DD``) gives, per day, the community's picks with
   upvotes. Over a month the lists are merged (deduplicated by arXiv id) and the ``top_n`` by upvotes are kept.
   The listing is used only to choose papers; nothing of it ends up in a probe.
2. **Ground truth** (:func:`fetch_arxiv`): the arXiv API (``https://export.arxiv.org/api/query?id_list=...``)
   gives title, authors in order, the ``published`` date of version 1 and the primary category. Every id is asked
   for twice, as ``<id>v1`` (the paper as it appeared on the event date) and bare (the latest version, the one the
   paper is cited under); a paper whose title or first author changed between the two is dropped, so a probe never
   depends on which version a model saw. Papers are filed under the month of the arXiv v1 date, never under the
   Hugging Face date.
3. **Events** (:func:`build_paper_events`): dense months get ``per_month`` papers, about ``interp_share`` of them
   from the interpretability pool (:func:`is_interp`); sparser *anchor* months earlier in the range take fewer.
   Each paper becomes one :class:`~cutoffmix.types.Event` (category ``paper``) asking for the first author, with a
   dated recall variant and a 4-way MCQ over first authors of nearby papers. About ``controls_share`` of the set
   are **fake papers** (category ``fake_paper``, ``control=True``): titles spliced from the subtitles of two real
   titles of different months (never a real system name, never a probe paper's title), which measure confabulation
   (any named author is wrong; only an abstention is right).

Etiquette: one request at a time, the project User-Agent (:data:`USER_AGENT`), ``Retry-After`` honoured,
Hugging Face requests at least :data:`HF_PACE_S` apart, arXiv requests at least :data:`ARXIV_PACE_S` apart
(the arXiv API's "one request every three seconds") with up to :data:`ARXIV_BATCH` ids per ``id_list``.
Every raw response is cached under ``cache_dir`` (default :func:`default_cache_dir`), so reruns cost nothing.
Downloaded data is untrusted: every field is checked before use and malformed items are skipped.

Licence notes: arXiv metadata is used under the arXiv API terms of use ("Thank you to arXiv for use of its open
access interoperability"); the Hugging Face daily-papers listing is a public page used only for selection.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import difflib
import hashlib
import json
import logging
import os
import random
import re
import tempfile
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from cutoffmix.bench import sources
from cutoffmix.bench.probegen import LETTERS, ascii_fold, date_tokens, month_year, stable_hash, validate_event
from cutoffmix.bench.sources import SourceError, polite_request
from cutoffmix.bench.sources.wikidata import USER_AGENT, make_client
from cutoffmix.timeutil import month_index, month_label
from cutoffmix.types import Event

log = logging.getLogger(__name__)

__all__ = [
    "ARXIV_API_URL", "ARXIV_BATCH", "ARXIV_PACE_S", "CONTROL_EXPECTED", "HF_DAILY_URL", "HF_PACE_S",
    "INTERP_FALSE_POSITIVES", "ArxivPaper", "Candidate", "HFPaper", "MonthPlan", "author_aliases",
    "build_paper_events", "collect_month", "default_cache_dir", "fake_title", "fetch_arxiv", "fetch_hf_daily",
    "hf_paper", "is_interp", "last_available_day", "normalize_arxiv_id", "paper_event", "parse_arxiv_atom",
    "plan_months", "plausible_person", "select_month", "split_name", "surname_key",
]

HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
ARXIV_API_URL = "https://export.arxiv.org/api/query"
HF_PACE_S = 1.0  # polite gap between Hugging Face requests
ARXIV_PACE_S = 3.0  # arXiv API terms: at most one request every three seconds
ARXIV_BATCH = 100  # ids per id_list request
DEFAULT_CANDIDATES = 300  # Hugging Face top-N per month before filtering (deep, so the interp pool is not empty)
CACHE_ENV = "CUTOFFMIX_PAPERS_CACHE"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
MAX_BODY = 50 * 1024 * 1024  # a response larger than this is not parsed

# Controls have no right answer: ``expected_current``/``expected_recall`` stay empty (``validate_event`` allows that for
# control events) and the judges read the ``fact`` ("control: no such arXiv paper exists; ...") as the ground truth.
CONTROL_EXPECTED = ""
_MAX_AUTHORS_IN_FACT = 12

# ------------------------------------------------------------------------------------------------- cache & pacing

_LAST_REQUEST: dict[str, float] = {}


def default_cache_dir() -> Path:
    """``$CUTOFFMIX_PAPERS_CACHE`` if set, else ``~/.cache/cutoffmix/papers``."""
    env = os.environ.get(CACHE_ENV)
    return Path(env) if env else Path.home() / ".cache" / "cutoffmix" / "papers"


def _paced(host: str, gap: float) -> None:
    """Wait (through :func:`cutoffmix.bench.sources.pause`) until ``gap`` seconds have passed since the last request
    to ``host`` in this process."""
    last = _LAST_REQUEST.get(host)
    if last is not None:
        wait = last + gap - time.monotonic()
        if wait > 0:
            sources.pause(wait)
    _LAST_REQUEST[host] = time.monotonic()


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _load_json(path: Path) -> Any:
    """Parsed JSON from a cache file, or ``None`` when the file is missing or corrupt (it is then refetched)."""
    try:
        return json.loads(path.read_bytes())
    except (OSError, ValueError):
        return None


def _safe_name(arxiv_id: str) -> str:
    return arxiv_id.replace("/", "_")


# ------------------------------------------------------------------------------------------------- small parsers

_NEW_ID = re.compile(r"^\d{4}\.\d{4,5}$")
_OLD_ID = re.compile(r"^[a-z][a-z\-]*(?:\.[A-Z]{2})?/\d{7}$")
_ID_PREFIX = re.compile(r"^(?:https?://(?:export\.)?arxiv\.org/(?:abs|pdf)/|arxiv:\s*)", re.IGNORECASE)


def normalize_arxiv_id(value: Any) -> str | None:
    """``'2610.20399v2'``, ``'http://arxiv.org/abs/2610.20399v1'``, ``'arXiv:2610.20399'`` -> ``'2610.20399'``;
    ``None`` for anything that is not an arXiv identifier (e.g. the API's error entries)."""
    if not isinstance(value, str):
        return None
    s = _ID_PREFIX.sub("", value.strip())
    s = re.sub(r"\.pdf$", "", s)
    s = re.sub(r"v\d+$", "", s)
    return s if _NEW_ID.match(s) or _OLD_ID.match(s) else None


def _clean_text(value: Any) -> str:
    """Whitespace-normalised string of a JSON/XML text value (non-strings -> ``''``)."""
    return " ".join(str(value).split()) if isinstance(value, str) else ""


def _iso_day(value: Any) -> str:
    """``'2026-10-14T16:03:01Z'`` -> ``'2026-10-14'``; ``''`` if not a date."""
    text = _clean_text(value)[:10]
    try:
        return _dt.date.fromisoformat(text).isoformat()
    except ValueError:
        return ""


def _text(node: ET.Element | None) -> str:
    return node.text or "" if node is not None else ""


def _as_date(value: _dt.date | str) -> _dt.date:
    return value if isinstance(value, _dt.date) else _dt.date.fromisoformat(str(value).strip()[:10])


# ------------------------------------------------------------------------------------------------- Hugging Face


@dataclass(frozen=True)
class HFPaper:
    """One paper of a Hugging Face daily list (selection data only)."""

    id: str
    title: str
    upvotes: int
    authors: tuple[str, ...]
    published_at: str  # Hugging Face's date for the paper (informational; the event date comes from arXiv)
    summary: str
    hf_date: str  # the daily list it was first seen in


def fetch_hf_daily(date: _dt.date | str, *, client: httpx.Client | None = None, cache_dir: Path | str | None = None,
                   retries: int = 4) -> list[Any]:
    """Raw items of the Hugging Face daily-papers list for ``date`` (an empty list for days without a list).

    Cached as ``<cache_dir>/hf/<date>.json`` (the raw body); a cached file is never refetched. Requests are paced
    by :data:`HF_PACE_S` and retried by :func:`~cutoffmix.bench.sources.polite_request`.
    """
    day = _as_date(date)
    path = Path(cache_dir or default_cache_dir()) / "hf" / f"{day.isoformat()}.json"
    cached = _load_json(path)
    if isinstance(cached, list):
        return cached
    own = client is None
    http = client or make_client(timeout=60.0)
    try:
        _paced("huggingface.co", HF_PACE_S)
        resp = polite_request(http, "GET", HF_DAILY_URL, retries=retries, params={"date": day.isoformat()},
                              headers={"Accept": "application/json", "User-Agent": USER_AGENT})
    finally:
        if own:
            http.close()
    if len(resp.content) > MAX_BODY:
        raise SourceError(f"HF daily papers {day}: response too large ({len(resp.content)} bytes)")
    try:
        data = json.loads(resp.content)
    except ValueError as e:
        raise SourceError(f"HF daily papers {day}: invalid JSON: {e}", status=resp.status_code) from e
    if not isinstance(data, list):
        raise SourceError(f"HF daily papers {day}: expected a JSON list, got {type(data).__name__}")
    _write_atomic(path, resp.content)
    return data


def hf_paper(item: Any, hf_date: str) -> HFPaper | None:
    """Defensive normalisation of one daily-papers item; ``None`` when it has no usable arXiv id or title."""
    if not isinstance(item, Mapping):
        return None
    paper = item.get("paper")
    if not isinstance(paper, Mapping):
        return None
    pid = normalize_arxiv_id(paper.get("id"))
    title = _clean_text(paper.get("title")) or _clean_text(item.get("title"))
    if not pid or not title:
        return None
    raw_up = paper.get("upvotes")
    upvotes = int(raw_up) if isinstance(raw_up, (int, float)) and not isinstance(raw_up, bool) else 0
    names: list[str] = []
    raw_authors = paper.get("authors")
    for a in raw_authors if isinstance(raw_authors, list) else []:
        name = _clean_text(a.get("name")) if isinstance(a, Mapping) else _clean_text(a)
        if name:
            names.append(name)
    summary = _clean_text(paper.get("summary")) or _clean_text(item.get("summary"))
    return HFPaper(id=pid, title=title, upvotes=max(0, upvotes), authors=tuple(names),
                   published_at=_iso_day(paper.get("publishedAt")), summary=summary[:2000], hf_date=hf_date)


def last_available_day(now: _dt.datetime | None = None) -> _dt.date:
    """Yesterday in UTC: Hugging Face answers a request for the current UTC day with HTTP 400."""
    now = now or _dt.datetime.now(_dt.timezone.utc)
    return now.astimezone(_dt.timezone.utc).date() - _dt.timedelta(days=1)


def _month_days(year: int, month: int, today: _dt.date) -> list[_dt.date]:
    first = _dt.date(year, month, 1)
    nxt = _dt.date(year + (month == 12), month % 12 + 1, 1)
    return [first + _dt.timedelta(days=i) for i in range((nxt - first).days) if first + _dt.timedelta(days=i) <= today]


def collect_month(year: int, month: int, top_n: int, *, client: httpx.Client | None = None,
                  cache_dir: Path | str | None = None, today: _dt.date | None = None, retries: int = 4,
                  errors: list[str] | None = None) -> list[HFPaper]:
    """The ``top_n`` papers by upvotes over all daily lists of the month, one entry per arXiv id.

    A paper listed on several days keeps its highest upvote count and the first day it appeared on. Days after
    ``today`` (default: yesterday UTC, the last list Hugging Face serves) are not requested. A day whose request fails for good is logged (and appended to ``errors`` when
    given) and skipped, so one bad day does not abort a long build; with the cache, a rerun fills the gap.
    """
    if top_n < 1:
        raise ValueError(f"top_n must be >= 1, got {top_n}")
    today = today or last_available_day()
    own = client is None
    http = client or make_client(timeout=60.0)
    best: dict[str, HFPaper] = {}
    try:
        for day in _month_days(year, month, today):
            try:
                items = fetch_hf_daily(day, client=http, cache_dir=cache_dir, retries=retries)
            except SourceError as e:
                log.warning("HF daily papers %s skipped: %s", day, e)
                if errors is not None:
                    errors.append(f"{day}: {e}")
                continue
            for item in items:
                p = hf_paper(item, day.isoformat())
                if p is None:
                    continue
                cur = best.get(p.id)
                if cur is None:
                    best[p.id] = p
                elif p.upvotes > cur.upvotes:
                    best[p.id] = dataclasses.replace(p, hf_date=cur.hf_date)
    finally:
        if own:
            http.close()
    ranked = sorted(best.values(), key=lambda p: (-p.upvotes, p.id))
    log.info("HF %04d-%02d: %d papers, keeping top %d", year, month, len(ranked), min(top_n, len(ranked)))
    return ranked[:top_n]


# ------------------------------------------------------------------------------------------------- arXiv


@dataclass(frozen=True)
class ArxivPaper:
    """arXiv metadata of one paper (the ground truth)."""

    id: str
    title: str
    authors: tuple[str, ...]  # in arXiv order; the first is the first author
    published: str  # YYYY-MM-DD of version 1
    primary_category: str
    summary: str
    version: int = 0  # the version the entry describes (the ``vN`` of the Atom id); 0 when unknown


_VERSION_SUFFIX = re.compile(r"v(\d+)$")


def parse_arxiv_atom(xml_data: bytes | str) -> dict[str, ArxivPaper]:
    """Entries of an arXiv API Atom feed keyed by arXiv id (version stripped).

    Titles and names are whitespace-normalised, authors keep their order, ``published`` is the v1 date, the primary
    category falls back to the first ``category`` term, ``version`` is the ``vN`` of the entry id (the version whose
    title and authors the entry describes). Entries without a valid id, title, author or date (the API answers
    unknown ids with an ``Error`` entry) are skipped. Malformed XML raises :class:`SourceError`.
    """
    if isinstance(xml_data, (bytes, bytearray)) and len(xml_data) > MAX_BODY:
        raise SourceError(f"arXiv API: response too large ({len(xml_data)} bytes)")
    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError as e:
        raise SourceError(f"arXiv API: malformed Atom XML: {e}") from e
    out: dict[str, ArxivPaper] = {}
    for entry in root.iterfind("atom:entry", ATOM_NS):
        raw_id = _clean_text(_text(entry.find("atom:id", ATOM_NS)))
        pid = normalize_arxiv_id(raw_id)
        vm = _VERSION_SUFFIX.search(raw_id)
        version = int(vm.group(1)) if vm else 0
        title = _clean_text(_text(entry.find("atom:title", ATOM_NS)))
        authors = tuple(n for n in (_clean_text(_text(a.find("atom:name", ATOM_NS)))
                                    for a in entry.iterfind("atom:author", ATOM_NS)) if n)
        published = _iso_day(_text(entry.find("atom:published", ATOM_NS)))
        if not pid or not title or not authors or not published:
            continue
        primary = entry.find("arxiv:primary_category", ATOM_NS)
        category = _clean_text(primary.get("term")) if primary is not None else ""
        if not category:
            first_cat = entry.find("atom:category", ATOM_NS)
            category = _clean_text(first_cat.get("term")) if first_cat is not None else ""
        summary = _clean_text(_text(entry.find("atom:summary", ATOM_NS)))[:2000]
        out.setdefault(pid, ArxivPaper(id=pid, title=title, authors=authors, published=published,
                                       primary_category=category, summary=summary, version=version))
    return out


def _cached_arxiv(path: Path) -> ArxivPaper | None:
    data = _load_json(path)
    if not isinstance(data, Mapping):
        return None
    try:
        raw_version = data.get("version", 0)
        paper = ArxivPaper(id=str(data["id"]), title=str(data["title"]), authors=tuple(str(a) for a in data["authors"]),
                           published=str(data["published"]), primary_category=str(data.get("primary_category", "")),
                           summary=str(data.get("summary", "")),
                           version=int(raw_version) if isinstance(raw_version, int) and raw_version >= 0 else 0)
    except (KeyError, TypeError):
        return None
    return paper if normalize_arxiv_id(paper.id) == paper.id and paper.authors and _iso_day(paper.published) else None


def fetch_arxiv(ids: Iterable[str], *, client: httpx.Client | None = None, cache_dir: Path | str | None = None,
                retries: int = 4, batch: int = ARXIV_BATCH, version: int | None = None) -> dict[str, ArxivPaper]:
    """arXiv metadata for ``ids`` (any form :func:`normalize_arxiv_id` accepts), keyed by normalised id.

    ``version=None`` asks for the bare ids, which arXiv answers with the latest version of each paper (cached as
    ``<cache_dir>/arxiv/<id>.json``); ``version=1`` asks for ``<id>v1``, the paper as it appeared on its first day
    (cached as ``arxiv/<id>v1.json``). Ids already in the per-id cache are not requested again; the rest go to the
    API in ``id_list`` batches of at most ``batch`` ids, at least :data:`ARXIV_PACE_S` apart, each raw feed stored
    under ``<cache_dir>/arxiv/raw/``. Ids the API does not know (or whose answer describes another version than the
    one asked for) are logged and left out.
    """
    if not 1 <= batch <= ARXIV_BATCH:
        raise ValueError(f"batch must be in 1..{ARXIV_BATCH}, got {batch}")
    if version is not None and version < 1:
        raise ValueError(f"version must be >= 1 or None, got {version}")
    suffix = f"v{version}" if version else ""
    cache = Path(cache_dir or default_cache_dir()) / "arxiv"
    wanted = sorted({n for n in (normalize_arxiv_id(i) for i in ids) if n})
    out: dict[str, ArxivPaper] = {}
    missing: list[str] = []
    for pid in wanted:
        paper = _cached_arxiv(cache / f"{_safe_name(pid)}{suffix}.json")
        if paper is not None:
            out[pid] = paper
        else:
            missing.append(pid)
    if not missing:
        return out
    own = client is None
    http = client or make_client(timeout=90.0)
    try:
        for i in range(0, len(missing), batch):
            chunk = missing[i:i + batch]
            id_list = ",".join(f"{pid}{suffix}" for pid in chunk)
            _paced("export.arxiv.org", ARXIV_PACE_S)
            resp = polite_request(http, "GET", ARXIV_API_URL, retries=retries,
                                  params={"id_list": id_list, "max_results": str(len(chunk))},
                                  headers={"Accept": "application/atom+xml", "User-Agent": USER_AGENT})
            digest = hashlib.sha1(id_list.encode("utf-8")).hexdigest()[:16]
            _write_atomic(cache / "raw" / f"{digest}.xml", resp.content)
            parsed = parse_arxiv_atom(resp.content)
            for pid in chunk:
                paper = parsed.get(pid)
                if paper is None:
                    log.info("arXiv %s%s: no entry in the API answer", pid, suffix)
                    continue
                if version is not None and paper.version not in (0, version):
                    log.warning("arXiv %s%s: the API answered with version %d; skipped", pid, suffix, paper.version)
                    continue
                out[pid] = paper
                _write_atomic(cache / f"{_safe_name(pid)}{suffix}.json",
                              json.dumps(dataclasses.asdict(paper), ensure_ascii=False).encode("utf-8"))
            log.info("arXiv batch %d-%d%s: %d of %d ids answered", i + 1, i + len(chunk), f" ({suffix})" if suffix
                     else "", len(parsed), len(chunk))
    finally:
        if own:
            http.close()
    return out


# ------------------------------------------------------------------------------------------------- tagging & names

# Specific interpretability terms: one occurrence anywhere (title or summary) tags the paper.
INTERP_SPECIFIC: tuple[str, ...] = (
    "mechanistic interpretability", "sparse autoencoder", "activation patching", "attribution patching", "logit lens",
    "monosemantic", "polysemantic", "representation engineering", "transcoder", "crosscoder", "causal tracing",
    "attribution graph", "circuit discovery", "feature steering", "activation steering", "steering vector",
    "probing classifier", "induction head", "feature attribution", "dictionary learning", "feature visualization",
)
# Generic words: they tag the paper when they are in the title or occur at least three times in the summary (one or
# two incidental "improves interpretability" in an abstract do not make an interpretability paper).
INTERP_GENERIC_STRONG: tuple[str, ...] = ("interpretab", "mechanistic", "superposition")
_GENERIC_SUMMARY_MIN = 3
# "superposition" (also a pre-training trick and a figure of speech) and the acronym SAE (also the automotive autonomy
# levels) need one of these context words somewhere in the text.
_SUPERPOSITION_CONTEXT = ("feature", "neuron", "polysemantic", "monosemantic", "interpretab", "linear", "representation")
_SAE_CONTEXT = ("autoencoder", "sparse", "interpretab", "feature", "latent", "dictionary")
INTERP_WEAK: tuple[str, ...] = ("circuit", "attribution", "probing", "steering", "neuron", "residual stream",
                                "attention head", "latent direction")
INTERP_GENERIC: tuple[str, ...] = ("feature", "activation")  # count only next to a weak family
_SAE = re.compile(r"\bSAEs?\b")
INTERP_STRONG: tuple[str, ...] = INTERP_SPECIFIC + INTERP_GENERIC_STRONG  # the full keyword list, for reference
# Applied domains whose papers call their own models "interpretable" or use an interpretability technique as a tool
# (medical agents, driving evaluation, brain decoding, model compression): with one of these in the title, only a
# specific term in the title itself tags the paper.
INTERP_APPLIED_DOMAINS: tuple[str, ...] = (
    "medical", "clinical", "diagnos\\w*", "driving", "meg", "eeg", "fmri", "brain", "neuroscien\\w*", "cortical",
    "compression", "compressing", "pruning", "quantiz\\w*", "quantis\\w*",
)  # regex fragments over the lower-cased title, matched on word boundaries
_APPLIED = re.compile(r"\b(?:" + "|".join(INTERP_APPLIED_DOMAINS) + r")\b")
# Hand-checked misses of the keyword tag (2026-11-25 build): the words occur, but the paper is not interpretability
# research. Applied in :func:`gather_candidates`; a corrected flag needs no rebuild of the events, only of the meta.
INTERP_FALSE_POSITIVES: dict[str, str] = {
    "2407.13301": "interpretable medical agent (chain of diagnosis)",
    "2501.04003": "VLM evaluation for autonomous driving",
    "2510.11288": "emergent misalignment via in-context learning; activation steering only as prior work",
    "2512.08560": "neuroscience: visual representations in the human brain",
    "2601.21571": "token-level data filtering; sparse autoencoders only as prior work",
    "2602.11008": "model compression; dictionary learning as a technique",
    "2603.12372": "reasoning-length control; steering vector as a technique",
    "2608.01481": "MEG decoding of perceived speech",
    "2608.02585": "test-time latent reasoning; 'interpretable' in the applied sense",
}


def is_interp(title: str, summary: str = "") -> bool:
    """Keyword tag for (mechanistic) interpretability papers.

    True when a specific term (:data:`INTERP_SPECIFIC`: ``mechanistic interpretability``, ``sparse autoencoder``,
    ``activation patching``, ``logit lens``, ``monosemantic``, ``representation engineering``, ``transcoder``,
    ``crosscoder``, ``causal tracing``, ...) occurs anywhere; when ``SAE`` occurs with an autoencoder context word;
    when a generic word (``interpretab``, ``mechanistic``, ``superposition``; the last with a feature/neuron/linear
    context) is in the title or occurs at least three times in the summary; or when two weak
    families occur (``circuit``, ``attribution``, ``probing``, ``steering``, ``neuron``, ``residual stream``,
    ``attention head``; ``feature``/``activation`` only next to one of those) and a specific one of them is in the
    title. A title that names an applied domain (:data:`INTERP_APPLIED_DOMAINS`: medical, driving, MEG/EEG/brain,
    compression, ...) is tagged only by a specific term in the title ("Interpretable MEG Decoding" is not
    interpretability research; "Sparse Autoencoders for Brain Decoding" is).
    """
    title_l, summary_l = title.lower(), summary.lower()
    text_l = f"{title_l} {summary_l}"
    sae_context = any(k in text_l for k in _SAE_CONTEXT)
    if any(k in title_l for k in INTERP_SPECIFIC) or (_SAE.search(title) and sae_context):
        return True
    if _APPLIED.search(title_l):
        return False
    if any(k in text_l for k in INTERP_SPECIFIC):
        return True
    if _SAE.search(f"{title} {summary}") and sae_context:
        return True
    for k in INTERP_GENERIC_STRONG:
        if k == "superposition" and not any(c in text_l for c in _SUPERPOSITION_CONTEXT):
            continue
        if k in title_l or summary_l.count(k) >= _GENERIC_SUMMARY_MIN:
            return True

    def found(k: str, text: str) -> bool:
        return re.search(r"\b" + re.escape(k) + r"\w*", text) is not None

    weak = [k for k in INTERP_WEAK if found(k, text_l)]
    families = weak + [k for k in INTERP_GENERIC if found(k, text_l)]
    return len(families) >= 2 and any(found(k, title_l) for k in weak)


_ORG_TOKENS = frozenset({
    "team", "teams", "lab", "labs", "laboratory", "research", "ai", "agi", "ml", "nlp", "inc", "group", "collaboration",
    "consortium", "et", "al", "university", "institute", "project", "foundation", "corp", "corporation", "llc", "ltd",
    "gmbh", "anonymous", "authors", "contributors", "community", "the", "robotics", "robotic", "seed", "x", "squad",
    "alliance", "initiative", "center", "centre", "studio", "org", "organization", "organisation", "department",
    "company", "technologies", "technology", "tech", "systems", "intelligence", "science", "sciences", "llm", "llms",
    "developers", "committee", "program", "network", "automation", "cloud", "software", "platform", "open", "openai",
    "google", "deepmind", "microsoft", "nvidia", "amazon", "alibaba", "tencent", "baidu", "bytedance", "huawei",
    "xiaomi", "kimi", "qwen", "deepseek", "mistral", "cohere", "anthropic", "moonshot", "zhipu", "minimax", "stepfun",
})
_PARTICLES = frozenset({"van", "von", "de", "der", "den", "da", "di", "du", "la", "le", "del", "della", "al", "bin",
                        "ibn", "dos", "das", "ter", "ten", "zu", "y", "e", "mac", "st."})
# "B.", "J.-P.", "Y": an initial (one letter per hyphen-separated part, dots optional)
_INITIAL_TOKEN = re.compile(r"^[^\W\d_]\.?(?:-[^\W\d_]\.?)*$")


def plausible_person(name: str) -> bool:
    """True for a person's name that the probe can ask for: at least two tokens with letters, no digits, no
    organisation words (``DeepSeek-AI``, ``Kimi Team``, ``Tencent Robotics X``, ``Bytedance Seed`` are not people),
    a surname of at least two letters with a lower-case one (not ``X``, ``AI``), and a given name that is not only
    initials (``B. An``, ``I. Apanasevich`` cannot be rule-graded against a full name)."""
    toks = name.split()
    if len(toks) < 2 or any(ch.isdigit() for ch in name):
        return False
    low = {ascii_fold(t).casefold().strip(".,;:()") for t in toks}
    if low & _ORG_TOKENS or not all(re.search(r"[^\W\d_]", t) for t in toks):
        return False
    given, surname = split_name(name)
    letters = [ch for ch in surname if ch.isalpha()]
    if len(letters) < 2 or not any(ch.islower() for ch in letters):
        return False
    return any(not _INITIAL_TOKEN.match(t) for t in given.split())


def split_name(name: str) -> tuple[str, str]:
    """``'Jan van der Berg'`` -> ``('Jan', 'van der Berg')``; ``'Jintian Zhang'`` -> ``('Jintian', 'Zhang')``."""
    toks = name.split()
    if len(toks) < 2:
        return "", name.strip()
    i = len(toks) - 1
    while i > 1 and toks[i - 1].casefold() in _PARTICLES:
        i -= 1
    return " ".join(toks[:i]), " ".join(toks[i:])


def surname_key(name: str) -> str:
    """Diacritic-free, casefolded surname (last token after particles) for distinctness checks."""
    return ascii_fold(split_name(name)[1]).casefold()


def _name_key(name: str) -> str:
    """Diacritic-free, casefolded, whitespace-normalised full name for equality checks."""
    return " ".join(ascii_fold(name).casefold().split())


def _title_key(title: str) -> str:
    """Diacritic-free, casefolded word tokens of a title (punctuation and spacing ignored)."""
    return " ".join(re.findall(r"\w+", ascii_fold(title).casefold()))


def author_aliases(name: str) -> list[str]:
    """Accepted surface forms of a first author: the full name, ``Surname, Given``, ``G. Surname``, ``Surname G.``,
    ``Surname Given`` (East Asian order), ``First Surname`` when there is a middle name, and the diacritic-free
    variant of each (order-preserving, case-insensitively unique)."""
    given, surname = split_name(name)
    forms = [name]
    if given:
        initial = given[0]
        forms += [f"{surname}, {given}", f"{initial}. {surname}", f"{surname} {initial}.", f"{surname} {given}"]
        if len(given.split()) > 1:
            forms.append(f"{given.split()[0]} {surname}")
    out: list[str] = []
    seen: set[str] = set()
    for v in [*forms, *(ascii_fold(f) for f in forms)]:
        v = " ".join(v.split())
        if len(v) >= 2 and v.casefold() not in seen:
            seen.add(v.casefold())
            out.append(v)
    return out


# ------------------------------------------------------------------------------------------------- planning


@dataclass(frozen=True)
class MonthPlan:
    month: str  # YYYY-MM
    count: int
    dense: bool


def _check_month(value: str, what: str) -> int:
    if not re.fullmatch(r"\d{4}-\d{2}", value or "") or not 1 <= int(value[5:7]) <= 12:
        raise ValueError(f"{what} must be YYYY-MM, got {value!r}")
    return month_index(int(value[:4]), int(value[5:7]))


def plan_months(dense_from: str, dense_to: str, per_month: int, *, anchor_from: str | None = None,
                anchor_to: str | None = None, anchor_every: int = 3, anchor_per_month: int = 6) -> list[MonthPlan]:
    """Months to fill: every month of ``dense_from..dense_to`` with ``per_month`` papers, plus, from the anchor
    range, every ``anchor_every``-th month (starting at ``anchor_from``, skipping dense months) with
    ``anchor_per_month``. Sorted by month."""
    d0, d1 = _check_month(dense_from, "dense_from"), _check_month(dense_to, "dense_to")
    if d1 < d0:
        raise ValueError(f"dense_to {dense_to} is before dense_from {dense_from}")
    if per_month < 1 or anchor_per_month < 0 or anchor_every < 1:
        raise ValueError("per_month and anchor_every must be >= 1 and anchor_per_month >= 0")
    plans = {month_label(x): MonthPlan(month_label(x), per_month, True) for x in range(d0, d1 + 1)}
    if anchor_from and anchor_to and anchor_per_month > 0:
        a0, a1 = _check_month(anchor_from, "anchor_from"), _check_month(anchor_to, "anchor_to")
        if a1 < a0:
            raise ValueError(f"anchor_to {anchor_to} is before anchor_from {anchor_from}")
        for x in range(a0, a1 + 1, anchor_every):
            plans.setdefault(month_label(x), MonthPlan(month_label(x), anchor_per_month, False))
    return [plans[m] for m in sorted(plans)]


# ------------------------------------------------------------------------------------------------- candidates


@dataclass(frozen=True)
class Candidate:
    """A Hugging Face pick joined with its arXiv v1 record (``ax``: the paper as it appeared on the event date) and,
    when fetched, the latest version (``latest``), filed under the arXiv v1 published month."""

    hf: HFPaper
    ax: ArxivPaper
    interp: bool
    latest: ArxivPaper | None = None

    @property
    def month(self) -> str:
        return self.ax.published[:7]

    @property
    def first_author(self) -> str:
        return self.ax.authors[0]

    @property
    def title_words(self) -> int:
        return len(self.ax.title.split())

    @property
    def preferred(self) -> bool:
        """At least three authors and six title words: harder to guess, easier to splice."""
        return len(self.ax.authors) >= 3 and self.title_words >= 6


def _unusable(c: Candidate) -> str | None:
    """Why a candidate cannot become a probe (``None`` if it can)."""
    name = c.first_author
    if not plausible_person(name):
        return f"first author {name!r} is not a person's name the probe can ask for"
    if c.latest is not None:
        if _title_key(c.latest.title) != _title_key(c.ax.title):
            return f"title changed after v1 (latest: {c.latest.title!r})"
        if _name_key(c.latest.authors[0]) != _name_key(name):
            return f"first author changed after v1 (latest: {c.latest.authors[0]!r})"
    tokens = date_tokens(c.ax.title)
    if tokens:
        return f"title contains date tokens {tokens}"
    title_l = c.ax.title.casefold()
    for alias in author_aliases(name):
        if len(alias) >= 3 and re.search(r"(?<!\w)" + re.escape(alias.casefold()) + r"(?!\w)", title_l):
            return f"author name {alias!r} appears in the title"
    return None


def gather_candidates(plans: Sequence[MonthPlan], *, candidates_per_month: int = DEFAULT_CANDIDATES,
                      client: httpx.Client | None = None, cache_dir: Path | str | None = None,
                      today: _dt.date | None = None, retries: int = 4,
                      errors: list[str] | None = None) -> dict[str, list[Candidate]]:
    """Hugging Face top-N of every planned month joined with arXiv metadata, filed under the arXiv v1 published
    month (so a paper listed by Hugging Face in one month but published in another moves or drops out). Every id
    is fetched as v1 (the probe's title, first author and date) and as the latest version (for the divergence
    check in :func:`_unusable` and the interp tag, whose summary is the fullest). Unusable candidates are logged
    and left out; the interp tag is :func:`is_interp` minus :data:`INTERP_FALSE_POSITIVES`."""
    own = client is None
    http = client or make_client(timeout=90.0)
    try:
        hf_by_id: dict[str, HFPaper] = {}
        for plan in plans:
            for p in collect_month(int(plan.month[:4]), int(plan.month[5:7]), candidates_per_month, client=http,
                                   cache_dir=cache_dir, today=today, retries=retries, errors=errors):
                cur = hf_by_id.get(p.id)
                if cur is None or p.upvotes > cur.upvotes:
                    hf_by_id[p.id] = p if cur is None else dataclasses.replace(p, hf_date=min(p.hf_date, cur.hf_date))
        latest = fetch_arxiv(hf_by_id, client=http, cache_dir=cache_dir, retries=retries)
        first = {pid: p for pid, p in latest.items() if p.version == 1}  # the latest version is v1: nothing to ask
        first.update(fetch_arxiv([pid for pid in latest if pid not in first], client=http, cache_dir=cache_dir,
                                 retries=retries, version=1))
    finally:
        if own:
            http.close()
    planned = {p.month for p in plans}
    by_month: dict[str, list[Candidate]] = {m: [] for m in planned}
    for pid in sorted(hf_by_id):
        ax, newest = first.get(pid), latest.get(pid)
        if ax is None or newest is None:
            continue
        interp = is_interp(newest.title, newest.summary or ax.summary or hf_by_id[pid].summary)
        if interp and pid in INTERP_FALSE_POSITIVES:
            log.info("%s: interp tag removed by hand (%s)", pid, INTERP_FALSE_POSITIVES[pid])
            interp = False
        c = Candidate(hf=hf_by_id[pid], ax=ax, interp=interp, latest=newest)
        if c.month not in planned:
            log.info("%s: arXiv month %s is not a planned month (HF listed it on %s)", pid, c.month, c.hf.hf_date)
            continue
        why = _unusable(c)
        if why:
            log.info("%s skipped: %s", pid, why)
            continue
        by_month[c.month].append(c)
    return by_month


def _rank(c: Candidate) -> tuple[int, int, str]:
    return (0 if c.preferred else 1, -c.hf.upvotes, c.ax.id)


def select_month(cands: Sequence[Candidate], n: int, interp_share: float) -> list[Candidate]:
    """``n`` candidates: about ``interp_share`` of them interp-tagged when the pool allows, the rest the top-upvoted
    others (preferred shape first, then upvotes, then id); interp papers fill in when the others run out."""
    if not 0.0 <= interp_share <= 1.0:
        raise ValueError(f"interp_share must be in 0..1, got {interp_share}")
    interp = sorted((c for c in cands if c.interp), key=_rank)
    rest = sorted((c for c in cands if not c.interp), key=_rank)
    k = min(len(interp), round(n * interp_share))
    chosen = interp[:k] + rest[:max(0, n - k)]
    if len(chosen) < n:
        chosen += interp[k:k + n - len(chosen)]
    return sorted(chosen, key=_rank)


# ------------------------------------------------------------------------------------------------- events


def _distractors(rng: random.Random, c: Candidate, by_month: Mapping[str, Sequence[Candidate]]) -> list[str]:
    """Three first authors of other papers, from the same or adjacent months first (then wider), with surnames
    distinct from the answer's and from each other; never a co-author of the paper itself."""
    answer_key = surname_key(c.first_author)
    coauthors = {_name_key(a) for a in c.ax.authors} | {_name_key(a) for a in (c.latest.authors if c.latest else ())}
    here = month_index(int(c.month[:4]), int(c.month[5:7]))
    for radius in (1, 2, 3, 6, 12, 60):
        pool: dict[str, str] = {}
        for month in sorted(by_month):
            if abs(month_index(int(month[:4]), int(month[5:7])) - here) > radius:
                continue
            for other in by_month[month]:
                if (other.ax.id != c.ax.id and surname_key(other.first_author) != answer_key
                        and _name_key(other.first_author) not in coauthors):
                    pool.setdefault(other.first_author.casefold(), other.first_author)
        names = sorted(pool.values())
        rng.shuffle(names)
        picked: list[str] = []
        keys = {answer_key}
        for name in names:
            key = surname_key(name)
            if key not in keys:
                keys.add(key)
                picked.append(name)
            if len(picked) == 3:
                return picked
    return []


def _question(title: str) -> str:
    return f"Who is the first author of the arXiv paper titled '{title}'?"


def _recall_question(title: str, date: str) -> str:
    return f"Who is the first author of the arXiv paper titled '{title}' (arXiv, {month_year(date)})?"


def paper_event(c: Candidate, distractors: Sequence[str], seed: int = 0) -> Event:
    """The probe event of one paper. ``distractors`` are three other first authors (empty -> no MCQ); the MCQ
    order is shuffled with ``seed`` and the event id."""
    name, title = c.first_author, c.ax.title
    authors = list(c.ax.authors)
    listed = ", ".join(authors[:_MAX_AUTHORS_IN_FACT]) + (
        f" et al. ({len(authors)} authors)" if len(authors) > _MAX_AUTHORS_IN_FACT else "")
    choices: list[str] = []
    answer = ""
    if len(distractors) == 3:
        order = [name, *distractors]
        random.Random(stable_hash(f"{seed}|paper:{c.ax.id}|mcq")).shuffle(order)
        choices = [f"{letter}) {text}" for letter, text in zip(LETTERS, order)]
        answer = LETTERS[order.index(name)]
    return Event(
        id=f"paper:{c.ax.id}", date=c.ax.published, month=c.month, topic="ai_ml", category="paper", subject=title,
        fact=f"first author {name}; authors: {listed}; arXiv {c.ax.id} ({c.month})", region="International",
        predictability="low", question_current=_question(title), expected_current=name,
        question_recall=_recall_question(title, c.ax.published), expected_recall=name,
        mcq_question=_question(title) if choices else "", mcq_choices=choices, mcq_answer=answer,
        aliases=author_aliases(name), reject_aliases=list(distractors), source=f"https://arxiv.org/abs/{c.ax.id}",
        origin="arxiv_hf", control=False,
    )


# Function words that cannot end the first part or start the second part of a spliced title: the join would be
# mid-phrase ("Boosting mathematical capability of | Large Solve Realistic ...").
_SPLICE_STOP = frozenset({
    "a", "an", "the", "of", "for", "with", "in", "on", "to", "and", "or", "via", "through", "from", "by", "as", "at",
    "into", "towards", "toward", "without", "under", "over", "between", "across", "beyond", "versus", "vs", "that",
    "which", "than", "is", "are", "be", "can", "do", "does", "should", "when", "where", "how", "why", "what", "not",
    "no", "more", "less", "its", "their", "your", "our", "all", "any", "both", "using", "about", "per", "but", "if",
    "so", "then", "up", "out", "off", "down", "while", "whether", "also", "only", "even", "just", "still", "yet",
    "it", "we", "you", "they", "i", "need", "needs", "make", "makes", "learn", "learns", "help", "helps", "let",
})
_QUESTION_START = frozenset({"can", "do", "does", "is", "are", "how", "why", "what", "when", "where", "which",
                             "should", "will", "who", "whose"})


def _subtitle(title: str) -> str | None:
    """The part of ``title`` after its first colon (the system name before it identifies the paper), or the whole
    title when it has no colon; ``None`` when a colon remains or the result is too short to split."""
    part = " ".join((title.split(":", 1)[1] if ":" in title else title).split())
    return part if part and ":" not in part and len(part.split()) >= 6 else None


def _joinable(head: Sequence[str], tail: Sequence[str]) -> bool:
    """A grammatical join: both parts have three words or more, the seam is not punctuation or a function word on
    either side, no question mark is left inside, and a trailing question mark only after a question head."""
    if len(head) < 3 or len(tail) < 3:
        return False
    last, first = head[-1], tail[0]
    if not re.search(r"\w$", last) or not re.match(r"^\(?[\w\"']", first) or re.search(r"[:;,]$", first):
        return False
    if last.casefold() in _SPLICE_STOP or first.casefold().strip("(\"'") in _SPLICE_STOP:
        return False
    if any("?" in w for w in [*head, *tail[:-1]]):
        return False
    return not tail[-1].endswith("?") or head[0].casefold() in _QUESTION_START


def fake_title(rng: random.Random, by_month: Mapping[str, Sequence[Candidate]], real_titles: Iterable[str],
               *, month: str | None = None, attempts: int = 200,
               exclude: Iterable[str] = ()) -> tuple[str, Candidate, Candidate] | None:
    """A plausible title that does not exist: the first half of one real title's subtitle (from ``month`` if
    given) joined with the second half of another's from a different month. Only the part after a title's colon is
    used, so the result never opens with a real system name; donors whose arXiv id is in ``exclude`` (the probe
    papers of the same build) are not used. Both halves have at least three words and join grammatically
    (:func:`_joinable`); the result must not equal or nearly equal (similarity >= 0.9) any real title or either
    donor's subtitle and must not contain date tokens. Returns ``(title, donor_a, donor_b)`` or ``None`` when
    ``attempts`` draws found nothing."""
    keys = {_title_key(t) for t in real_titles}
    excluded = set(exclude)
    pools = {m: [c for c in cs if c.ax.id not in excluded and _subtitle(c.ax.title)] for m, cs in by_month.items()}
    months = sorted(m for m in pools if pools[m])
    if len(months) < 2:
        return None
    for _ in range(attempts):
        month_a = month if month is not None and pools.get(month) else rng.choice(months)
        month_b = rng.choice([m for m in months if m != month_a])
        a, b = rng.choice(pools[month_a]), rng.choice(pools[month_b])
        sub_a, sub_b = _subtitle(a.ax.title) or "", _subtitle(b.ax.title) or ""
        wa, wb = sub_a.split(), sub_b.split()
        na, nb = len(wa), len(wb)
        # split at the middle, or one word off it when the seam would not join
        joins = ((wa[:k], wb[j:]) for k in (na // 2, na // 2 + 1, na // 2 - 1) for j in (nb // 2, nb // 2 - 1,
                                                                                         nb // 2 + 1))
        head, tail = next(((h, t) for h, t in joins if _joinable(h, t)), ([], []))
        if not head:
            continue
        title = " ".join(head + tail)
        key = _title_key(title)
        if key in keys or date_tokens(title):
            continue
        near = [k for k in keys if abs(len(k) - len(key)) < 40] + [_title_key(sub_a), _title_key(sub_b)]
        if any(difflib.SequenceMatcher(None, key, k).ratio() >= 0.9 for k in near):
            continue
        return title, a, b
    return None


def control_event(n: int, title: str, donor: Candidate) -> Event:
    """The fake-paper control ``fakepaper:<n>``, dated to the first donor's arXiv date; no MCQ, no aliases, no
    expected answer (:data:`CONTROL_EXPECTED`; the ``fact`` is what the judges read)."""
    return Event(
        id=f"fakepaper:{n}", date=donor.ax.published, month=donor.month, topic="ai_ml", category="fake_paper",
        subject=title, fact="control: no such arXiv paper exists; the title splices the subtitles of two real papers",
        region="International", predictability="low", question_current=_question(title),
        expected_current=CONTROL_EXPECTED, question_recall=_recall_question(title, donor.ax.published),
        expected_recall=CONTROL_EXPECTED, origin="arxiv_hf", control=True,
    )


def _meta(event: Event, c: Candidate | None, *, distractors: Sequence[str] = (),
          donors: Sequence[Candidate] = ()) -> dict[str, Any]:
    row: dict[str, Any] = {"event_id": event.id, "month": event.month, "control": event.control}
    if c is not None:
        row.update({
            "arxiv_id": c.ax.id, "interp": c.interp, "upvotes": c.hf.upvotes, "n_authors": len(c.ax.authors),
            "first_author": c.first_author, "authors": list(c.ax.authors), "hf_authors": list(c.hf.authors),
            "primary_category": c.ax.primary_category, "published": c.ax.published, "hf_date": c.hf.hf_date,
            "hf_published_at": c.hf.published_at, "title_words": c.title_words, "preferred": c.preferred,
            "distractors": list(distractors), "latest_version": c.latest.version if c.latest else None,
        })
    if donors:
        row["donors"] = [d.ax.id for d in donors]
        row["donor_titles"] = [d.ax.title for d in donors]
    return row


def build_paper_events(
    dense_from: str = "2025-10",
    dense_to: str = "2026-11",
    *,
    per_month: int = 12,
    anchor_from: str | None = "2024-01",
    anchor_to: str | None = "2025-09",
    anchor_every: int = 3,
    anchor_per_month: int = 6,
    interp_share: float = 0.5,
    controls_share: float = 0.1,
    seed: int = 0,
    candidates_per_month: int = DEFAULT_CANDIDATES,
    client: httpx.Client | None = None,
    cache_dir: Path | str | None = None,
    today: _dt.date | None = None,
    retries: int = 4,
    errors: list[str] | None = None,
) -> tuple[list[Event], list[dict[str, Any]]]:
    """Build the paper probe set: ``(events, meta)``.

    Dense months ``dense_from..dense_to`` get ``per_month`` papers each, anchor months (every ``anchor_every``-th
    month of ``anchor_from..anchor_to`` that is not dense) ``anchor_per_month``. Per month the Hugging Face top
    ``candidates_per_month`` by upvotes are joined with arXiv metadata and filed under the arXiv published month;
    about ``interp_share`` of a month's picks are interp-tagged when the pool allows (:func:`select_month`).
    ``round(controls_share * real)`` fake-paper controls are added, spread over the planned months. Every event
    passes :func:`~cutoffmix.bench.probegen.validate_event`; the ``meta`` rows (one per event, keyed by
    ``event_id``) carry the interp tag, upvotes, author count, author lists and distractors. Selection is
    deterministic; ``seed`` drives the MCQ order, the distractor draw and the control titles.
    """
    if not 0.0 <= controls_share < 1.0:
        raise ValueError(f"controls_share must be in 0..1, got {controls_share}")
    plans = plan_months(dense_from, dense_to, per_month, anchor_from=anchor_from, anchor_to=anchor_to,
                        anchor_every=anchor_every, anchor_per_month=anchor_per_month)
    by_month = gather_candidates(plans, candidates_per_month=candidates_per_month, client=client, cache_dir=cache_dir,
                                 today=today, retries=retries, errors=errors)
    events: list[Event] = []
    meta: list[dict[str, Any]] = []
    for plan in plans:
        chosen = select_month(by_month.get(plan.month, []), plan.count, interp_share)
        if len(chosen) < plan.count:
            log.warning("%s: only %d of %d papers available", plan.month, len(chosen), plan.count)
        for c in chosen:
            rng = random.Random(stable_hash(f"{seed}|paper:{c.ax.id}|distractors"))
            distractors = _distractors(rng, c, by_month)
            if not distractors:
                log.warning("%s: fewer than 3 distractor first authors; event without MCQ", c.ax.id)
            event = paper_event(c, distractors, seed)
            problems = validate_event(event)
            if problems:
                log.info("dropping %s: %s", event.id, "; ".join(problems))
                continue
            events.append(event)
            meta.append(_meta(event, c, distractors=distractors))

    n_controls = round(len(events) * controls_share)
    if n_controls:
        rng = random.Random(stable_hash(f"{seed}|controls"))
        real_titles = [t for cs in by_month.values() for c in cs for t in (c.ax.title, c.latest.title if c.latest
                                                                             else c.ax.title)]
        probed = {e.id.removeprefix("paper:") for e in events}
        months = sorted(m for m in by_month if by_month[m])
        made: set[str] = set()
        n = 0
        for i in range(n_controls):
            month = months[i % len(months)] if months else None
            fake = fake_title(rng, by_month, real_titles + sorted(made), month=month, exclude=probed)
            if fake is None:
                log.warning("control %d: no fake title found", i + 1)
                continue
            title, a, b = fake
            made.add(title)
            n += 1
            event = control_event(n, title, a)
            problems = validate_event(event)
            if problems:
                log.info("dropping %s: %s", event.id, "; ".join(problems))
                continue
            events.append(event)
            meta.append(_meta(event, None, donors=[a, b]))

    order = {e.id: i for i, e in enumerate(sorted(events, key=lambda e: (e.date, e.id)))}
    events.sort(key=lambda e: order[e.id])
    meta.sort(key=lambda m: order[m["event_id"]])
    return events, meta
