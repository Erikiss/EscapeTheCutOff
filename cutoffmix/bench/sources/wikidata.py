"""Template probe events from the Wikidata Query Service (WDQS); Wikidata content is CC0.

Four event families, each from a selective SPARQL query (:data:`QUERIES`):

* ``heads``      new heads of state / government (P39 + P580; offices from P1906/P1313) -> ``office_change``, government
* ``elections``  elections with one successful candidate (P991, P585)                    -> ``election``, elections
* ``awards``     recipients of a curated set of major awards (P166 + P585)              -> ``award``, per-award topic
* ``sports``     editions of recurring competitions with one winner (P1346, P582/P585)  -> ``sports_result``, sports

Probes are templates and need no LLM: ``question_current`` names no date (and is left empty when a later event
superseded the answer), ``question_recall`` names month and year, and the MCQ holds the correct answer, the stale
answer (predecessor / previous winner) and two plausible same-type candidates from the same query results, with
the correct letter chosen by a stable hash of the event id. ``aliases`` are the label and simple variants,
``reject_aliases`` the stale answers. Noise filters: bare-QID labels, deprecated statements, date precision coarser
than a month, collective offices, co-holders, multi-winner items, and minimum sitelinks for elections and sports.

WDQS etiquette: one query at a time, a descriptive User-Agent (:data:`USER_AGENT`), ``Retry-After`` honoured.
Date-range queries run in yearly windows; a window that times out is split in half and retried.
"""

from __future__ import annotations

import datetime as _dt
import difflib
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import httpx

from cutoffmix.bench.probegen import (
    ascii_fold,
    assemble_mcq,
    long_date,
    month_year,
    name_variants,
    stable_hash,
    validate_event,
)
from cutoffmix.bench.sources import SourceError, polite_request
from cutoffmix.types import Event

log = logging.getLogger(__name__)

USER_AGENT = "cutoffmix/0.1 (https://github.com/Erikiss/EscapeTheCutOff)"
WDQS_URL = "https://query.wikidata.org/sparql"
ENTITY_PREFIX = "http://www.wikidata.org/entity/"
XSD = "http://www.w3.org/2001/XMLSchema#"

PRECISION_MONTH = 10
PRECISION_DAY = 11
CONTEXT_CHUNK = 200  # entities per VALUES block in follow-up queries
US_QID = "Q30"

_QID = re.compile(r"^Q\d+$")
_YEARS = re.compile(r"\b(?:1[89]|20)\d{2}(?:\s*[–-]\s*\d{2,4})?\b")


def make_client(timeout: float = 90.0) -> httpx.Client:
    """An ``httpx.Client`` with the project User-Agent (no personal data) for Wikimedia and dataset downloads."""
    return httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=timeout, follow_redirects=True)


# --------------------------------------------------------------------------------------
# Awards: QID -> topic, family (for distractors) and the usual announcement month for year-precision dates
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AwardInfo:
    topic: str
    family: str
    month: int | None = None  # month a year-precision P585 maps to (the award's usual announcement); None -> drop


AWARDS: dict[str, AwardInfo] = {
    "Q38104": AwardInfo("science_research", "Nobel", 10),  # Nobel Prize in Physics
    "Q44585": AwardInfo("science_research", "Nobel", 10),  # Nobel Prize in Chemistry
    "Q80061": AwardInfo("science_research", "Nobel", 10),  # Nobel Prize in Physiology or Medicine
    "Q37922": AwardInfo("arts_entertainment", "Nobel", 10),  # Nobel Prize in Literature
    "Q35637": AwardInfo("international_relations", "Nobel", 10),  # Nobel Peace Prize
    "Q47170": AwardInfo("business_economy", "Nobel", 10),  # Nobel Memorial Prize in Economic Sciences
    "Q188184": AwardInfo("science_research", "science prize", 3),  # Abel Prize
    "Q845333": AwardInfo("science_research", "science prize", 2),  # Wolf Prize in Physics
    "Q540561": AwardInfo("science_research", "science prize", 2),  # Wolf Prize in Medicine
    "Q1314470": AwardInfo("science_research", "science prize", 4),  # Breakthrough Prize in Fundamental Physics
    "Q5019489": AwardInfo("science_research", "science prize", 4),  # Breakthrough Prize in Life Sciences
    "Q584250": AwardInfo("science_research", "science prize", 5),  # Shaw Prize
    "Q658444": AwardInfo("science_research", "science prize", 6),  # Kyoto Prize
    "Q921415": AwardInfo("science_research", "science prize", 9),  # Lasker Award
    "Q908745": AwardInfo("science_research", "science prize", 1),  # Japan Prize
    "Q185667": AwardInfo("computing_software", "science prize"),  # Turing Award (announced the year after)
    "Q160082": AwardInfo("arts_entertainment", "book prize", 11),  # Booker Prize
    "Q2052291": AwardInfo("arts_entertainment", "book prize", 5),  # International Booker Prize
    "Q187300": AwardInfo("arts_entertainment", "book prize", 11),  # Prix Goncourt
    "Q833633": AwardInfo("arts_entertainment", "book prize", 5),  # Pulitzer Prize for Fiction
    "Q18884": AwardInfo("arts_entertainment", "book prize", 6),  # Women's Prize for Fiction
    "Q255032": AwardInfo("arts_entertainment", "book prize", 8),  # Hugo Award for Best Novel
    "Q133160": AwardInfo("arts_entertainment", "architecture prize", 3),  # Pritzker Architecture Prize
    "Q102427": AwardInfo("arts_entertainment", "Academy Award"),  # Best Picture
    "Q103360": AwardInfo("arts_entertainment", "Academy Award"),  # Best Director
    "Q103916": AwardInfo("arts_entertainment", "Academy Award"),  # Best Actor
    "Q103618": AwardInfo("arts_entertainment", "Academy Award"),  # Best Actress
    "Q106291": AwardInfo("arts_entertainment", "Academy Award"),  # Best Supporting Actor
    "Q106301": AwardInfo("arts_entertainment", "Academy Award"),  # Best Supporting Actress
    "Q105304": AwardInfo("arts_entertainment", "Academy Award"),  # Best International Feature Film
    "Q106800": AwardInfo("arts_entertainment", "Academy Award"),  # Best Animated Feature
    "Q131520": AwardInfo("arts_entertainment", "Academy Award"),  # Best Cinematography
    "Q179808": AwardInfo("arts_entertainment", "film festival prize", 5),  # Palme d'Or
    "Q209459": AwardInfo("arts_entertainment", "film festival prize", 9),  # Golden Lion
    "Q154590": AwardInfo("arts_entertainment", "film festival prize", 2),  # Golden Bear
    "Q1011509": AwardInfo("arts_entertainment", "screen award"),  # Golden Globe Best Motion Picture - Drama
    "Q139184": AwardInfo("arts_entertainment", "screen award"),  # BAFTA Award for Best Film
    "Q989438": AwardInfo("arts_entertainment", "screen award"),  # Emmy for Outstanding Drama Series
    "Q904528": AwardInfo("arts_entertainment", "Grammy"),  # Album of the Year
    "Q843219": AwardInfo("arts_entertainment", "Grammy"),  # Record of the Year
    "Q1027904": AwardInfo("arts_entertainment", "Grammy"),  # Song of the Year
    "Q1453643": AwardInfo("arts_entertainment", "Grammy"),  # Best New Artist
    "Q166177": AwardInfo("sports", "football award"),  # Ballon d'Or
    "Q59411480": AwardInfo("sports", "football award"),  # Ballon d'Or Feminin
    "Q28156245": AwardInfo("sports", "football award"),  # The Best FIFA Men's Player
    "Q222047": AwardInfo("sports", "season award"),  # NBA Most Valuable Player
    "Q678383": AwardInfo("sports", "season award"),  # Hart Memorial Trophy
    "Q634857": AwardInfo("sports", "season award", 11),  # Cy Young Award
    "Q1035067": AwardInfo("sports", "season award", 12),  # Heisman Trophy
}

# --------------------------------------------------------------------------------------
# Queries. Placeholders: __START__/__END__ (xsd:dateTime, end exclusive) and __VALUES__ (space-separated wd:Q..).
# --------------------------------------------------------------------------------------

_LABELS = 'SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }'

QUERIES: dict[str, str] = {
    "heads": f"""
SELECT DISTINCT ?person ?personLabel ?office ?officeLabel ?country ?countryLabel ?start ?prec ?end
       ?replaces ?replacesLabel ?nature ?sitelinks WHERE {{
  ?country wdt:P31 wd:Q6256 .
  {{ ?country wdt:P1906 ?office }} UNION {{ ?country wdt:P1313 ?office }}
  ?person p:P39 ?st . ?st ps:P39 ?office ; pqv:P580 ?sv .
  ?sv wikibase:timeValue ?start ; wikibase:timePrecision ?prec .
  FILTER(?start >= "__START__"^^xsd:dateTime && ?start < "__END__"^^xsd:dateTime)
  ?st wikibase:rank ?rank . FILTER(?rank != wikibase:DeprecatedRank)
  OPTIONAL {{ ?st pq:P582 ?end }}
  OPTIONAL {{ ?st pq:P1365 ?replaces }}
  OPTIONAL {{ ?st pq:P5102 ?nature }}
  OPTIONAL {{ ?person wikibase:sitelinks ?sitelinks }}
  {_LABELS}
}}""",
    "elections": f"""
SELECT DISTINCT ?e ?eLabel ?d ?winner ?winnerLabel ?country ?countryLabel ?sitelinks WHERE {{
  ?e wdt:P991 ?winner ; wdt:P585 ?d .
  FILTER(?d >= "__START__"^^xsd:dateTime && ?d < "__END__"^^xsd:dateTime)
  ?e wikibase:sitelinks ?sitelinks . FILTER(?sitelinks >= 3)
  OPTIONAL {{ ?e wdt:P17 ?country }}
  {_LABELS}
}}""",
    "election_context": f"""
SELECT ?e ?d ?prec ?prevWinner ?prevWinnerLabel ?cand ?candLabel WHERE {{
  VALUES ?e {{ __VALUES__ }}
  {{ ?e p:P585/psv:P585 ?dn . ?dn wikibase:timeValue ?d ; wikibase:timePrecision ?prec . }}
  UNION {{ ?e wdt:P155 ?prev . ?prev wdt:P991 ?prevWinner . }}
  UNION {{ ?e wdt:P726 ?cand . }}
  {_LABELS}
}}""",
    "awards": f"""
SELECT DISTINCT ?award ?awardLabel ?who ?whoLabel ?d ?prec ?human WHERE {{
  VALUES ?award {{ {" ".join("wd:" + q for q in AWARDS)} }}
  ?who p:P166 ?st . ?st ps:P166 ?award ; pqv:P585 ?dv .
  ?dv wikibase:timeValue ?d ; wikibase:timePrecision ?prec .
  FILTER(?d >= "__START__"^^xsd:dateTime && ?d < "__END__"^^xsd:dateTime)
  ?st wikibase:rank ?rank . FILTER(?rank != wikibase:DeprecatedRank)
  BIND(EXISTS {{ ?who wdt:P31 wd:Q5 }} AS ?human)
  {_LABELS}
}}""",
    "sports": f"""
SELECT DISTINCT ?ed ?edLabel ?comp ?compLabel ?winner ?winnerLabel ?date ?sport ?sportLabel ?edLinks WHERE {{
  ?ed wdt:P3450 ?comp ; wdt:P1346 ?winner ; wikibase:sitelinks ?edLinks .
  FILTER(?edLinks >= 3)
  {{ ?ed wdt:P582 ?date }} UNION {{ ?ed wdt:P585 ?date }}
  FILTER(?date >= "__START__"^^xsd:dateTime && ?date < "__END__"^^xsd:dateTime)
  OPTIONAL {{ ?comp wdt:P641 ?sport }}
  {_LABELS}
}}""",
    "sports_series": f"""
SELECT DISTINCT ?ed ?edLabel ?comp ?compLabel ?winner ?winnerLabel ?date ?sport ?sportLabel ?edLinks WHERE {{
  ?ed wdt:P179 ?comp . ?comp wdt:P31/wdt:P279* wd:Q15275719 .
  ?ed wdt:P1346 ?winner .
  {{ ?ed wdt:P582 ?date }} UNION {{ ?ed wdt:P585 ?date }}
  FILTER(?date >= "__START__"^^xsd:dateTime && ?date < "__END__"^^xsd:dateTime)
  ?ed wikibase:sitelinks ?edLinks . FILTER(?edLinks >= 3)
  OPTIONAL {{ ?comp wdt:P641 ?sport }}
  {_LABELS}
}}""",
    "sports_context": f"""
SELECT ?ed ?d ?prec ?prevWinner ?prevWinnerLabel ?prevHuman ?humanWinner WHERE {{
  VALUES ?ed {{ __VALUES__ }}
  {{ ?ed p:P582/psv:P582 ?dn . ?dn wikibase:timeValue ?d ; wikibase:timePrecision ?prec . }}
  UNION {{ ?ed p:P585/psv:P585 ?dn . ?dn wikibase:timeValue ?d ; wikibase:timePrecision ?prec . }}
  UNION {{ ?ed wdt:P155 ?prev . ?prev wdt:P1346 ?prevWinner .
           BIND(EXISTS {{ ?prevWinner wdt:P31 wd:Q5 }} AS ?prevHuman) }}
  UNION {{ ?ed wdt:P1346 ?humanWinner . ?humanWinner wdt:P31 wd:Q5 . }}
  {_LABELS}
}}""",
}


def render_query(name: str, *, start: _dt.date | str | None = None, end: _dt.date | str | None = None,
                 values: Iterable[str] = ()) -> str:
    """Fill a :data:`QUERIES` template: ``start``/``end`` dates (end exclusive), ``values`` QIDs for VALUES."""
    try:
        query = QUERIES[name]
    except KeyError as e:
        raise KeyError(f"unknown query {name!r}; known: {', '.join(QUERIES)}") from e
    if "__START__" in query:
        if start is None or end is None:
            raise ValueError(f"query {name!r} needs start and end dates")
        query = query.replace("__START__", f"{start}T00:00:00Z").replace("__END__", f"{end}T00:00:00Z")
    if "__VALUES__" in query:
        qids = [q for q in values if _QID.match(q)]
        if not qids:
            raise ValueError(f"query {name!r} needs at least one QID in values")
        query = query.replace("__VALUES__", " ".join("wd:" + q for q in qids))
    return query


# --------------------------------------------------------------------------------------
# SPARQL transport
# --------------------------------------------------------------------------------------


def _flatten_value(cell: Mapping[str, str]) -> Any:
    value = cell.get("value", "")
    if cell.get("type") == "uri" and value.startswith(ENTITY_PREFIX):
        return value[len(ENTITY_PREFIX):]
    datatype = cell.get("datatype", "")
    if datatype == XSD + "integer":
        return int(value)
    if datatype == XSD + "boolean":
        return value == "true"
    return value


def flatten_bindings(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    """SPARQL JSON results -> one dict per row. Entity URIs become bare QIDs, xsd:integer -> int,
    xsd:boolean -> bool, everything else (dates included) stays a string; unbound variables are absent."""
    return [{k: _flatten_value(v) for k, v in row.items()} for row in data["results"]["bindings"]]


def run_sparql(query: str, *, client: httpx.Client | None = None, retries: int = 4) -> list[dict[str, Any]]:
    """POST ``query`` to WDQS and return the flattened rows (:func:`flatten_bindings`).

    Retries 429/5xx/timeouts up to ``retries`` times, honouring ``Retry-After``. Raises
    :class:`~cutoffmix.bench.sources.SourceError`; ``err.timeout`` is set for query timeouts, including WDQS's
    habit of ending a timed-out query inside a 200 response (cut-off JSON).
    """
    own = client is None
    http = client or make_client()
    try:
        resp = polite_request(http, "POST", WDQS_URL, retries=retries, data={"query": query},
                              headers={"Accept": "application/sparql-results+json", "User-Agent": USER_AGENT})
        try:
            data = json.loads(resp.content)
        except json.JSONDecodeError as e:
            raise SourceError(f"WDQS returned cut-off JSON ({len(resp.content)} bytes): {e}", status=resp.status_code,
                              timeout=True) from e
        if "results" not in data:
            raise SourceError("WDQS response has no 'results'", status=resp.status_code)
        return flatten_bindings(data)
    finally:
        if own:
            http.close()


def year_windows(start: _dt.date, end: _dt.date, months: int = 12) -> list[tuple[_dt.date, _dt.date]]:
    """Split ``[start, end)`` into consecutive windows of ``months`` calendar months (the last may be shorter)."""
    out = []
    cur = start
    while cur < end:
        m = cur.month - 1 + months
        nxt = min(end, _dt.date(cur.year + m // 12, m % 12 + 1, min(cur.day, 28)))
        out.append((cur, nxt))
        cur = nxt
    return out


def run_windowed(name: str, start: _dt.date, end: _dt.date, *, client: httpx.Client, retries: int = 2,
                 months: int = 12, min_days: int = 7,
                 runner: Callable[..., list[dict[str, Any]]] | None = None) -> list[dict[str, Any]]:
    """Run date-range query ``name`` over ``[start, end)`` in windows of ``months``; a window that times out is
    split in half (down to ``min_days``) instead of being repeated. ``runner`` defaults to :func:`run_sparql`."""
    run = runner or run_sparql

    def window(ws: _dt.date, we: _dt.date) -> list[dict[str, Any]]:
        try:
            rows = run(render_query(name, start=ws, end=we), client=client, retries=retries)
        except SourceError as e:
            if not e.timeout or (we - ws).days <= min_days:
                raise
            mid = ws + (we - ws) / 2
            log.warning("WDQS %s %s..%s timed out; splitting at %s", name, ws, we, mid)
            return window(ws, mid) + window(mid, we)
        log.info("WDQS %s %s..%s: %d rows", name, ws, we, len(rows))
        return rows

    rows: list[dict[str, Any]] = []
    for ws, we in year_windows(start, end, months):
        rows += window(ws, we)
    return rows


def run_context(name: str, qids: Iterable[str], *, client: httpx.Client, retries: int = 4,
                runner: Callable[..., list[dict[str, Any]]] | None = None) -> list[dict[str, Any]]:
    """Run VALUES query ``name`` for ``qids`` in chunks of :data:`CONTEXT_CHUNK`."""
    run = runner or run_sparql
    ids = sorted(set(qids))
    rows: list[dict[str, Any]] = []
    for i in range(0, len(ids), CONTEXT_CHUNK):
        rows += run(render_query(name, values=ids[i:i + CONTEXT_CHUNK]), client=client, retries=retries)
    log.info("WDQS %s: %d ids, %d rows", name, len(ids), len(rows))
    return rows


# --------------------------------------------------------------------------------------
# Shared helpers for the template builders
# --------------------------------------------------------------------------------------


def is_qid_label(label: Any) -> bool:
    """True for missing labels and for labels that are bare QIDs (no English label on Wikidata)."""
    return not isinstance(label, str) or not label.strip() or bool(_QID.match(label.strip()))


def _key(name: str) -> str:
    return ascii_fold(name).casefold().strip()


_WOMEN = re.compile(r"\bwomen|\bwomen's|\bfemenino|\bféminin|\bladies\b", re.I)
_YOUTH = re.compile(r"\bunder-?\d+|\bu-?\d{2}\b|\bjunior|\byouth\b", re.I)


def _team_kind(label: str) -> tuple[bool, bool, bool]:
    """(women's, national team, youth) markers of a winner label, so distractors match the answer's kind."""
    return bool(_WOMEN.search(label)), "national" in label.lower(), bool(_YOUTH.search(label))


def same_name(a: str, b: str) -> bool:
    """True if two labels name the same entity: equal ignoring case and diacritics, or the same number of words
    with every word pair equal or a spelling variant (same first three letters, similarity >= 0.75: "FC Bayern
    Munich" / "FC Bayern München"), or one label containing the other's words (years ignored, at least two
    words); always with the same team kind (a women's or youth team is never the senior team)."""
    ka, kb = _key(a), _key(b)
    if ka == kb:
        return True
    if _team_kind(a) != _team_kind(b):
        return False
    ta, tb = re.findall(r"[\w']+", _YEARS.sub(" ", ka)), re.findall(r"[\w']+", _YEARS.sub(" ", kb))
    short, long_ = sorted((ta, tb), key=len)
    if len(short) >= 2 and any(long_[i:i + len(short)] == short for i in range(len(long_) - len(short) + 1)):
        return True  # "2025 Indiana Hoosiers football team" / "Indiana Hoosiers football"
    return len(ta) == len(tb) and all(
        x == y or (x[:3] == y[:3] and difflib.SequenceMatcher(None, x, y).ratio() >= 0.75) for x, y in zip(ta, tb))


def _day(value: str) -> str:
    return value[:10]


def _as_of(as_of: str | None) -> str:
    return as_of or _dt.date.today().isoformat()


def _the(name: str) -> str:
    return name if name.lower().startswith("the ") else f"the {name}"


def pick_distractors(event_id: str, tiers: Sequence[Iterable[str]], exclude: Iterable[str], n: int) -> list[str]:
    """Pick ``n`` distinct candidate labels, tier by tier (most plausible tier first), skipping bare QIDs and
    anything that :func:`same_name` matches with ``exclude`` or an earlier pick; order within a tier is a stable
    hash of (event id, label), so different events get different distractors."""
    taken = [x for x in exclude if isinstance(x, str) and x.strip()]
    out: list[str] = []
    for tier in tiers:
        cands = sorted({c.strip() for c in tier if not is_qid_label(c)}, key=lambda c: stable_hash(f"{event_id}|{c}"))
        for c in cands:
            if len(out) == n:
                return out
            if not any(same_name(c, x) for x in taken):
                taken.append(c)
                out.append(c)
    return out


def _mcq(event_id: str, correct: str, stale: str | None, tiers: Sequence[Iterable[str]],
         exclude: Iterable[str]) -> tuple[list[str], str]:
    """MCQ with the correct answer, the stale answer (if any) and same-type distractors; ([], "") if too few."""
    n = 2 if stale else 3
    wrong = pick_distractors(event_id, tiers, [correct, *([stale] if stale else []), *exclude], n)
    if len(wrong) < n:
        return [], ""
    return assemble_mcq(event_id, correct, ([stale] if stale else []) + wrong)


def _variants(labels: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(v for label in labels for v in name_variants(label)))


def _event(*, eid: str, date: str, topic: str, category: str, subject: str, fact: str, region: str,
           q_current: str, a_current: str, q_recall: str, a_recall: str, correct: str, accepted: Iterable[str],
           stale: Sequence[str], mcq_stale: str | None, tiers: Sequence[Iterable[str]], exclude: Iterable[str],
           predictability: str, qid: str, mcq_question: str | None = None) -> Event:
    choices, letter = _mcq(eid, correct, mcq_stale, tiers, exclude)
    aliases = _variants(accepted)
    accepted_keys = {a.casefold() for a in aliases}
    reject = [r for r in _variants(stale) if r.casefold() not in accepted_keys]
    return Event(
        id=eid, date=date, month=date[:7], topic=topic, category=category, subject=subject, fact=fact,
        region=region, predictability=predictability,  # type: ignore[arg-type]
        question_current=q_current, expected_current=a_current if q_current else "",
        question_recall=q_recall, expected_recall=a_recall,
        mcq_question=(mcq_question or q_current or q_recall) if choices else "", mcq_choices=choices,
        mcq_answer=letter,
        aliases=aliases, reject_aliases=reject, source=f"https://www.wikidata.org/wiki/{qid}", origin="wikidata",
    )


def _region(country: str | None) -> str:
    return "US" if country == US_QID else "International"


# --------------------------------------------------------------------------------------
# Heads of state / government
# --------------------------------------------------------------------------------------

_COLLECTIVE_OFFICE = re.compile(r"^member of\b|\b(council|presidency|committee|board|junta|collegium)\b", re.I)


def events_from_heads(rows: Iterable[Mapping[str, Any]], *, as_of: str | None = None) -> list[Event]:
    """``office_change`` events (topic government) from :data:`QUERIES` ``["heads"]`` rows.

    Acting holders (P5102) and collective or co-held offices are not events, but acting holders still count as
    later holders and as stale answers. ``question_current`` ("Who is the <office>?") is set only when nobody took
    the office later and the statement has no end date on or before ``as_of`` (default today).
    """
    today = _as_of(as_of)
    stmts: dict[tuple[str, str, str], dict[str, Any]] = {}
    replaces: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for r in rows:
        if "person" not in r or "office" not in r or "start" not in r:
            continue
        k = (r["office"], r["person"], _day(r["start"]))
        s = stmts.setdefault(k, dict(r))
        if "end" in r:
            s["end"] = r["end"]
        if "nature" in r:
            s["nature"] = r["nature"]
        if r.get("replaces") and not is_qid_label(r.get("replacesLabel")):
            replaces[k].add(r["replacesLabel"])

    by_office: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in stmts.values():
        by_office[s["office"]].append(s)
    starts: dict[tuple[str, str], set[str]] = defaultdict(set)
    for s in stmts.values():
        starts[(s["office"], _day(s["start"]))].add(s["person"])
    collective = {office for (office, _), people in starts.items() if len(people) > 1}
    for lst in by_office.values():
        lst.sort(key=lambda s: _day(s["start"]))

    def label(s: Mapping[str, Any]) -> str:
        return str(s.get("personLabel", ""))

    pool_by_country: dict[str, set[str]] = defaultdict(set)
    pool_by_kind: dict[str, set[str]] = defaultdict(set)
    pool_all: set[str] = set()
    for s in stmts.values():
        kind = str(s.get("officeLabel", "")).lower().split(" of ")[0]
        for name in [label(s), *replaces[(s["office"], s["person"], _day(s["start"]))]]:
            pool_by_country[s.get("country", "")].add(name)
            pool_by_kind[kind].add(name)
            pool_all.add(name)

    events: list[Event] = []
    for (office, person, start), s in sorted(stmts.items(), key=lambda kv: (kv[0][2], kv[0][0], kv[0][1])):
        name, office_label = label(s), str(s.get("officeLabel", ""))
        if ("nature" in s or office in collective or _COLLECTIVE_OFFICE.search(office_label)
                or is_qid_label(name) or is_qid_label(office_label)
                or int(s.get("prec", PRECISION_DAY)) < PRECISION_MONTH):
            continue
        reps = sorted(replaces[(office, person, start)])
        if any(same_name(x, name) for x in reps):
            continue
        holders = by_office[office]
        earlier = [h for h in holders if _day(h["start"]) < start and h["person"] != person]
        later = [h for h in holders if _day(h["start"]) > start and h["person"] != person]
        superseded = bool(later) or ("end" in s and _day(s["end"]) <= today)
        # Stale answers, most direct first: the 'replaces' qualifier (and, when that predecessor only acted, the
        # holder they stood in for), the latest earlier holder and the latest earlier non-acting holder in the data.
        acting = {_key(label(h)): h for h in earlier if "nature" in h}
        chain = list(reps)
        for rep in reps:
            if _key(rep) in acting:
                h = acting[_key(rep)]
                chain += sorted(replaces[(office, h["person"], _day(h["start"]))])
        substantive = [h for h in earlier if "nature" not in h]
        chain += [label(h) for h in (earlier[-1:] + substantive[-1:])]
        stale = [x for x in dict.fromkeys(chain) if not is_qid_label(x) and not same_name(x, name)]
        # the MCQ's stale option is the first substantive (non-acting) predecessor: what most stale models answer
        mcq_stale = next((x for x in stale if _key(x) not in acting), stale[0] if stale else None)

        when = month_year(start)
        country = s.get("country", "")
        kind = office_label.lower().split(" of ")[0]
        eid = f"wd-office-{office}-{start}-{person}"
        pred = f", succeeding {mcq_stale}" if mcq_stale else ""
        events.append(_event(
            eid=eid, date=start, topic="government", category="office_change", subject=name,
            fact=f"{name} became {office_label} on {long_date(start)}{pred}.", region=_region(country),
            q_current="" if superseded else f"Who is the {office_label}?",
            a_current=f"{name} (in office since {long_date(start)}{pred})",
            q_recall=f"Who became {office_label} in {when}?",
            a_recall=f"{name} became {office_label} on {long_date(start)}.",
            correct=name, accepted=[name], stale=stale, mcq_stale=mcq_stale,
            tiers=[pool_by_country.get(country, set()), pool_by_kind.get(kind, set()), pool_all],
            exclude=stale, predictability="medium", qid=person,
        ))
    return events


# --------------------------------------------------------------------------------------
# Elections
# --------------------------------------------------------------------------------------

_MONTH_WORDS = re.compile(r"\b(?:January|February|March|April|May|June|July|August|September|October|November"
                          r"|December)\b")
_ORDINAL = re.compile(r"^\d+(?:st|nd|rd|th)\s+", re.I)


def series_name(label: str) -> tuple[str, bool]:
    """Strip year and month tokens (and a leading ordinal) from an election label:
    ``'2026 Brazilian presidential election'`` -> ``('Brazilian presidential election', False)``; the flag says
    whether an ordinal was removed."""
    text = re.sub(r"\s+", " ", _MONTH_WORDS.sub(" ", _YEARS.sub(" ", label))).strip(" ,–-")
    stripped = _ORDINAL.sub("", text)
    return stripped.strip(), stripped != text


def _election_type(label: str) -> str:
    """Coarse election kind for distractor tiers: the word before "election" ("presidential", "mayoral", ...)."""
    m = re.search(r"(\w+)\s+(?:special\s+)?(?:by-)?elections?\b", label.lower())
    return m.group(1) if m else ""


def _context_by(rows: Iterable[Mapping[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    ctx: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"dates": [], "prev": set(), "prev_human": set(), "cands": set(), "human": set()})
    for r in rows:
        c = ctx[r[key]]
        if "d" in r and "prec" in r:
            c["dates"].append((_day(r["d"]), int(r["prec"])))
        if not is_qid_label(r.get("prevWinnerLabel")):
            c["prev"].add(r["prevWinnerLabel"])
            if r.get("prevHuman") is True:
                c["prev_human"].add(r["prevWinnerLabel"])
        if not is_qid_label(r.get("candLabel")):
            c["cands"].add(r["candLabel"])
        if r.get("humanWinner"):
            c["human"].add(r["humanWinner"])
    return dict(ctx)


def _best_date(dates: Sequence[tuple[str, int]], fallback: str | None) -> str | None:
    """Latest date among the most precise values (at least month precision); a month-precision date gets day 01."""
    good = [(p >= PRECISION_DAY, d if p >= PRECISION_DAY else d[:8] + "01") for d, p in dates if p >= PRECISION_MONTH]
    if dates:
        return max(good)[1] if good else None
    return fallback


def election_context(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Group :data:`QUERIES` ``["election_context"]`` rows by election: dates with precision, previous winners,
    candidates."""
    return _context_by(rows, "e")


def events_from_elections(rows: Iterable[Mapping[str, Any]],
                          context: Mapping[str, Mapping[str, Any]] | None = None) -> list[Event]:
    """``election`` events (topic elections) from :data:`QUERIES` ``["elections"]`` rows.

    Only elections with exactly one successful candidate. ``context`` (:func:`election_context`) adds date
    precision (coarser than a month -> dropped), the previous election's winner (stale answer; a re-elected winner
    makes the event high-predictability without a stale option) and the candidates (preferred distractors).
    ``question_current`` ("Who won the most recent <series>?") is empty when a later election of the same series
    is in the data.
    """
    ctx = context or {}
    groups: dict[str, dict[str, Any]] = {}
    for r in rows:
        g = groups.setdefault(r["e"], {"label": r.get("eLabel"), "winners": {}, "country": None, "cl": None,
                                       "date": _day(r["d"])})
        g["winners"][r["winner"]] = r.get("winnerLabel")
        if r.get("country") and not g["country"]:
            g["country"], g["cl"] = r["country"], r.get("countryLabel")

    elections = []
    for e, g in groups.items():
        if is_qid_label(g["label"]) or len(g["winners"]) != 1:
            continue
        (winner_label,) = g["winners"].values()
        date = _best_date(ctx.get(e, {}).get("dates", []), g["date"])
        if is_qid_label(winner_label) or date is None:
            continue
        series, had_ordinal = series_name(g["label"])
        if had_ordinal and g["cl"] and not is_qid_label(g["cl"]):
            series = f"{series} ({g['cl']})"
        elif had_ordinal:
            series = ""
        elections.append({**g, "e": e, "winner": winner_label, "date": date, "series": series})

    by_series: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for el in elections:
        by_series[el["series"].casefold() or el["e"]].append(el)
    for lst in by_series.values():
        lst.sort(key=lambda x: x["date"])
    winners_by_country: dict[str, set[str]] = defaultdict(set)
    winners_by_type: dict[str, set[str]] = defaultdict(set)
    for el in elections:
        winners_by_country[el["country"] or ""].add(el["winner"])
        winners_by_type[_election_type(el["label"])].add(el["winner"])
    all_winners = {el["winner"] for el in elections}

    events = []
    for el in sorted(elections, key=lambda x: (x["date"], x["e"])):
        siblings = by_series[el["series"].casefold() or el["e"]]
        idx = siblings.index(el)
        superseded = idx < len(siblings) - 1 or not el["series"]
        c = ctx.get(el["e"], {})
        prev = sorted(c.get("prev", set())) or ([siblings[idx - 1]["winner"]] if idx > 0 else [])
        repeat = any(same_name(p, el["winner"]) for p in prev)
        stale = [] if repeat else prev
        label, winner, date = el["label"], el["winner"], el["date"]
        held = f"held on {long_date(date)}"
        events.append(_event(
            eid=f"wd-election-{el['e']}", date=date, topic="elections", category="election", subject=label,
            fact=f"{winner} won the {label}, {held}.", region=_region(el["country"]),
            q_current="" if superseded else f"Who won the most recent {el['series']}?",
            a_current=f"{winner} (winner of the {label}, {held})",
            q_recall=f"Who won the {label}, held in {month_year(date)}?", a_recall=f"{winner} won the {label}, {held}.",
            correct=winner, accepted=[winner], stale=stale, mcq_stale=stale[0] if stale else None,
            tiers=[c.get("cands", set()), winners_by_country.get(el["country"] or "", set()),
                   winners_by_type.get(_election_type(label), set()), all_winners],
            exclude=prev, predictability="high" if repeat else "medium", qid=el["e"],
        ))
    return events


# --------------------------------------------------------------------------------------
# Awards
# --------------------------------------------------------------------------------------


def events_from_awards(rows: Iterable[Mapping[str, Any]], *, since: str | None = None) -> list[Event]:
    """``award`` events from :data:`QUERIES` ``["awards"]`` rows, one per award and year.

    Year-precision dates map to the award's usual announcement month (:data:`AWARDS`) or are dropped; recipients
    are restricted to humans when any recipient is human (drops films/works attached to person awards). Awards
    dated before ``since`` only serve as previous recipients (stale answers) and distractors; query a year more
    than needed so the first events in range have a predecessor.
    """
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        info = AWARDS.get(r.get("award", ""))
        if info is None or is_qid_label(r.get("awardLabel")) or is_qid_label(r.get("whoLabel")):
            continue
        prec, day = int(r.get("prec", 0)), _day(r["d"])
        if prec >= PRECISION_MONTH:
            date, exact = (day if prec >= PRECISION_DAY else day[:8] + "01"), prec
        elif info.month is not None:
            date, exact = f"{day[:4]}-{info.month:02d}-01", PRECISION_MONTH
        else:
            continue
        g = groups.setdefault((r["award"], day[:4]), {"award": r["award"], "label": r["awardLabel"], "info": info,
                                                       "who": {}, "human": set(), "dates": []})
        g["who"][r["who"]] = r["whoLabel"]
        if r.get("human") is True:
            g["human"].add(r["who"])
        g["dates"].append((exact, date))
    for g in groups.values():
        if g["human"]:
            g["who"] = {q: lab for q, lab in g["who"].items() if q in g["human"]}
        g["date"] = max(g["dates"])[1]
        g["names"] = sorted(set(g["who"].values()))

    by_award: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_family: dict[str, set[str]] = defaultdict(set)
    all_names = {n for g in groups.values() for n in g["names"]}
    humans: set[str] = set()
    for g in groups.values():
        by_award[g["award"]].append(g)
        by_family[g["info"].family].update(g["names"])
        if g["human"]:
            humans.update(g["names"])
    for lst in by_award.values():
        lst.sort(key=lambda x: x["date"])

    events = []
    for g in sorted(groups.values(), key=lambda x: (x["date"], x["award"])):
        if since and g["date"] < since:
            continue
        lst = by_award[g["award"]]
        idx = lst.index(g)
        superseded = idx < len(lst) - 1
        prev = lst[idx - 1]["names"] if idx > 0 else []
        repeat = any(same_name(p, n) for p in prev for n in g["names"])
        stale = [p for p in prev if not any(same_name(p, n) for n in g["names"])]
        label, names, date, info = g["label"], g["names"], g["date"], g["info"]
        who = " and ".join(names) if len(names) <= 2 else ", ".join(names[:-1]) + " and " + names[-1]
        correct = names[stable_hash(f"wd-award-{g['award']}-{date[:4]}") % len(names)]
        if len(names) > 1:
            mcq_q = (f"Which of the following won the {label} in {month_year(date)}?" if superseded
                     else f"Which of the following won the most recent {label}?")
        else:
            mcq_q = None
        events.append(_event(
            eid=f"wd-award-{g['award']}-{date[:4]}", date=date, topic=info.topic, category="award", subject=label,
            fact=f"{who} received the {label} ({month_year(date)}).", region="International",
            q_current="" if superseded else f"Who won the most recent {label}?", a_current=who,
            q_recall=f"Who won the {label} in {month_year(date)}?", a_recall=who,
            correct=correct, accepted=names, stale=stale, mcq_stale=stale[0] if stale else None,
            tiers=[by_family[info.family], humans if g["human"] else set(), all_names],
            exclude=[*names, *prev], predictability="high" if repeat else "medium", qid=g["award"],
            mcq_question=mcq_q,
        ))
    return events


# --------------------------------------------------------------------------------------
# Sports
# --------------------------------------------------------------------------------------


def sports_context(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Group :data:`QUERIES` ``["sports_context"]`` rows by edition: dates with precision, previous edition's
    winners (and which of them are people), human winners."""
    return _context_by(rows, "ed")


def events_from_sports(rows: Iterable[Mapping[str, Any]],
                       context: Mapping[str, Mapping[str, Any]] | None = None) -> list[Event]:
    """``sports_result`` events (topic sports) from :data:`QUERIES` ``["sports"]``/``["sports_series"]`` rows.

    One event per edition with exactly one winner (or exactly one human winner, e.g. a driver next to the
    constructor) and a known sport (P641 of the competition; drops pageants and quizzes filed as seasons).
    ``context`` (:func:`sports_context`) adds date precision (an edition without a date of at least
    month precision is dropped), the previous edition's winner (stale answer; a repeat champion makes the event
    high-predictability) and which winners are people. ``question_current`` ("Who won the most recent edition of
    the <competition>?") is empty when a later edition of the competition (by label) is in the data, or another
    winner under the same competition label within 30 days makes the question ambiguous. Items describing the
    same result (same competition label, date and winner) are merged.
    """
    ctx = context or {}
    groups: dict[str, dict[str, Any]] = {}
    for r in rows:
        g = groups.setdefault(r["ed"], {"label": r.get("edLabel"), "comps": {}, "winners": {}, "sports": set(),
                                        "dates": set(), "links": int(r.get("edLinks", 0))})
        g["comps"][r["comp"]] = r.get("compLabel")
        g["winners"][r["winner"]] = r.get("winnerLabel")
        g["dates"].add(_day(r["date"]))
        if not is_qid_label(r.get("sportLabel")):
            g["sports"].add(r["sportLabel"])

    editions = []
    for ed, g in groups.items():
        c = ctx.get(ed, {})
        winners = g["winners"]
        if len(winners) > 1 and len(c.get("human", set()) & set(winners)) == 1:
            winners = {q: lab for q, lab in winners.items() if q in c["human"]}
        comps = sorted((q, lab) for q, lab in g["comps"].items() if not is_qid_label(lab))
        if len(winners) != 1 or not comps or not g["sports"] or is_qid_label(g["label"]):
            continue
        ((winner_q, winner),) = winners.items()
        date = _best_date(c.get("dates", []), max(g["dates"]))
        if is_qid_label(winner) or date is None:
            continue
        comp = comps[0][1]
        human = winner_q in c.get("human", set())
        # previous winners of the same kind (person vs team) first: the stale option must match the answer type
        prev = sorted(c.get("prev", set()), key=lambda p: ((p in c.get("prev_human", set())) != human, p))
        series = re.sub(r"\s+", " ", _YEARS.sub(" ", comp)).strip()
        editions.append({"ed": ed, "label": g["label"], "series": series, "winner": winner, "human": human,
                         "sports": g["sports"], "date": date, "prev": prev, "links": g["links"]})

    # Several items can describe one result (an edition and its final, or two competition items with one label):
    # keep the best-linked one per (competition label, date, winner).
    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for x in sorted(editions, key=lambda x: (-x["links"], x["ed"])):
        unique.setdefault((x["series"].casefold(), x["date"], _key(x["winner"])), x)
    editions = list(unique.values())

    by_comp: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_team_kind: dict[tuple[str, bool, tuple[bool, ...]], set[str]] = defaultdict(set)
    by_kind_any_sport: dict[tuple[bool, tuple[bool, ...]], set[str]] = defaultdict(set)
    by_sport: dict[tuple[str, bool], set[str]] = defaultdict(set)
    by_kind: dict[bool, set[str]] = defaultdict(set)
    for x in editions:
        by_comp[x["series"].casefold()].append(x)
        by_kind[x["human"]].add(x["winner"])
        by_kind_any_sport[(x["human"], _team_kind(x["winner"]))].add(x["winner"])
        for sp in x["sports"]:
            by_sport[(sp, x["human"])].add(x["winner"])
            by_team_kind[(sp, x["human"], _team_kind(x["winner"]))].add(x["winner"])
    for lst in by_comp.values():
        lst.sort(key=lambda x: (x["date"], x["ed"]))

    events = []
    for x in sorted(editions, key=lambda x: (x["date"], x["ed"])):
        lst = by_comp[x["series"].casefold()]
        idx = lst.index(x)
        # a later edition answers "most recent" instead; so does a different winner within 30 days under the same
        # competition label (e.g. men's and women's leagues sharing a name), which makes the question ambiguous
        day = _dt.date.fromisoformat(x["date"])
        superseded = idx < len(lst) - 1 or any(
            y is not x and abs((_dt.date.fromisoformat(y["date"]) - day).days) <= 30
            and not same_name(y["winner"], x["winner"]) for y in lst)
        prev = x["prev"] or ([lst[idx - 1]["winner"]] if idx > 0 else [])
        repeat = any(same_name(p, x["winner"]) for p in prev)
        stale = [] if repeat else prev
        label, winner, date = x["label"], x["winner"], x["date"]
        same_sport = set().union(*(by_sport[(sp, x["human"])] for sp in x["sports"]))
        same_team_kind = set().union(*(by_team_kind[(sp, x["human"], _team_kind(winner))] for sp in x["sports"]))
        events.append(_event(
            eid=f"wd-sports-{x['ed']}", date=date, topic="sports", category="sports_result", subject=label,
            fact=f"{winner} won the {label} ({long_date(date)}).", region="International",
            q_current="" if superseded else f"Who won the most recent edition of {_the(x['series'])}?",
            a_current=f"{winner} (winner of the {label}, {long_date(date)})",
            q_recall=f"Who won the {label}, which concluded in {month_year(date)}?",
            a_recall=f"{winner} won the {label} ({long_date(date)}).",
            correct=winner, accepted=[winner], stale=stale, mcq_stale=stale[0] if stale else None,
            tiers=[{y["winner"] for y in lst}, same_team_kind, same_sport,
                   by_kind_any_sport[(x["human"], _team_kind(winner))], by_kind[x["human"]]],
            exclude=prev, predictability="high" if repeat else "medium", qid=x["ed"],
        ))
    return events


# --------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------

BUILDERS: tuple[str, ...] = ("heads", "elections", "awards", "sports")


def build_wikidata_events(start: str = "2024-01-01", end: str | None = None, *, client: httpx.Client | None = None,
                          kinds: Iterable[str] = BUILDERS, retries: int = 2,
                          runner: Callable[..., list[dict[str, Any]]] | None = None) -> list[Event]:
    """Query WDQS and build validated template events dated within ``[start, end]`` (``end`` default today, also
    the "as of" date for which ``question_current`` must still be true).

    Queries run sequentially; date-range queries use yearly windows that are split on timeout (see
    :func:`run_windowed`). Awards are queried from one year before ``start`` so the first awards in range have a
    previous recipient. Events failing :func:`~cutoffmix.bench.probegen.validate_event` are dropped (logged).
    Returns events sorted by (date, id). ``runner`` replaces :func:`run_sparql` (tests).
    """
    d0 = _dt.date.fromisoformat(start)
    d1 = _dt.date.fromisoformat(end) if end else _dt.date.today()
    if d1 < d0:
        raise ValueError(f"end {d1} is before start {d0}")
    unknown = set(kinds) - set(BUILDERS)
    if unknown:
        raise ValueError(f"unknown kinds {sorted(unknown)}; known: {', '.join(BUILDERS)}")
    stop = d1 + _dt.timedelta(days=1)
    as_of = d1.isoformat()
    own = client is None
    http = client or make_client()
    kw: dict[str, Any] = {"client": http, "retries": retries, "runner": runner}
    events: list[Event] = []
    try:
        if "heads" in kinds:
            events += events_from_heads(run_windowed("heads", d0, stop, **kw), as_of=as_of)
        if "elections" in kinds:
            rows = run_windowed("elections", d0, stop, **kw)
            ctx = election_context(run_context("election_context", (r["e"] for r in rows), **kw))
            events += events_from_elections(rows, ctx)
        if "awards" in kinds:
            a0 = _dt.date(d0.year - 1, d0.month, min(d0.day, 28))
            events += events_from_awards(run_windowed("awards", a0, stop, **kw), since=start)
        if "sports" in kinds:
            rows = run_windowed("sports", d0, stop, **kw) + run_windowed("sports_series", d0, stop, **kw)
            ctx = sports_context(run_context("sports_context", (r["ed"] for r in rows), **kw))
            events += events_from_sports(rows, ctx)
    finally:
        if own:
            http.close()

    kept: dict[str, Event] = {}
    for e in events:
        if not (start <= e.date <= as_of):
            continue
        problems = validate_event(e)
        if problems:
            log.info("dropping %s: %s", e.id, "; ".join(problems))
            continue
        kept.setdefault(e.id, e)
    return sorted(kept.values(), key=lambda e: (e.date, e.id))
