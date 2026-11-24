"""English Wikipedia event sources: the Current events portal and the monthly "Deaths in <Month> <Year>" lists.

Fetching uses the MediaWiki Action API politely: batched ``prop=revisions`` queries (50 titles per request),
``maxlag=5``, ``Retry-After`` honoured on 429/503 and on maxlag errors, one request at a time with a pause between
batches, and the project User-Agent (:data:`cutoffmix.bench.sources.wikidata.USER_AGENT`). Wikipedia text is
CC BY-SA 4.0; keep the page title and revision when redistributing parsed items.

Parsing is pure: :func:`parse_ce_wikitext` turns one portal day page into leaf items, :func:`parse_deaths` one
deaths list (or the month section of "Deaths in <Year>") into entries, and :func:`ce_item_topic` files a
Current-events item under one of the 15 taxonomy topics.
"""

from __future__ import annotations

import datetime as _dt
import difflib
import functools
import logging
import re
from typing import Any, Iterable, Mapping

import httpx

from cutoffmix.bench import sources
from cutoffmix.bench.sources import SourceError, backoff_delay, polite_request, retry_after_seconds
from cutoffmix.bench.sources.wikidata import USER_AGENT, make_client
from cutoffmix.taxonomy import CE_SECTION_TO_TOPICS, TOPIC_IDS

log = logging.getLogger(__name__)

API_URL = "https://en.wikipedia.org/w/api.php"
BATCH_TITLES = 50
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December")


# --------------------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------------------


def _api_query(client: httpx.Client, params: Mapping[str, str], *, retries: int) -> dict[str, Any]:
    """One Action API GET with ``maxlag=5``; maxlag errors (HTTP 200 + error code) wait and retry."""
    full = {"format": "json", "formatversion": "2", "maxlag": "5", **params}
    attempt = 0
    while True:
        resp = polite_request(client, "GET", API_URL, params=full, retries=retries,
                              headers={"User-Agent": USER_AGENT})
        data = resp.json()
        error = data.get("error")
        if not error:
            return data
        if error.get("code") != "maxlag" or attempt >= retries:
            raise SourceError(f"MediaWiki API error {error.get('code')}: {error.get('info', '')}")
        delay = backoff_delay(attempt, retry_after_seconds(resp.headers.get("Retry-After")) or 5.0)
        log.info("Wikipedia API maxlag; retry %d/%d in %.1fs", attempt + 1, retries, delay)
        sources.pause(delay)
        attempt += 1


def fetch_pages(titles: Iterable[str], *, client: httpx.Client | None = None, redirects: bool = False,
                retries: int = 6, pause: float = 1.0) -> dict[str, dict[str, Any]]:
    """Fetch the current wikitext of ``titles`` in batches of :data:`BATCH_TITLES`, one request at a time.

    Returns ``{requested title: {"title", "content", "timestamp", "revid", "fragment"}}``; missing pages are
    absent. ``title`` is the page actually read (after normalisation and, with ``redirects``, redirects);
    ``fragment`` is a redirect's section anchor or "". ``pause`` seconds separate batches.
    """
    wanted = list(dict.fromkeys(titles))
    own = client is None
    http = client or make_client()
    out: dict[str, dict[str, Any]] = {}
    try:
        for i in range(0, len(wanted), BATCH_TITLES):
            if i:
                sources.pause(pause)
            batch = wanted[i:i + BATCH_TITLES]
            params = {"action": "query", "prop": "revisions", "rvprop": "content|timestamp|ids", "rvslots": "main",
                      "titles": "|".join(batch)}
            if redirects:
                params["redirects"] = "1"
            query = _api_query(http, params, retries=retries).get("query", {})
            resolved = {t: (t, "") for t in batch}
            for step in ("normalized", "redirects"):
                hops = {h["from"]: (h["to"], h.get("tofragment", "")) for h in query.get(step, [])}
                resolved = {t: (hops[cur][0], hops[cur][1] or frag) if cur in hops else (cur, frag)
                            for t, (cur, frag) in resolved.items()}
            pages = {p["title"]: p for p in query.get("pages", []) if p.get("revisions")}
            for title, (target, fragment) in resolved.items():
                page = pages.get(target)
                if page is None:
                    continue
                rev = page["revisions"][0]
                out[title] = {"title": target, "content": rev["slots"]["main"]["content"],
                              "timestamp": rev.get("timestamp", ""), "revid": rev.get("revid"), "fragment": fragment}
            log.info("Wikipedia: %d/%d pages in batch %d", sum(t in out for t in batch), len(batch),
                     i // BATCH_TITLES + 1)
    finally:
        if own:
            http.close()
    return out


def ce_title(day: _dt.date) -> str:
    """Portal page title of one day: ``Portal:Current events/2026 March 3``."""
    return f"Portal:Current events/{day.year} {MONTHS[day.month - 1]} {day.day}"


def fetch_ce_days(start: _dt.date, end: _dt.date, *, client: httpx.Client | None = None) -> dict[_dt.date, str]:
    """Wikitext of the Current events portal day pages from ``start`` to ``end`` (inclusive); days without a page
    are absent. Batched (50 titles per request), maxlag=5, Retry-After honoured, one request at a time."""
    if end < start:
        raise ValueError(f"end {end} is before start {start}")
    days = [start + _dt.timedelta(days=n) for n in range((end - start).days + 1)]
    pages = fetch_pages((ce_title(d) for d in days), client=client)
    return {d: pages[ce_title(d)]["content"] for d in days if ce_title(d) in pages}


def deaths_title(year: int, month: int) -> str:
    """``Deaths in October 2026``."""
    return f"Deaths in {MONTHS[month - 1]} {year}"


def fetch_deaths_month(year: int, month: int, *, client: httpx.Client | None = None) -> str:
    """Wikitext of "Deaths in <Month> <Year>". The current month usually redirects to "Deaths in <Year>#<Month>";
    then the whole year page is returned and :func:`parse_deaths` narrows it to the month section."""
    title = deaths_title(year, month)
    page = fetch_pages([title], client=client, redirects=True).get(title)
    if page is None:
        raise SourceError(f"Wikipedia page {title!r} not found")
    return str(page["content"])


# --------------------------------------------------------------------------------------
# Current events portal
# --------------------------------------------------------------------------------------

CE_CATEGORIES: tuple[str, ...] = (
    "Armed conflicts and attacks", "Arts and culture", "Business and economy", "Disasters and accidents",
    "Health and environment", "International relations", "Law and crime", "Politics and elections",
    "Science and technology", "Sports",
)
_CE_ALIASES = {
    "attacks and armed conflicts": "Armed conflicts and attacks", "armed conflict and attacks":
    "Armed conflicts and attacks", "arts and entertainment": "Arts and culture", "music": "Arts and culture",
    "business and economics": "Business and economy", "disasters": "Disasters and accidents",
    "health and environments": "Health and environment", "health": "Health and environment",
    "environment": "Health and environment", "international affairs": "International relations",
    "law & crime": "Law and crime", "politics and election": "Politics and elections",
    "elections": "Politics and elections", "science and technologies": "Science and technology",
    "sport": "Sports",
}
_COMMENT = re.compile(r"<!--.*?-->", re.S)
_HEADING = re.compile(r"^\s*(?:'''\s*(.+?)\s*'''|;\s*(.+?))\s*$")
_WIKILINK = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|([^\]]*))?\]\]")
_EXTLINK = re.compile(r"\[(https?://\S+)\s+([^\]]*)\]")
_BULLET = re.compile(r"^(\*+)\s*(.*)$")


def canonical_ce_category(heading: str) -> str | None:
    """Map a portal heading to one of :data:`CE_CATEGORIES` (aliases, then fuzzy match); None if unknown."""
    h = re.sub(r"[^a-z& ]", "", heading.lower()).strip()
    for c in CE_CATEGORIES:
        if h == c.lower():
            return c
    if h in _CE_ALIASES:
        return _CE_ALIASES[h]
    match = difflib.get_close_matches(h, [c.lower() for c in CE_CATEGORIES], n=1, cutoff=0.8)
    return next(c for c in CE_CATEGORIES if c.lower() == match[0]) if match else None


def plain_text(wikitext: str) -> str:
    """Wikitext fragment -> plain text: external links and refs removed, wikilinks replaced by their label,
    bold/italic quotes and simple templates dropped, whitespace collapsed."""
    s = _EXTLINK.sub("", wikitext)
    s = re.sub(r"<ref[^>]*?/>|<ref.*?</ref>", "", s, flags=re.S)
    s = _WIKILINK.sub(lambda m: m.group(2) if m.group(2) else m.group(1), s)
    s = re.sub(r"'{2,}", "", s)
    s = re.sub(r"\{\{[^{}]*\}\}", "", s)
    return re.sub(r"\s+", " ", s).strip()


def parse_ce_wikitext(text: str, day: _dt.date) -> list[dict[str, Any]]:
    """Parse one portal day page into leaf items ``{date, ce_category, story_path, text, links, sources}``.

    Headings (``'''Armed conflicts and attacks'''``) set ``ce_category`` (canonical name, None if unknown).
    Bullets nest stories (``*[[Story]]`` / ``**[[Sub-story]]`` / ``***Leaf. [url (Outlet)]``); a bullet is a leaf
    when the next line is not deeper, and is kept if it has a source link or more than 60 characters of text.
    ``story_path`` holds the first link target of each ancestor bullet, ``links`` the leaf's wikilink targets and
    ``sources`` its ``{url, outlet}`` external links. Both the ``content=`` wrapper and the newer
    ``top=yes``/``bottom=yes`` wrapper templates are handled (single-line patterns only, so no content is lost).
    """
    text = _COMMENT.sub("\n", text)
    text = re.sub(r"\{\{Current events\|[^{}\n]*?content=", "\n", text)
    text = re.sub(r"\{\{Current events\|[^{}\n]*?\}\}", "\n", text)
    lines = text.splitlines()
    category: str | None = None
    stack: list[str] = []
    out: list[dict[str, Any]] = []
    for idx, line in enumerate(lines):
        heading = _HEADING.match(line)
        if heading and not line.lstrip().startswith("*"):
            category = canonical_ce_category(heading.group(1) or heading.group(2))
            stack = []
            continue
        bullet = _BULLET.match(line)
        if not bullet:
            continue
        depth, body = len(bullet.group(1)), bullet.group(2)
        stack = stack[:depth - 1] + [body]
        nxt = _BULLET.match(lines[idx + 1]) if idx + 1 < len(lines) else None
        if nxt and len(nxt.group(1)) > depth:
            continue
        sources = [{"url": u, "outlet": re.sub(r"[()']", "", o).strip()} for u, o in _EXTLINK.findall(body)]
        plain = plain_text(body)
        if not sources and len(plain) <= 60:
            continue
        path = []
        for ancestor in stack[:-1]:
            links = _WIKILINK.findall(ancestor)
            path.append(links[0][0].strip() if links else plain_text(ancestor))
        out.append({"date": day.isoformat(), "ce_category": category, "story_path": path, "text": plain,
                    "links": sorted({m[0].strip() for m in _WIKILINK.findall(body)}), "sources": sources})
    return out


# Refinement rules over text + story path + links (lower case), applied within the heading's candidate topics.
_RULES: dict[str, re.Pattern[str]] = {k: re.compile(v) for k, v in {
    "ai": r"\b(artificial intelligence|ai|a\.i\.|chatgpt|openai|anthropic|claude|gemini|llms?|large language models?"
          r"|deepfakes?|deepmind|xai|grok|mistral ai|deepseek|machine learning|chatbots?|generative ai)\b",
    "space": r"\b(spaceflight|spacecraft|nasa|esa|spacex|rockets?|satellites?|orbit(al)?|moon|lunar|mars|"
             r"asteroids?|comets?|telescopes?|astronauts?|cosmonauts?|starship|artemis|launch vehicle|exoplanets?|"
             r"eclipse|space station|isro|cnsa|jaxa|roscosmos)\b",
    "tech": r"\b(software|smartphones?|iphone|apple inc|microsoft|google|meta platforms|semiconductors?|chips?|"
            r"cyberattacks?|ransomware|hack(ers?|ing|ed)?|data breach|internet|social media|tiktok|"
            r"telecommunications?|quantum comput\w*|outage|operating system|app store|cybersecurity)\b",
    "cyber": r"\b(cyberattacks?|ransomware|hack(ers?|ing|ed)?|data breach|cybercrime|malware|spyware)\b",
    "elections": r"\b(elections?|referendum|ballots?|polls?|primary|primaries|runoff|electoral|vote count|"
                 r"exit polls?|voters?|by-election|snap election)\b",
    "climate": r"\b(climate|emissions?|cop\d\d|heatwaves?|heat wave|wildfires?|drought|deforestation|pollution|"
               r"biodiversity|glaciers?|carbon|renewable|environment(al)?|endangered|extinct|oil spill|smog)\b",
    "health": r"\b(disease|outbreaks?|epidemic|pandemic|vaccines?|virus|cholera|measles|mpox|ebola|dengue|"
              r"influenza|bird flu|h5n1|world health organization|who|fda|drugs?|hospitals?|cancer|health|"
              r"medical|patients?|malaria|tuberculosis|polio)\b",
    "natural": r"\b(earthquakes?|floods?|flooding|cyclones?|hurricanes?|typhoons?|wildfires?|landslides?|volcan\w*|"
               r"eruptions?|storms?|heatwaves?|heat wave|drought|tsunami|avalanche|tornado(es)?|blizzard|monsoon|"
               r"magnitude)\b",
}.items()}


def keyword_topic(text: str, candidates: Iterable[str] = TOPIC_IDS) -> str | None:
    """Best of ``candidates`` for ``text`` by the zero-training keyword classifier (``cutoffmix.router.topics``,
    imported lazily); None when that module is unavailable or the text carries no signal for them."""
    classifier = _classifier()
    if classifier is None:
        return None
    post = dict(zip(classifier.topics, classifier.posterior(text)))
    scores = {t: post.get(t, 0.0) for t in candidates}
    best = max(scores, key=lambda t: scores[t])
    return best if scores[best] > 1.0 / len(classifier.topics) + 1e-9 else None


@functools.lru_cache(maxsize=1)
def _classifier() -> Any:
    try:
        from cutoffmix.router.topics import KeywordTopicClassifier
    except ImportError:
        log.info("cutoffmix.router.topics unavailable; keyword topic fallback disabled")
        return None
    return KeywordTopicClassifier()


def ce_item_topic(item: Mapping[str, Any]) -> str:
    """Topic id for a Current-events item.

    The heading's candidate topics come from :data:`cutoffmix.taxonomy.CE_SECTION_TO_TOPICS`; regex rules refine
    within them (AI, space, software/cyber, elections, climate, health, natural vs human-made disasters) and the
    keyword classifier breaks the remaining ties. Items without a known heading go to the classifier over all
    topics (default ``conflict_security``, the largest section).
    """
    category = item.get("ce_category")
    blob = " ".join([str(item.get("text", "")), " ".join(map(str, item.get("story_path") or [])),
                     " ".join(map(str, item.get("links") or []))]).lower()

    def hit(rule: str) -> bool:
        return _RULES[rule].search(blob) is not None

    if category is None or category not in CE_SECTION_TO_TOPICS:
        return keyword_topic(blob, TOPIC_IDS) or "conflict_security"
    if hit("ai") and category in ("Science and technology", "Business and economy", "Law and crime",
                                  "Politics and elections", "International relations"):
        return "ai_ml"
    if category == "Science and technology":
        if hit("space"):
            return "space_astronomy"
        if hit("tech"):
            return "computing_software"
        return "science_research"
    if category == "Health and environment":
        climate, health = hit("climate"), hit("health")
        if climate != health:
            return "natural_disasters_climate" if climate else "health_medicine"
        return keyword_topic(blob, ("health_medicine", "natural_disasters_climate")) or "health_medicine"
    if category == "Disasters and accidents":
        return "natural_disasters_climate" if hit("natural") else "accidents"
    if category == "Politics and elections":
        return "elections" if hit("elections") else "government"
    if category in ("Law and crime", "Business and economy") and hit("cyber"):
        return "computing_software"
    return CE_SECTION_TO_TOPICS[category][0]


# --------------------------------------------------------------------------------------
# Deaths lists
# --------------------------------------------------------------------------------------

_DEATH_ENTRY = re.compile(
    r"^\*\s*\[\[([^\]|]+)(?:\|([^\]]*))?\]\]\s*,\s*(\d{1,3}|c\.\s?\d+|\d+[–-]\d+)?\s*,?\s*(.*)$")
_DAY_HEADING = re.compile(r"^===\s*(\d{1,2})\s*===")
_DATE_NOTE = re.compile(r"\.?\s*\([^()]*on this date\)\s*$")  # "(death announced on this date)"


_CAUSE_WORDS = re.compile(
    r"cancer|attack|failure|stroke|disease|complication|pneumonia|collision|crash|accident|shot|killed|suicide|"
    r"drown|fall|covid|infection|sepsis|arrest|illness|alzheimer|parkinson|\bals\b|leuk|lymphoma|tumou?r|injur|"
    r"overdose|murder|homicide|bleeding|ha?emorrhage|aneurysm|cirrhosis|dementia|euthanasia|execut|explosion|"
    r"fire|airstrike|syndrome|insufficiency|sclerosis|diabetes|asphyxia|poison|natural causes|old age")


def _split_cause(desc: str) -> tuple[str, str | None]:
    """Split "Nationality occupation (works), cause" at the last comma outside parentheses when the tail starts
    in lower case and names a cause of death (a lower-case tail can also be more description: "lawyer, jurist")."""
    depth, cut = 0, -1
    for i, ch in enumerate(desc):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            cut = i
    if cut > 0:
        tail = desc[cut + 1:].strip()
        if tail[:1].islower() and "(" not in tail and _CAUSE_WORDS.search(tail):
            return desc[:cut].strip(), tail
    return desc.strip(), None


def parse_deaths(text: str, year: int, month: int) -> list[dict[str, Any]]:
    """Parse a deaths list into ``{date, title, name, age, desc, cause, ref_urls}`` entries.

    Entries look like ``*[[Name]], 93, British actor (Star Trek), cancer.<ref>[url title]</ref>`` under
    ``===<day>===`` headings. For a year page ("Deaths in 2026") only the ``==<Month>==`` section is read (none
    if the page has month sections but not this one); text without month headings is read whole.
    ``age`` is an int when given as a plain number, ``cause`` None when not stated.
    """
    section = re.search(rf"^==\s*{MONTHS[month - 1]}(?:\s+{year})?\s*==\s*$(.*?)(?=^==[^=]|\Z)", text, re.M | re.S)
    if section is None and re.search(rf"^==\s*(?:{'|'.join(MONTHS)})\b", text, re.M):
        return []  # a page with month sections, but not this month
    body = section.group(1) if section else text
    day: int | None = None
    out: list[dict[str, Any]] = []
    for line in body.splitlines():
        heading = _DAY_HEADING.match(line)
        if heading:
            day = int(heading.group(1))
            continue
        if day is None:
            continue
        m = _DEATH_ENTRY.match(line)
        if not m:
            continue
        refs = re.findall(r"<ref[^>]*>\s*\[(https?://\S+)", line)
        desc, cause = _split_cause(_DATE_NOTE.sub("", plain_text(m.group(4))).rstrip(". "))
        age = m.group(3)
        title = m.group(1).strip()
        out.append({"date": f"{year:04d}-{month:02d}-{day:02d}", "title": title,
                    "name": (m.group(2) or title).strip(), "age": int(age) if age and age.isdigit() else None,
                    "desc": desc, "cause": cause, "ref_urls": refs})
    return out
