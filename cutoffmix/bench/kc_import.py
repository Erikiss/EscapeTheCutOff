"""Import the apoorvumang/knowledge-cutoff benchmark ("kc") into cutoffmix records.

The kc *data* (330 dated events, 2024-01..2026-06, and per-model graded answers) is CC BY 4.0 (attribution:
apoorvumang/knowledge-cutoff). Its code has no license, so nothing here is copied from it: topic tagging, label
mapping and letter extraction are reimplemented.

* :func:`download_kc` fetches ``events.jsonl`` (Hugging Face dataset) and ``report_answers.json`` (GitHub repo).
* :func:`tag_topic` files each kc event under one of the 15 taxonomy topics: deaths (and alive/fake controls) by
  the person's occupation, office changes by the kind of office (government, business, elections, ...).
* :func:`convert_events` / :func:`convert_answers` produce :class:`Event` / :class:`ProbeResult` records
  (probe ``direct`` -> ``current``, ``mcq`` -> ``mcq``; ``grader="kc"``).
* :func:`import_kc` does all of it and writes ``kc_events.jsonl`` and ``kc_results.jsonl``.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import httpx

from cutoffmix.bench.io import ORIGIN_ATTRIBUTION, read_jsonl, save_events, save_results
from cutoffmix.bench.probegen import name_variants
from cutoffmix.bench.sources import polite_request
from cutoffmix.bench.sources.wikidata import make_client
from cutoffmix.bench.sources.wikipedia import keyword_topic
from cutoffmix.taxonomy import OCCUPATION_TO_TOPIC, topic_for_occupation
from cutoffmix.types import Event, ProbeResult

log = logging.getLogger(__name__)

KC_EVENTS_URL = "https://huggingface.co/datasets/apoorvumang/knowledge-cutoff-benchmark/resolve/main/events.jsonl"
KC_ANSWERS_URL = "https://raw.githubusercontent.com/apoorvumang/knowledge-cutoff/main/report_answers.json"
KC_ATTRIBUTION = ORIGIN_ATTRIBUTION["kc"]

CONTROL_CATEGORIES = frozenset({"control_alive", "fake_event"})
KC_PROBES = {"direct": "current", "mcq": "mcq"}


def download_kc(dest_dir: str | Path, *, client: httpx.Client | None = None,
                force: bool = False) -> tuple[Path, Path]:
    """Download kc ``events.jsonl`` and ``report_answers.json`` into ``dest_dir``; existing files are kept unless
    ``force``. Files are written atomically. Returns ``(events_path, answers_path)``."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    targets = ((KC_EVENTS_URL, dest / "events.jsonl"), (KC_ANSWERS_URL, dest / "report_answers.json"))
    own = client is None
    http = client or make_client(timeout=120.0)
    try:
        for url, path in targets:
            if path.exists() and not force:
                continue
            resp = polite_request(http, "GET", url, follow_redirects=True)
            tmp = path.with_name(path.name + ".part")
            tmp.write_bytes(resp.content)
            tmp.replace(path)
            log.info("downloaded %s (%d bytes)", path, len(resp.content))
    finally:
        if own:
            http.close()
    return targets[0][1], targets[1][1]


# --------------------------------------------------------------------------------------
# Topic tagging
# --------------------------------------------------------------------------------------


def _rx(words: Iterable[str]) -> re.Pattern[str]:
    return re.compile(r"\b(?:" + "|".join(words) + r")(?:s|es)?\b", re.I)


# Checked first, in order, on subject + descriptor: roles that decide the topic whatever else is mentioned.
_OVERRIDES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("law_crime", _rx([r"accuser", r"spied", r"convicted spy", r"gang leader", r"cartel", r"tren de aragua",
                       r"crime boss"])),
    ("conflict_security", _rx([r"hamas", r"hezbollah", r"houthi", r"militant"])),
    ("government", _rx([r"pope", r"cardinal", r"archbishop", r"aga khan", r"ismaili", r"supreme leader",
                        r"ayatollah", r"imam", r"monarch"])),
    ("business_economy", _rx([r"ceo", r"chief executive", r"executive chairman", r"chairman of", r"industrialist",
                              r"media mogul", r"billionaire", r"tycoon", r"businessman", r"businesswoman",
                              r"economist"])),
    ("ai_ml", _rx([r"openai", r"anthropic", r"deepmind", r"ai researcher", r"artificial intelligence"])),
    ("space_astronomy", _rx([r"astronaut", r"cosmonaut", r"nasa", r"astronomer", r"astrophysicist"])),
    ("science_research", _rx([r"physicist", r"chemist", r"biochemist", r"biologist", r"psychologist",
                              r"primatologist", r"mathematician", r"geneticist", r"scientist"])),
    ("health_medicine", _rx([r"physician", r"surgeon", r"epidemiologist", r"virologist"])),
)

# Earliest mention wins among these (taxonomy.OCCUPATION_TO_TOPIC with word boundaries, minus keywords that are
# ambiguous as whole words, plus the descriptors kc facts actually use).
_AMBIGUOUS = frozenset({"general", "king", "queen"})
_EXTRA: dict[str, list[str]] = {
    "sports": [r"football(?:er)?", r"soccer", r"fifa", r"world cup", r"nba", r"nfl", r"mlb", r"nhl", r"pga",
               r"baseball", r"basketball", r"golf(?:er)?", r"cricket(?:er)?", r"boxing", r"heavyweight", r"marathon",
               r"runner", r"formula one", r"racing", r"nascar", r"wrestl\w*", r"chess", r"sailor", r"paralympic",
               r"olympic\w*", r"tennis", r"super bowl", r"pitcher", r"catcher", r"wide receiver", r"defensive end",
               r"quarterback", r"forward", r"sportscaster", r"espn", r"colts", r"cowboys", r"vikings", r"grizzlies",
               r"dodgers", r"hall of famer", r"sprinter", r"jockey", r"skier"],
    "arts_entertainment": [r"actor", r"actress", r"singer", r"songwriter", r"musician", r"band", r"guitarist",
                           r"bassist", r"drummer", r"rapper", r"hype man", r"comedian", r"filmmaker", r"film",
                           r"broadway", r"composer", r"novelist", r"author", r"writer", r"poet", r"playwright",
                           r"manga", r"dancer", r"photographer", r"designer", r"architect", r"critic", r"game show",
                           r"host", r"chef", r"food network", r"reggae", r"jazz", r"hip-hop", r"pianist",
                           r"clarinetist", r"saxophonist", r"flugelhornist", r"vibraphonist", r"music", r"star",
                           r"teen idol", r"frontman", r"grammy", r"oscar", r"personality", r"blues", r"folk", r"soul",
                           r"r&b", r"rock", r"punk", r"highlife", r"afrobeat", r"comic", r"cartoonist", r"icon",
                           r"correspondent", r"broadcaster", r"producer", r"entertainer"],
    "government": [r"president", r"presidential", r"prime minister", r"senator", r"senate", r"representative",
                   r"congress\w*", r"governor", r"minister", r"chancellor", r"vice president", r"vice-president",
                   r"opposition leader", r"party", r"mayor", r"activist", r"secretary", r"parliament\w*",
                   r"lawmaker", r"taoiseach", r"nominee", r"precandidate", r"statesman", r"cia", r"white house",
                   r"far-right", r"far-left", r"nationalist", r"politic\w*"],
    "law_crime": [r"attorney", r"prosecutor", r"defendant", r"murderer", r"alias"],
    "conflict_security": [r"army", r"commander"],
    "business_economy": [r"company", r"executive", r"investor", r"microsoft", r"tesla", r"google", r"alphabet",
                         r"jpmorgan", r"apple", r"amazon"],
    "computing_software": [r"software", r"computer"],
}


def _competition() -> tuple[tuple[str, re.Pattern[str]], ...]:
    words: dict[str, list[str]] = {t: list(v) for t, v in _EXTRA.items()}
    for kw, topic in OCCUPATION_TO_TOPIC:
        if kw not in _AMBIGUOUS:
            words.setdefault(topic, []).append(re.escape(kw))
    return tuple((t, _rx(w)) for t, w in words.items())


_COMPETITION = _competition()
_DEATH_VERB = re.compile(r"\b(?:died|dies|has died|was killed|was shot|was assassinated|passed away|is alive)\b", re.I)


def _descriptor(text: str, subject: str) -> str:
    """The words describing the person: text before the subject plus the apposition after it, up to the verb."""
    for name in (subject, re.sub(r"\s*\([^)]*\)", "", subject).strip()):
        i = text.find(name) if name else -1
        if i >= 0:
            after = text[i + len(name):]
            cut = _DEATH_VERB.search(after)
            return (text[:i] + " " + (after[:cut.start()] if cut else after)).strip()
    return text


def _earliest(text: str) -> str | None:
    best: tuple[int, int, str] | None = None
    for topic, rx in _COMPETITION:
        m = rx.search(text)
        if m and (best is None or (m.start(), -len(m.group(0))) < best[:2]):
            best = (m.start(), -len(m.group(0)), topic)
    return best[2] if best else None


def person_topic(subject: str, text: str) -> str:
    """Topic of a person-centred event (death, alive/fake control) from how ``text`` describes ``subject``.

    Order: role overrides on subject + descriptor (popes -> government, CEOs -> business, physicists -> science,
    ...); else the earliest occupation word in the descriptor, then in the whole text (taxonomy occupation table
    plus kc's descriptors: "Actor and former NFL player" -> arts); else
    :func:`cutoffmix.taxonomy.topic_for_occupation` (substring match, catches compounds such as "screenwriter");
    else the keyword classifier; else ``arts_entertainment``.
    """
    desc = _descriptor(text, subject)
    blob = f"{subject} {desc}"
    for topic, rx in _OVERRIDES:
        if rx.search(blob):
            return topic
    return (_earliest(desc) or _earliest(text) or topic_for_occupation(desc, default="")
            or keyword_topic(text) or "arts_entertainment")


_OFFICE_OVERRIDES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("business_economy", _rx([r"ceo", r"chief executive", r"executive chairman", r"board", r"company"])),
    ("international_relations", _rx([r"nato", r"united nations", r"un secretary-general"])),
    ("health_medicine", _rx([r"centers for disease control", r"cdc", r"world health organization"])),
    ("law_crime", _rx([r"u\.s\. attorney for", r"prosecutor", r"supreme court justice"])),
)
_ELECTION = _rx([r"election", r"primary", r"primaries", r"referendum", r"most seats"])


def office_topic(subject: str, question: str) -> str:
    """Topic of an office change from its subject and question: business, international organisations, health
    agencies and prosecutors by keyword; a question about an election, primary or seat count -> elections;
    otherwise government (heads of state/government, ministers, party and religious leaders)."""
    text = f"{subject} {question}"
    for topic, rx in _OFFICE_OVERRIDES:
        if rx.search(text):
            return topic
    return "elections" if _ELECTION.search(question) else "government"


def tag_topic(kc_event: Mapping[str, Any]) -> str:
    """Taxonomy topic id of one kc event (raw kc dict): office changes via :func:`office_topic`, everything else
    (deaths and the alive/fake controls, which name the person) via :func:`person_topic`."""
    subject = str(kc_event.get("subject", ""))
    question = str(kc_event.get("question_direct", ""))
    if kc_event.get("category") == "office_change":
        topic = office_topic(subject, question)
    elif kc_event.get("category") == "death":
        topic = person_topic(subject, str(kc_event.get("fact", "")))
    else:
        topic = person_topic(subject, f"{question} {kc_event.get('fact', '')}")
    return topic


# --------------------------------------------------------------------------------------
# Events and answers
# --------------------------------------------------------------------------------------

_PREDECESSOR = re.compile(r"\b(?:succeed(?:ing|ed|s)?|replac(?:ing|ed|es)?|predecessor)\s+((?:[A-Z][\w.'’-]*\s*){1,6})")
_TITLES = frozenset({"President", "Prime", "Minister", "Chancellor", "Taoiseach", "CEO", "Chief", "Executive",
                     "Speaker", "Governor", "Secretary", "General", "Senator", "King", "Queen", "Pope", "First",
                     "Vice", "Archbishop", "Mayor", "Leader", "Chair", "Chairman", "The", "U.S.", "US", "Former"})
_STALE_DEATH = re.compile(r"\b(?:alive|living|still|remains?|continu\w*)\b", re.I)
_STALE_OFFICE = re.compile(r"\b(?:still|remains?|continu\w*|kept|keeps|stayed|stays)\b", re.I)


def _choice_text(choice: str) -> str:
    return re.sub(r"^\s*\(?[A-D][).:]\s*", "", choice).strip()


def _correct_choice(d: Mapping[str, Any]) -> str:
    letter = str(d.get("mcq_answer", "")).strip().upper()
    for c in d.get("mcq_choices") or []:
        if c.strip()[:1].upper() == letter:
            return _choice_text(c)
    return ""


def _predecessors(text: str) -> list[str]:
    """Names (2-4 capitalised words, leading titles dropped) that follow 'succeeding'/'replacing'/'predecessor'."""
    names = []
    for m in _PREDECESSOR.finditer(text):
        tokens = m.group(1).split()
        while tokens and tokens[0] in _TITLES:
            tokens.pop(0)
        if 2 <= len(tokens) <= 4:
            names.append(" ".join(tokens).rstrip(".’'"))
    return names


def _office_aliases(d: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """Accepted names (the correct MCQ option, when it is a short name) and stale names (the person the question
    or fact names right after 'succeeding'/'replacing'/'predecessor') of an office change."""
    correct = _correct_choice(d)
    aliases = name_variants(correct) if correct and len(correct.split()) <= 6 else []
    accepted = {a.casefold() for a in aliases}
    reject = [v for name in _predecessors(f"{d.get('question_direct', '')} {d.get('fact', '')}")
              for v in name_variants(name)]
    return aliases, [r for r in dict.fromkeys(reject) if r.casefold() not in accepted]


def convert_events(kc_events: Iterable[Mapping[str, Any]]) -> list[Event]:
    """kc event dicts -> :class:`Event` (``origin="kc"``).

    ``question_current``/``expected_current`` := kc ``question_direct``/``expected_direct``; MCQ fields pass
    through; ``topic`` from :func:`tag_topic`; ``control`` for ``control_alive``/``fake_event``. Office changes get
    the correct MCQ option as accepted alias and the predecessor (when the text names it) as reject alias. kc has
    no dated (recall) question, so ``question_recall`` stays empty.
    """
    events: list[Event] = []
    for d in kc_events:
        category = str(d["category"])
        aliases, reject = _office_aliases(d) if category == "office_change" else ([], [])
        events.append(Event(
            id=str(d["id"]), date=str(d["date"]), month=str(d["month"]), topic=tag_topic(d), category=category,
            subject=str(d.get("subject", "")), fact=str(d.get("fact", "")),
            region=str(d.get("region", "International")), predictability=d.get("predictability", "low"),
            question_current=str(d.get("question_direct", "")), expected_current=str(d.get("expected_direct", "")),
            mcq_question=str(d.get("mcq_question", "")), mcq_choices=[str(c) for c in d.get("mcq_choices") or []],
            mcq_answer=str(d.get("mcq_answer", "")).strip().upper(), aliases=aliases, reject_aliases=reject,
            source=str(d.get("source", "")), origin="kc", control=category in CONTROL_CATEGORIES,
        ))
    return events


_LETTER_PATTERNS = (
    re.compile(r"\s*\(?([A-D])\)?[.):]?\b", re.I),  # leading letter
    re.compile(r"answer\s*(?:is|:)?\s*\(?([A-D])\)?", re.I),  # "the answer is C"
)


def _chosen_letter(response: str, letters: Iterable[str]) -> str | None:
    """The letter an MCQ response chose, in kc's order of precedence: leading letter, "answer is X", then the
    first standalone capital A-D that is a valid option."""
    text = response.strip()
    if not text:
        return None
    m = _LETTER_PATTERNS[0].match(text) or _LETTER_PATTERNS[1].search(text)
    if m:
        return m.group(1).upper()
    valid = set(letters)
    return next((ch for ch in re.findall(r"\b([A-D])\b", text) if ch in valid), None)


def _stale_letters(event: Event) -> set[str]:
    """MCQ options that state the pre-event world (the person alive, the predecessor still in office)."""
    out = set()
    for c in event.mcq_choices:
        text = _choice_text(c)
        stale = (_STALE_DEATH.search(text) if event.category == "death"
                 else _STALE_OFFICE.search(text) or any(r.casefold() in text.casefold() for r in event.reject_aliases))
        if stale:
            out.add(c.strip()[:1].upper())
    return out


def _wrong_grade(event: Event, probe: str, response: str) -> str:
    stale_category = event.category in ("death", "office_change")
    if probe == "mcq":
        letter = _chosen_letter(response, [c.strip()[:1].upper() for c in event.mcq_choices])
        if letter is not None:
            return "outdated" if letter in _stale_letters(event) else "incorrect"
    return "outdated" if stale_category else "incorrect"


def convert_answers(report_answers: Mapping[str, Any], events: Mapping[str, Event],
                    models: Iterable[str] | None = None) -> list[ProbeResult]:
    """kc ``report_answers.json`` (``{model: {probe: {event_id: {"l": code, "r": response}}}}``) ->
    :class:`ProbeResult` with ``grader="kc"``.

    Probes: ``direct`` -> ``current``, ``mcq`` -> ``mcq``. Labels: ``c`` -> correct, ``a`` -> abstain,
    ``u`` -> ungraded, ``w`` -> outdated for death/office_change events and incorrect otherwise; for MCQ answers
    whose letter can be read, ``w`` is outdated only if the chosen option states the pre-event world. Events not in
    ``events`` are skipped (logged); ``models`` restricts the models (kc keys equal cutoffmix model keys).
    """
    wanted = set(models) if models is not None else None
    results: list[ProbeResult] = []
    skipped = 0
    for model in sorted(report_answers):
        if wanted is not None and model not in wanted:
            continue
        for kc_probe, answers in report_answers[model].items():
            probe = KC_PROBES.get(kc_probe)
            if probe is None:
                raise ValueError(f"unknown kc probe {kc_probe!r} for model {model!r}")
            for eid, entry in answers.items():
                event = events.get(eid)
                if event is None:
                    skipped += 1
                    continue
                code, response = entry.get("l"), str(entry.get("r", ""))
                if code == "w":
                    grade = _wrong_grade(event, probe, response)
                elif code in ("c", "a", "u"):
                    grade = {"c": "correct", "a": "abstain", "u": "ungraded"}[code]
                else:
                    raise ValueError(f"unknown kc label {code!r} ({model}/{kc_probe}/{eid})")
                results.append(ProbeResult(event_id=eid, model=model, probe=probe, response=response,  # type: ignore[arg-type]
                                           grade=grade, grader="kc"))  # type: ignore[arg-type]
    if skipped:
        log.warning("convert_answers: skipped %d answers for unknown events", skipped)
    return results


def import_kc(out_dir: str | Path, *, client: httpx.Client | None = None, models: Iterable[str] | None = None,
              force: bool = False) -> tuple[Path, Path]:
    """Download kc into ``out_dir`` and write ``kc_events.jsonl`` and ``kc_results.jsonl`` there.

    Returns ``(events_path, results_path)``. ``models`` restricts the imported answers; ``force`` re-downloads.
    """
    out = Path(out_dir)
    events_raw, answers_raw = download_kc(out, client=client, force=force)
    events = convert_events(read_jsonl(events_raw))
    results = convert_answers(json.loads(answers_raw.read_text(encoding="utf-8")), {e.id: e for e in events},
                              models)
    events_path, results_path = out / "kc_events.jsonl", out / "kc_results.jsonl"
    save_events(events_path, events)
    save_results(results_path, results)
    log.info("kc import: %d events, %d results (%s)", len(events), len(results), KC_ATTRIBUTION)
    return events_path, results_path
