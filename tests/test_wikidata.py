"""Tests for cutoffmix.bench.sources.wikidata: SPARQL transport, windowing and the template event builders.

All WDQS traffic is faked (httpx.MockTransport or a fake runner over the JSON fixtures); no network.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import urllib.parse
from pathlib import Path
from typing import Any

import httpx
import pytest

from cutoffmix.bench import sources
from cutoffmix.bench.probegen import date_tokens, validate_event
from cutoffmix.bench.sources import SourceError
from cutoffmix.bench.sources import wikidata as wd

FIX = Path(__file__).parent / "fixtures"


def rows(name: str) -> list[dict[str, Any]]:
    return wd.flatten_bindings(json.loads((FIX / name).read_text(encoding="utf-8")))


def by_id(events):
    return {e.id: e for e in events}


def choice_texts(event) -> list[str]:
    return [c[3:] for c in event.mcq_choices]


def correct_text(event) -> str:
    return event.mcq_choices["ABCD".index(event.mcq_answer)][3:]


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    waits: list[float] = []
    monkeypatch.setattr(sources, "pause", waits.append)
    return waits


# --------------------------------------------------------------------------------------
# transport
# --------------------------------------------------------------------------------------


def test_flatten_bindings_types():
    (row,) = [r for r in rows("wd_heads.json") if r["personLabel"] == "Ilie Bolojan"]
    assert row["person"] == "Q18538164" and row["office"] == "Q15304810"
    assert row["prec"] == 11 and row["sitelinks"] == 25
    assert row["start"] == "2025-06-23T00:00:00Z" and row["end"] == "2026-10-20T00:00:00Z"
    assert "nature" not in row
    award = rows("wd_awards.json")[0]
    assert award["human"] is True


def _sparql_json(n: int = 1) -> dict:
    return {"head": {"vars": ["x"]}, "results": {"bindings": [
        {"x": {"type": "uri", "value": f"http://www.wikidata.org/entity/Q{i}"}} for i in range(n)]}}


def test_run_sparql_posts_form_with_headers():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_sparql_json(2))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert wd.run_sparql("SELECT ?x WHERE {}", client=client) == [{"x": "Q0"}, {"x": "Q1"}]
    req = seen[0]
    assert req.method == "POST" and str(req.url) == wd.WDQS_URL
    assert req.headers["Accept"] == "application/sparql-results+json"
    assert req.headers["User-Agent"] == wd.USER_AGENT == "cutoffmix/0.1 (https://github.com/Erikiss/EscapeTheCutOff)"
    assert urllib.parse.parse_qs(req.content.decode())["query"] == ["SELECT ?x WHERE {}"]


def test_run_sparql_honours_retry_after(no_waiting):
    replies = iter([httpx.Response(429, headers={"Retry-After": "7"}),
                    httpx.Response(503, headers={"Retry-After": "3"}),
                    httpx.Response(200, json=_sparql_json(1))])
    with httpx.Client(transport=httpx.MockTransport(lambda r: next(replies))) as client:
        assert wd.run_sparql("q", client=client) == [{"x": "Q0"}]
    assert no_waiting == [7.0, 3.0]


def test_run_sparql_timeout_and_errors(no_waiting):
    calls = []

    def timeout(request):
        calls.append(request)
        return httpx.Response(500, text="java.util.concurrent.TimeoutException")

    with httpx.Client(transport=httpx.MockTransport(timeout)) as client:
        with pytest.raises(SourceError) as err:
            wd.run_sparql("q", client=client, retries=2)
    assert err.value.timeout and err.value.status == 500 and len(calls) == 3
    assert no_waiting == [2.0, 4.0]  # exponential back-off without Retry-After

    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, text='{"results": {"bind'))) as c:
        with pytest.raises(SourceError) as err:
            wd.run_sparql("q", client=c)
    assert err.value.timeout  # WDQS ends timed-out queries inside a 200 response

    calls.clear()

    def bad(request):
        calls.append(request)
        return httpx.Response(400, text="MalformedQueryException")

    with httpx.Client(transport=httpx.MockTransport(bad)) as client:
        with pytest.raises(SourceError, match="HTTP 400") as err:
            wd.run_sparql("q", client=client)
    assert not err.value.timeout and len(calls) == 1  # client errors are not retried


def test_render_query():
    q = wd.render_query("heads", start=dt.date(2024, 1, 1), end="2025-01-01")
    assert '"2024-01-01T00:00:00Z"^^xsd:dateTime' in q and '"2025-01-01T00:00:00Z"^^xsd:dateTime' in q
    assert "__" not in q
    q = wd.render_query("election_context", values=["Q1", "not-a-qid", "Q2"])
    assert "VALUES ?e { wd:Q1 wd:Q2 }" in q
    awards = wd.render_query("awards", start="2023-01-01", end="2024-01-01")
    assert all(f"wd:{qid}" in awards for qid in wd.AWARDS)
    with pytest.raises(KeyError, match="unknown query"):
        wd.render_query("films")
    with pytest.raises(ValueError, match="start and end"):
        wd.render_query("heads")
    with pytest.raises(ValueError, match="QID"):
        wd.render_query("sports_context", values=[])
    assert set(wd.QUERIES) >= {"heads", "elections", "awards", "sports"}


def test_year_windows():
    w = wd.year_windows(dt.date(2024, 1, 1), dt.date(2026, 11, 25))
    assert w == [(dt.date(2024, 1, 1), dt.date(2025, 1, 1)), (dt.date(2025, 1, 1), dt.date(2026, 1, 1)),
                 (dt.date(2026, 1, 1), dt.date(2026, 11, 25))]
    assert wd.year_windows(dt.date(2024, 1, 31), dt.date(2024, 4, 1), months=1)[0] == (dt.date(2024, 1, 31),
                                                                                     dt.date(2024, 2, 28))


def test_run_windowed_splits_timeouts():
    asked: list[tuple[str, str]] = []

    def runner(query, client, retries):
        start, end = re.findall(r'"(\d{4}-\d{2}-\d{2})T00:00:00Z"', query)[:2]
        asked.append((start, end))
        if (dt.date.fromisoformat(end) - dt.date.fromisoformat(start)).days > 100:
            raise SourceError("timeout", status=500, timeout=True)
        return [{"window": (start, end)}]

    out = wd.run_windowed("heads", dt.date(2024, 1, 1), dt.date(2025, 1, 1), client=None, runner=runner)
    windows = [r["window"] for r in out]
    assert windows[0][0] == "2024-01-01" and windows[-1][1] == "2025-01-01"
    assert all(a[1] == b[0] for a, b in zip(windows, windows[1:]))  # contiguous, no gaps or overlaps
    assert len(windows) == 4 and len(asked) == 7

    def broken(query, client, retries):
        raise SourceError("HTTP 403", status=403)

    with pytest.raises(SourceError, match="403"):
        wd.run_windowed("heads", dt.date(2024, 1, 1), dt.date(2025, 1, 1), client=None, runner=broken)


def test_run_context_chunks():
    seen = []

    def runner(query, client, retries):
        seen.append(re.search(r"VALUES \?e \{ (.*?) \}", query).group(1).split())
        return []

    wd.run_context("election_context", [f"Q{i}" for i in range(450)] + ["Q1"], client=None, runner=runner)
    assert [len(s) for s in seen] == [200, 200, 50]


# --------------------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------------------


def test_pick_distractors():
    picked = wd.pick_distractors("e", [["A", "Q123", "b"], ["C", "D"]], exclude=["B"], n=2)
    assert picked == ["A", "C"] or picked == ["A", "D"]
    assert wd.pick_distractors("e", [["Ülle Madise"]], exclude=["Ulle Madise"], n=1) == []


@pytest.mark.parametrize("a,b,same", [
    ("FC Bayern Munich", "FC Bayern München", True),
    ("Ana Brnabić", "ana brnabic", True),
    ("2025 Indiana Hoosiers football team", "Indiana Hoosiers football", True),
    ("Real Madrid", "Madrid", False),
    ("Portugal men's national football team", "Spain men's national football team", False),
    ("Spain men's national football team", "Spain women's national football team", False),
    ("Oscar Piastri", "Oscar Pistorius", False),
    ("Joe Biden", "Jill Biden", False),
    ("Manchester United", "Manchester City", False),
])
def test_same_name(a, b, same):
    assert wd.same_name(a, b) is same


def test_sports_duplicates_ambiguity_and_label_variants():
    def row(ed, label, comp, comp_label, winner, date, links=5, sport="association football"):
        return {"ed": ed, "edLabel": label, "comp": comp, "compLabel": comp_label, "winner": winner + "Q",
                "winnerLabel": winner, "date": date + "T00:00:00Z", "sportLabel": sport, "edLinks": links}

    data = [
        # one final described by two items under two competition items with the same label -> one event
        row("Q1", "2026 Example World Cup", "Q10", "Example World Cup", "Spain men's national football team",
            "2026-07-19", links=40),
        row("Q2", "2026 Example World Cup final", "Q11", "Example World Cup", "Spain men's national football team",
            "2026-07-19", links=12),
        # men's and women's leagues sharing a label, two days apart -> no current question for either
        row("Q3", "2025 Example League (men)", "Q20", "Example League", "Club One", "2025-05-05"),
        row("Q4", "2025 Example League (women)", "Q21", "Example League", "Club Two", "2025-05-03"),
        # the previous champion under a label variant counts as a repeat champion
        row("Q5", "2025-26 Liga", "Q30", "Liga", "FC Bayern Munich", "2026-05-16"),
    ]
    ctx = {"Q5": {"dates": [("2026-05-16", 11)], "prev": {"FC Bayern München"}, "prev_human": set(),
                  "cands": set(), "human": set()}}
    events = by_id(wd.events_from_sports(data, ctx))
    assert set(events) == {"wd-sports-Q1", "wd-sports-Q3", "wd-sports-Q4", "wd-sports-Q5"}
    assert events["wd-sports-Q3"].question_current == "" and events["wd-sports-Q4"].question_current == ""
    bayern = events["wd-sports-Q5"]
    assert bayern.predictability == "high" and bayern.reject_aliases == []
    assert "FC Bayern München" not in choice_texts(bayern)


def test_series_name():
    assert wd.series_name("2026 Brazilian presidential election") == ("Brazilian presidential election", False)
    assert wd.series_name("2025–26 Exampleland general election") == ("Exampleland general election", False)
    assert wd.series_name("March 2026 Bihar election") == ("Bihar election", False)
    assert wd.series_name("51st General Election of the House") == ("General Election of the House", True)


def test_events_from_heads():
    events = by_id(wd.events_from_heads(rows("wd_heads.json"), as_of="2026-11-24"))
    assert len(events) == 8  # acting holder, co-holders, collective office, QID label, year precision dropped
    names = {e.subject for e in events.values()}
    assert names == {"Gabriel Attal", "Michel Barnier", "François Bayrou", "Donald Trump", "Ilie Bolojan",
                     "Ülle Madise", "Luca Niculescu", "Aleksandar Čupić"}
    for e in events.values():
        assert validate_event(e) == [] and e.topic == "government" and e.category == "office_change"
        assert correct_text(e) == e.subject and e.origin == "wikidata"
        assert date_tokens(e.question_current) == []
        assert e.source == f"https://www.wikidata.org/wiki/{e.id.rsplit('-', 1)[1]}"

    cupic = events["wd-office-Q889830-2026-11-19-Q142195374"]
    assert cupic.question_current == "Who is the President of Serbia?"
    assert cupic.question_recall == "Who became President of Serbia in November 2026?"
    assert cupic.aliases == ["Aleksandar Čupić", "Aleksandar Cupic"]
    assert {"Ana Brnabić", "Aleksandar Vučić"} <= set(cupic.reject_aliases)
    assert "Aleksandar Vučić" in choice_texts(cupic)  # the substantive predecessor, not the acting one
    assert "Ana Brnabić" not in choice_texts(cupic)

    niculescu = events["wd-office-Q15304810-2026-10-20-Q121140873"]
    assert niculescu.question_current == "Who is the prime minister of Romania?"
    assert {"Ilie Bolojan", "Cătălin Predoiu"} <= set(choice_texts(niculescu))  # stale + same-country distractor

    bolojan = events["wd-office-Q15304810-2025-06-23-Q18538164"]
    assert bolojan.question_current == "" and bolojan.expected_current == ""  # superseded
    assert bolojan.mcq_question == bolojan.question_recall == "Who became prime minister of Romania in June 2025?"

    attal = events["wd-office-Q1587677-2024-01-09-Q30339350"]
    assert attal.question_current == "" and attal.reject_aliases == ["Élisabeth Borne", "Elisabeth Borne"]
    assert events["wd-office-Q1587677-2024-12-13-Q439632"].question_current == "Who is the Prime Minister of France?"
    assert events["wd-office-Q11696-2025-01-20-Q22686"].region == "US"
    assert events["wd-office-Q890005-2026-10-12-Q16407395"].region == "International"


def test_heads_end_date_relative_to_as_of():
    early = by_id(wd.events_from_heads(rows("wd_heads.json"), as_of="2026-10-19"))
    # as of 2026-10-19 Bolojan's statement has not ended yet, but Niculescu (2026-10-20) is in the data
    assert early["wd-office-Q15304810-2025-06-23-Q18538164"].question_current == ""
    only_bolojan = [r for r in rows("wd_heads.json") if r["personLabel"] == "Ilie Bolojan"]
    (alone,) = wd.events_from_heads(only_bolojan, as_of="2026-10-19")
    assert alone.question_current == "Who is the prime minister of Romania?"
    (ended,) = wd.events_from_heads(only_bolojan, as_of="2026-10-20")
    assert ended.question_current == ""


def test_events_from_elections_with_context():
    ctx = wd.election_context(rows("wd_election_context.json"))
    events = by_id(wd.events_from_elections(rows("wd_elections.json"), ctx))
    assert set(events) == {"wd-election-Q115632204", "wd-election-Q107481430", "wd-election-Q56400471",
                           "wd-election-Q900001", "wd-election-Q900002"}
    for e in events.values():
        assert validate_event(e) == [] and e.topic == "elections" and e.category == "election"
        assert correct_text(e) in e.aliases

    brazil = events["wd-election-Q115632204"]
    assert brazil.date == "2026-10-25"  # the runoff, the latest dated round
    assert brazil.predictability == "high" and brazil.reject_aliases == []  # incumbent re-elected
    assert brazil.question_current == "Who won the most recent Brazilian presidential election?"
    assert "Tarcísio de Freitas" in choice_texts(brazil)  # a candidate of the same election

    georgia = events["wd-election-Q107481430"]
    assert {"Mike Collins", "Derek Dooley"} <= set(choice_texts(georgia)) and georgia.region == "US"

    finland = events["wd-election-Q56400471"]
    assert finland.reject_aliases == ["Sauli Niinistö", "Sauli Niinisto"]
    assert "Sauli Niinistö" in choice_texts(finland)
    assert len({"Pekka Haavisto", "Jussi Halla-aho", "Li Andersson"} & set(choice_texts(finland))) == 2
    assert finland.question_recall == "Who won the 2024 Finnish presidential election, held in February 2024?"

    assert events["wd-election-Q900001"].question_current == ""  # superseded by the 2026 election
    assert events["wd-election-Q900002"].reject_aliases == ["Anna Example"]


def test_events_from_elections_without_context():
    events = by_id(wd.events_from_elections(rows("wd_elections.json")))
    assert "wd-election-Q900003" in events  # no precision information -> the row date is used
    assert events["wd-election-Q900002"].reject_aliases == ["Anna Example"]  # previous election in the same series
    assert events["wd-election-Q115632204"].date == "2026-10-25"


def test_events_from_awards():
    events = by_id(wd.events_from_awards(rows("wd_awards.json"), since="2024-01-01"))
    assert set(events) == {"wd-award-Q38104-2024", "wd-award-Q38104-2025", "wd-award-Q103916-2024",
                           "wd-award-Q103916-2025", "wd-award-Q103916-2026", "wd-award-Q160082-2024",
                           "wd-award-Q160082-2025"}  # 2023 only as predecessors; Turing (year precision) dropped
    for e in events.values():
        assert validate_event(e) == [] and e.category == "award"
    physics24 = events["wd-award-Q38104-2024"]
    assert physics24.date == "2024-10-01" and physics24.topic == "science_research"  # Nobel: October
    assert physics24.aliases == ["Geoffrey Hinton", "John Hopfield"]
    assert physics24.question_current == ""
    assert physics24.mcq_question == "Which of the following won the Nobel Prize in Physics in October 2024?"
    assert set(physics24.reject_aliases) == {"Anne L'Huillier", "Pierre Agostini"}
    physics25 = events["wd-award-Q38104-2025"]
    assert physics25.date == "2025-10-07"  # the day-precision statement wins over year-only ones
    assert physics25.question_current == "Who won the most recent Nobel Prize in Physics?"
    assert physics25.mcq_question == "Which of the following won the most recent Nobel Prize in Physics?"
    assert correct_text(physics25) in {"John Clarke", "Michel Devoret", "John M. Martinis"}
    assert not {"John Clarke", "Michel Devoret", "John M. Martinis"} - {correct_text(physics25)} & set(
        choice_texts(physics25))
    actor = events["wd-award-Q103916-2026"]
    assert actor.aliases == ["Michael B. Jordan"] and actor.reject_aliases == ["Adrien Brody"]  # film dropped
    assert actor.topic == "arts_entertainment"
    assert events["wd-award-Q160082-2024"].date == "2024-11-01"  # Booker: November


def test_events_from_sports_with_context():
    ctx = wd.sports_context(rows("wd_sports_context.json"))
    events = by_id(wd.events_from_sports(rows("wd_sports.json"), ctx))
    assert set(events) == {"wd-sports-Q501", "wd-sports-Q502", "wd-sports-Q701", "wd-sports-Q702",
                           "wd-sports-Q703"}  # QID label and year-precision edition dropped
    for e in events.values():
        assert validate_event(e) == [] and e.topic == "sports" and e.category == "sports_result"
    gp24, gp25 = events["wd-sports-Q501"], events["wd-sports-Q502"]
    assert gp24.subject == "2024 Dutch Grand Prix" and gp24.aliases == ["Lando Norris"]  # the human winner
    assert gp24.question_current == ""
    assert gp25.question_current == "Who won the most recent edition of the Dutch Grand Prix?"
    assert set(gp25.reject_aliases) == {"Lando Norris", "McLaren"}
    assert "Lando Norris" in choice_texts(gp25) and "McLaren" not in choice_texts(gp25)  # same-kind stale option
    tdf = events["wd-sports-Q702"]
    assert tdf.predictability == "high" and tdf.reject_aliases == []  # repeat champion
    assert events["wd-sports-Q703"].question_recall == "Who won the 2026 Paris–Roubaix, which concluded in April 2026?"


def _fake_runner(queries: list[str]):
    data = {
        "heads": (rows("wd_heads.json"), "start"), "elections": (rows("wd_elections.json"), "d"),
        "awards": (rows("wd_awards.json"), "d"), "sports": (rows("wd_sports.json"), "date"),
        "election_context": (rows("wd_election_context.json"), "e"),
        "sports_context": (rows("wd_sports_context.json"), "ed"),
    }

    def which(q: str) -> str:
        for name, marker in (("election_context", "P726"), ("sports_context", "humanWinner"), ("heads", "P1906"),
                             ("elections", "wdt:P991 ?winner"), ("awards", "p:P166"), ("sports", "wdt:P3450"),
                             ("sports_series", "wdt:P179")):
            if marker in q:
                return name
        raise AssertionError(q)

    def runner(query: str, client: Any, retries: int) -> list[dict[str, Any]]:
        queries.append(query)
        name = which(query)
        if name == "sports_series":
            return []
        table, key = data[name]
        if name.endswith("context"):
            wanted = set(re.findall(r"wd:(Q\d+)", query))
            return [r for r in table if r[key] in wanted]
        start, end = re.findall(r'"(\d{4}-\d{2}-\d{2})T00:00:00Z"', query)[:2]
        return [r for r in table if start <= r[key][:10] < end]

    return runner


def test_build_wikidata_events_offline():
    queries: list[str] = []
    events = wd.build_wikidata_events("2024-06-01", "2026-11-24", runner=_fake_runner(queries))
    assert [(e.date, e.id) for e in events] == sorted((e.date, e.id) for e in events)
    assert len({e.id for e in events}) == len(events)
    assert all("2024-06-01" <= e.date <= "2026-11-24" and validate_event(e) == [] for e in events)
    cats = {e.category for e in events}
    assert cats == {"office_change", "election", "award", "sports_result"}
    ids = {e.id for e in events}
    assert "wd-office-Q1587677-2024-01-09-Q30339350" not in ids  # before start
    assert "wd-award-Q38104-2024" in ids
    award_starts = [re.findall(r'"(\d{4}-\d{2}-\d{2})T', q)[0] for q in queries if "p:P166" in q]
    assert award_starts[0] == "2023-06-01"  # one year earlier, for previous recipients
    assert len([q for q in queries if "P1906" in q]) == 3  # yearly windows


def test_build_wikidata_events_kinds_and_dates():
    queries: list[str] = []
    events = wd.build_wikidata_events("2024-01-01", "2026-11-24", kinds=["elections"], runner=_fake_runner(queries))
    assert {e.category for e in events} == {"election"}
    assert not any("P1906" in q for q in queries)
    with pytest.raises(ValueError, match="unknown kinds"):
        wd.build_wikidata_events("2024-01-01", "2026-11-24", kinds=["films"], runner=_fake_runner([]))
    with pytest.raises(ValueError, match="before start"):
        wd.build_wikidata_events("2026-01-01", "2025-01-01", runner=_fake_runner([]))
