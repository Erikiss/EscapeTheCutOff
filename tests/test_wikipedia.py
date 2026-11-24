"""Tests for cutoffmix.bench.sources.wikipedia (Current events portal and deaths lists; no network)."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from cutoffmix.bench import sources
from cutoffmix.bench.sources import SourceError
from cutoffmix.bench.sources import wikipedia as wp
from cutoffmix.bench.sources.wikidata import USER_AGENT
from cutoffmix.taxonomy import TOPIC_IDS

FIX = Path(__file__).parent / "fixtures"
DAY = dt.date(2026, 3, 3)


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    waits: list[float] = []
    monkeypatch.setattr(sources, "pause", waits.append)
    return waits


# --------------------------------------------------------------------------------------
# Current events parsing
# --------------------------------------------------------------------------------------


def test_parse_ce_fixture():
    items = wp.parse_ce_wikitext((FIX / "ce_2026_March_3.wikitext").read_text(encoding="utf-8"), DAY)
    assert len(items) == 14
    assert all(set(it) == {"date", "ce_category", "story_path", "text", "links", "sources"} for it in items)
    assert all(it["date"] == "2026-03-03" for it in items)
    cats = [it["ce_category"] for it in items]
    assert cats.count("Armed conflicts and attacks") == 5 and cats.count("Science and technology") == 2
    first = items[0]
    assert first["story_path"] == ["Middle Eastern crisis (2023–present)", "2026 Iran war",
                                   "2026 Iranian strikes on Arab countries",
                                   "2026 Iranian strikes on the United Arab Emirates"]
    assert first["text"].startswith("An Iranian drone strike damages Al Minhad Air Base")
    assert "[[" not in first["text"] and "http" not in first["text"]
    assert first["sources"][0]["outlet"] == "Reuters"
    assert "Richard Marles" in first["links"]
    tanker = next(it for it in items if "Arctic Metagaz" in it["text"])
    assert [s["outlet"] for s in tanker["sources"]] == ["Kyiv Independent", "The Moscow Times"]
    helicopter = next(it for it in items if "helicopter" in it["text"])
    assert helicopter["story_path"] == []  # top-level leaf


def test_parse_ce_variants():
    text = (
        "{{Current events|year=2024|month=01|day=1|top=yes}}\n"
        "<!-- All news items below this line -->'''Armed conflict and attacks'''\n"
        "*[[Story]]\n"
        "**[[Sub-story]]\n"
        "***Short leaf. [https://example.org/a (Reuters)]\n"
        "**Too short\n"
        ";Science and technologies\n"
        "*A rather long leaf sentence that has no source link but is longer than sixty characters.\n"
        "'''Music'''\n"
        "*The band releases an album. [https://example.org/b (''BBC News'')]\n"
        "'''Weather oddities'''\n"
        "*Something unclassified happens somewhere in the world today. [https://example.org/c (AP)]\n"
        "{{Current events|bottom=yes}}\n"
    )
    items = wp.parse_ce_wikitext(text, dt.date(2024, 1, 1))
    assert [(it["ce_category"], it["text"]) for it in items] == [
        ("Armed conflicts and attacks", "Short leaf."),
        ("Science and technology",
         "A rather long leaf sentence that has no source link but is longer than sixty characters."),
        ("Arts and culture", "The band releases an album."),
        (None, "Something unclassified happens somewhere in the world today."),
    ]
    assert items[0]["story_path"] == ["Story", "Sub-story"]
    assert items[2]["sources"] == [{"url": "https://example.org/b", "outlet": "BBC News"}]


def test_canonical_ce_category_and_plain_text():
    assert wp.canonical_ce_category("Law & crime") == "Law and crime"
    assert wp.canonical_ce_category("Disasters and accident") == "Disasters and accidents"
    assert wp.canonical_ce_category("Weather") is None
    assert wp.plain_text("[[Iran]]ian ''[[drone warfare|drone]]'' {{small|x}} [https://x.org (AP)]") == "Iranian drone"


def _item(category, text, **kw) -> dict[str, Any]:
    return {"date": "2026-03-03", "ce_category": category, "story_path": [], "text": text, "links": [],
            "sources": [], **kw}


@pytest.mark.parametrize("category,text,topic", [
    ("Business and economy", "OpenAI raises $40 billion in a funding round.", "ai_ml"),
    ("Science and technology", "SpaceX launches a crewed mission to the space station.", "space_astronomy"),
    ("Science and technology", "A ransomware attack hits the hospital software provider.", "computing_software"),
    ("Science and technology", "Archaeologists discover a Bronze Age tomb.", "science_research"),
    ("Health and environment", "Global carbon emissions reach a record high.", "natural_disasters_climate"),
    ("Health and environment", "A measles outbreak spreads in the region.", "health_medicine"),
    ("Disasters and accidents", "A magnitude 7.1 earthquake strikes off the coast.", "natural_disasters_climate"),
    ("Disasters and accidents", "A passenger train derails, killing 12.", "accidents"),
    ("Politics and elections", "Voters go to the polls in the presidential runoff.", "elections"),
    ("Politics and elections", "The prime minister reshuffles the cabinet.", "government"),
    ("Law and crime", "Hackers leak data from a data breach at the retailer.", "computing_software"),
    ("Law and crime", "The court sentences the former governor to 10 years.", "law_crime"),
    ("Armed conflicts and attacks", "A drone strike hits the base.", "conflict_security"),
    ("International relations", "The foreign ministers sign a treaty.", "international_relations"),
    ("Sports", "The club wins the league title.", "sports"),
    ("Arts and culture", "The film wins the top prize at the festival.", "arts_entertainment"),
])
def test_ce_item_topic(category, text, topic):
    assert wp.ce_item_topic(_item(category, text)) == topic


def test_ce_item_topic_uses_story_path_and_unknown_heading():
    assert wp.ce_item_topic(_item("Science and technology", "The probe reaches its target.",
                                  story_path=["Artemis program"])) == "space_astronomy"
    assert wp.ce_item_topic(_item(None, "Something happens.")) in TOPIC_IDS
    items = wp.parse_ce_wikitext((FIX / "ce_2026_March_3.wikitext").read_text(encoding="utf-8"), DAY)
    topics = [wp.ce_item_topic(it) for it in items]
    assert topics.count("conflict_security") == 5 and topics[-2:] == ["space_astronomy", "space_astronomy"]
    assert "accidents" in topics and "government" in topics


# --------------------------------------------------------------------------------------
# Deaths lists
# --------------------------------------------------------------------------------------


def test_parse_deaths_month_page():
    entries = wp.parse_deaths((FIX / "deaths_october_2026.wikitext").read_text(encoding="utf-8"), 2026, 10)
    assert len(entries) == 8
    anderson = next(e for e in entries if e["title"] == "J. Todd Anderson")
    assert anderson == {
        "date": "2026-10-01", "title": "J. Todd Anderson", "name": "J. Todd Anderson", "age": 67,
        "desc": "American storyboard artist (The Big Lebowski, Fargo, The Nice Guys)", "cause": "cancer",
        "ref_urls": ["https://www.hollywoodreporter.com/movies/movie-news/"
                     "j-todd-anderson-dead-storyboard-artist-1236721947/"],
    }
    baird = next(e for e in entries if e["title"] == "Bruce Baird")
    assert baird["date"] == "2026-10-02" and baird["cause"] is None
    assert baird["desc"] == "Australian politician, MP (1998–2007) and New South Wales MLA (1984–1995)"
    assert next(e for e in entries if e["title"] == "Duke Carlisle")["cause"] == "lung cancer"


def test_parse_deaths_year_page_reads_only_the_month():
    entries = wp.parse_deaths((FIX / "deaths_2026_year.wikitext").read_text(encoding="utf-8"), 2026, 11)
    assert [e["title"] for e in entries] == ["Irinej Bulović", "Hikmet Çetin", "Fábio Konder Comparato",
                                             "Delfina Guzmán", "Orazak Ismagulov"]
    assert all(e["date"] == "2026-11-24" for e in entries)
    guzman = entries[3]
    assert guzman["desc"] == "Chilean actress (Three Sad Tigers, The Expropriation, La Nana)"  # date note dropped
    assert entries[2]["desc"] == "Brazilian lawyer, jurist and writer" and entries[2]["cause"] is None
    assert wp.parse_deaths("==December==\n===1===\n*[[A B]], 70, X.\n", 2026, 11) == []


def test_parse_deaths_piped_link_and_missing_age():
    text = "===5===\n*[[Sergey Lebedev (footballer)|Sergey Lebedev]], 57, Uzbek footballer, heart attack.\n" \
           "*[[Someone]], Peruvian singer.\n"
    first, second = wp.parse_deaths(text, 2026, 11)
    assert first["title"] == "Sergey Lebedev (footballer)" and first["name"] == "Sergey Lebedev"
    assert first["cause"] == "heart attack" and second["age"] is None and second["desc"] == "Peruvian singer"


# --------------------------------------------------------------------------------------
# Fetching (MockTransport)
# --------------------------------------------------------------------------------------


def _api(handler):
    return httpx.Client(transport=httpx.MockTransport(handler), headers={"User-Agent": USER_AGENT})


def _pages_response(titles: list[str], missing: set[str] = frozenset(), **query: Any) -> httpx.Response:
    pages = []
    for t in titles:
        if t in missing:
            pages.append({"ns": 100, "title": t, "missing": True})
        else:
            pages.append({"pageid": 1, "ns": 100, "title": t, "revisions": [
                {"revid": 7, "timestamp": "2026-11-01T00:00:00Z",
                 "slots": {"main": {"contentmodel": "wikitext", "content": f"text of {t}"}}}]})
    return httpx.Response(200, json={"batchcomplete": True, "query": {"pages": pages, **query}})


def test_fetch_ce_days_batches_of_50(no_waiting):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        titles = request.url.params["titles"].split("|")
        return _pages_response(titles, missing={"Portal:Current events/2026 February 1"})

    start, end = dt.date(2026, 1, 1), dt.date(2026, 2, 25)
    with _api(handler) as client:
        days = wp.fetch_ce_days(start, end, client=client)
    assert len(requests) == 2
    assert [len(r.url.params["titles"].split("|")) for r in requests] == [50, 6]
    for r in requests:
        assert r.url.params["maxlag"] == "5" and r.url.params["formatversion"] == "2"
        assert r.url.params["prop"] == "revisions" and r.headers["User-Agent"] == USER_AGENT
    assert len(days) == 55 and dt.date(2026, 2, 1) not in days
    assert days[dt.date(2026, 1, 1)] == "text of Portal:Current events/2026 January 1"
    assert no_waiting == [1.0]  # pause between batches
    with pytest.raises(ValueError):
        wp.fetch_ce_days(end, start)


def test_fetch_pages_retries_maxlag_and_429(no_waiting):
    replies = iter([
        httpx.Response(429, headers={"Retry-After": "4"}),
        httpx.Response(200, headers={"Retry-After": "6"},
                       json={"error": {"code": "maxlag", "info": "Waiting for a database server"}}),
        _pages_response(["Portal:Current events/2026 March 3"]),
    ])
    with _api(lambda r: next(replies)) as client:
        days = wp.fetch_ce_days(DAY, DAY, client=client)
    assert days == {DAY: "text of Portal:Current events/2026 March 3"}
    assert no_waiting == [4.0, 6.0]


def test_api_error_raises():
    with _api(lambda r: httpx.Response(200, json={"error": {"code": "badvalue", "info": "nope"}})) as client:
        with pytest.raises(SourceError, match="badvalue"):
            wp.fetch_ce_days(DAY, DAY, client=client)


def test_fetch_deaths_month_follows_redirect():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["redirects"] == "1"
        assert request.url.params["titles"] == "Deaths in November 2026"
        return _pages_response(
            ["Deaths in 2026"],
            redirects=[{"from": "Deaths in November 2026", "to": "Deaths in 2026", "tofragment": "November"}])

    with _api(handler) as client:
        assert wp.fetch_deaths_month(2026, 11, client=client) == "text of Deaths in 2026"
        pages = wp.fetch_pages(["Deaths in November 2026"], client=client, redirects=True)
    assert pages["Deaths in November 2026"]["title"] == "Deaths in 2026"
    assert pages["Deaths in November 2026"]["fragment"] == "November"


def test_fetch_deaths_month_missing_page_and_normalised_titles():
    def handler(request: httpx.Request) -> httpx.Response:
        return _pages_response(["Deaths in March 2031"], missing={"Deaths in March 2031"},
                               normalized=[{"from": "deaths in March 2031", "to": "Deaths in March 2031"}])

    with _api(handler) as client:
        with pytest.raises(SourceError, match="not found"):
            wp.fetch_deaths_month(2031, 3, client=client)
    assert wp.deaths_title(2026, 10) == "Deaths in October 2026"
    assert wp.ce_title(DAY) == "Portal:Current events/2026 March 3"


def test_fetch_pages_maps_normalised_titles():
    def handler(request: httpx.Request) -> httpx.Response:
        return _pages_response(["Deaths in October 2026"],
                               normalized=[{"from": "Deaths_in_October_2026", "to": "Deaths in October 2026"}])

    with _api(handler) as client:
        pages = wp.fetch_pages(["Deaths_in_October_2026"], client=client)
    assert json.dumps(pages["Deaths_in_October_2026"]["content"]) == '"text of Deaths in October 2026"'
