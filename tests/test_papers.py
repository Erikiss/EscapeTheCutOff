"""Tests for cutoffmix.bench.sources.papers (Hugging Face daily papers + arXiv API) and the paper grading rules.

No network: all traffic goes through ``httpx.MockTransport``. Fixtures under ``tests/fixtures``:

* ``hf_daily_2026-10-15.json``: the Hugging Face daily-papers list of 2026-10-15, cut to its first ten items and to
  the fields the source reads (``paper.id/title/upvotes/authors/publishedAt/summary``); public listing metadata.
* ``arxiv_batch_2026-10.xml``: the arXiv API Atom answer for those ten ids (arXiv metadata, arXiv API terms of use).

A second month (2026-11) is synthesised in this module so that month filtering, author filters, interp share,
distractors across months and the spliced-title controls can be exercised.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import random
import xml.sax.saxutils as sax
from pathlib import Path
from typing import Any

import httpx
import pytest

from cutoffmix.bench import sources
from cutoffmix.bench.grading import grade_paper, grade_response, grade_rule
from cutoffmix.bench.io import load_events, read_jsonl
from cutoffmix.bench.probegen import validate_event
from cutoffmix.bench.sources import SourceError
from cutoffmix.bench.sources import papers as pp
from cutoffmix.cli import main
from cutoffmix.types import Event

FIX = Path(__file__).parent / "fixtures"
HF_DAY = FIX / "hf_daily_2026-10-15.json"
ARXIV_XML = FIX / "arxiv_batch_2026-10.xml"
TODAY = dt.date(2026, 11, 30)


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    waits: list[float] = []
    monkeypatch.setattr(sources, "pause", waits.append)
    pp._LAST_REQUEST.clear()
    return waits


# --------------------------------------------------------------------------------------
# synthetic second month and the fake web
# --------------------------------------------------------------------------------------

# (id, title, authors, published, category); 2611.00006..8 are the edge cases described in the tests
NOVEMBER: list[tuple[str, str, list[str], str, str]] = [
    ("2611.00001", "Sparse Autoencoders Reveal Monosemantic Features in Vision Transformers",
     ["Anna Kowalski", "Ben Oyelaran", "Chiara Rossi"], "2026-11-03", "cs.LG"),
    ("2611.00002", "Circuit Discovery with Attribution Patching at Scale",
     ["Dmitri Volkov", "Eva Lindqvist", "Farid Haddad", "Gao Wen"], "2026-11-04", "cs.LG"),
    ("2611.00003", "Steering Language Models through Activation Feature Directions",
     ["Hiroshi Tanaka", "Ines Moreau", "Jonas Berg"], "2026-11-05", "cs.CL"),
    ("2611.00004", "Efficient Long-Context Training via Blockwise Attention Scheduling",
     ["Kavya Raman", "Liam O'Connor", "Marta Silva"], "2026-11-02", "cs.LG"),
    ("2611.00005", "Scaling Laws for Agentic Reinforcement Learning in Simulated Worlds",
     ["Noah Fischer", "Olga Petrova", "Pedro Alves"], "2026-11-04", "cs.AI"),
    ("2611.00006", "A Reality Check on Tool Use Benchmarks for Coding Agents",
     ["DeepSeek-AI", "Someone Else", "Third Person"], "2026-11-04", "cs.SE"),  # one-token first author: skipped
    ("2611.00007", "Benchmarking Multimodal Agents in 2026 and Beyond for Everyday Tasks",
     ["Quinn Baker", "Rosa Diaz", "Sam Lee"], "2026-11-04", "cs.AI"),  # year in the title: skipped
    ("2611.00008", "Late Listing: Memory Consolidation for Streaming Video Agents",
     ["Tomas Novak", "Uma Iyer", "Viktor Hansen"], "2026-10-29", "cs.CV"),  # listed in Nov, published in Oct
    ("2611.00009", "Robust Preference Optimization under Noisy Human Feedback Signals",
     ["Wei Chen", "Xin Zhao", "Yara Haddad"], "2026-11-05", "cs.LG"),
    ("2611.00010", "Distilling Reasoning Traces into Compact Student Models Efficiently",
     ["Zoe Martin", "Adam Weiss", "Bella Nkemelu"], "2026-11-05", "cs.CL"),
]
UNKNOWN_TO_ARXIV = "2611.00011"


def november_hf_items() -> list[Any]:
    items: list[Any] = ["junk", {"title": "no paper key"}, {"paper": {"id": "not-an-id", "title": "x"}}]
    for i, (pid, title, authors, published, _cat) in enumerate(NOVEMBER):
        items.append({"title": title, "publishedAt": f"{published}T20:00:00.000Z",
                      "paper": {"id": pid, "title": title, "upvotes": 90 - 5 * i,
                                "authors": [{"name": a, "hidden": False} for a in authors],
                                "publishedAt": f"{published}T00:00:00.000Z", "summary": "Synthetic summary."}})
    items.append({"paper": {"id": UNKNOWN_TO_ARXIV, "title": "Not on arXiv at all", "upvotes": 99, "authors": []}})
    for pid, upvotes in ((RETITLED, 97), (STABLE_V2, 95)):  # both at v2 on arXiv (see FakeWeb)
        items.append({"paper": {"id": pid, "title": "x", "upvotes": upvotes, "authors": [],
                                "publishedAt": "2026-11-06T00:00:00.000Z", "summary": "s"}})
    return items


# at v2 on arXiv: the first was retitled (and its author order changed) after v1, the second only fixed its abstract
RETITLED, STABLE_V2 = "2611.00012", "2611.00013"


def atom_feed(records: dict[str, pp.ArxivPaper]) -> bytes:
    entries = []
    for p in records.values():
        authors = "".join(f"<author><name>{sax.escape(a)}</name></author>" for a in p.authors)
        entries.append(
            f"<entry><id>http://arxiv.org/abs/{p.id}v{p.version or 1}</id><title>{sax.escape(p.title)}</title>"
            f"<summary>{sax.escape(p.summary)}</summary><published>{p.published}T10:00:00Z</published>"
            f'<arxiv:primary_category term="{p.primary_category}"/>{authors}</entry>')
    return ("<?xml version='1.0' encoding='UTF-8'?>"
            '<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">'
            "<entry><id>http://arxiv.org/api/errors#bad</id><title>Error</title></entry>"
            + "".join(entries) + "</feed>").encode("utf-8")


class FakeWeb:
    """Hugging Face lists per day and arXiv records; serves them through an httpx.MockTransport."""

    def __init__(self) -> None:
        self.hf: dict[str, bytes] = {"2026-10-15": HF_DAY.read_bytes(),
                                     "2026-11-05": json.dumps(november_hf_items()).encode("utf-8")}
        # latest versions (what a bare id returns); the fixture entries are all v1
        self.arxiv: dict[str, pp.ArxivPaper] = dict(pp.parse_arxiv_atom(ARXIV_XML.read_bytes()))
        for pid, title, authors, published, cat in NOVEMBER:
            self.arxiv[pid] = pp.ArxivPaper(pid, title, tuple(authors), published, cat, "Synthetic summary.", 1)
        self.arxiv[RETITLED] = pp.ArxivPaper(
            RETITLED, "MemSurvey: A Comprehensive Survey of Agentic Memory Systems for Long-Horizon Tasks",
            ("Yara Haddad", "Omar Said", "Lin Qiao"), "2026-11-06", "cs.AI", "Synthetic summary.", 2)
        self.arxiv[STABLE_V2] = pp.ArxivPaper(
            STABLE_V2, "Stable Title Across Versions for Agent Evaluation Suites",
            ("Carla Mendes", "Dev Patel", "Eli Katz"), "2026-11-07", "cs.AI", "Revised summary.", 2)
        # version-1 records that differ from the latest (``<id>v1`` requests); others are served as the latest at v1
        self.arxiv_v1: dict[str, pp.ArxivPaper] = {
            RETITLED: pp.ArxivPaper(RETITLED, "Agentic Memory Systems: A Short Survey",
                                    ("Omar Said", "Yara Haddad", "Lin Qiao"), "2026-11-06", "cs.AI", "Old summary.", 1),
            STABLE_V2: dataclasses.replace(self.arxiv[STABLE_V2], summary="First summary.", version=1),
        }
        self.calls: list[httpx.Request] = []
        self.fail: dict[str, int] = {}  # date -> status code to answer with

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        if request.url.host == "huggingface.co":
            day = request.url.params["date"]
            if day in self.fail:
                return httpx.Response(self.fail[day], text="nope")
            return httpx.Response(200, content=self.hf.get(day, b"[]"))
        assert request.url.host == "export.arxiv.org"
        ids = request.url.params["id_list"].split(",")
        assert len(ids) <= pp.ARXIV_BATCH and request.url.params["max_results"] == str(len(ids))
        records: dict[str, pp.ArxivPaper] = {}
        for raw in ids:
            pid = pp.normalize_arxiv_id(raw)
            if pid not in self.arxiv:
                continue
            if raw.endswith("v1"):
                records[pid] = self.arxiv_v1.get(pid, dataclasses.replace(self.arxiv[pid], version=1))
            else:
                records[pid] = self.arxiv[pid]
        return httpx.Response(200, content=atom_feed(records))

    def arxiv_requests(self, *, v1: bool) -> list[list[str]]:
        """The id lists of the arXiv requests that asked for version 1 (``v1=True``) or for the bare ids."""
        out = []
        for r in self.requests("export.arxiv.org"):
            ids = r.url.params["id_list"].split(",")
            if all(i.endswith("v1") for i in ids) == v1:
                out.append(ids)
        return out

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler))

    def requests(self, host: str) -> list[httpx.Request]:
        return [r for r in self.calls if r.url.host == host]


@pytest.fixture
def web() -> FakeWeb:
    return FakeWeb()


def build(web: FakeWeb, cache: Path, **kw: Any):
    args: dict[str, Any] = {"per_month": 6, "anchor_from": None, "anchor_to": None, "controls_share": 0.25,
                            "candidates_per_month": 40, "cache_dir": cache, "today": TODAY, **kw}
    with web.client() as client:
        return pp.build_paper_events("2026-10", "2026-11", client=client, **args)


# --------------------------------------------------------------------------------------
# parsers and fetchers
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("value, expected", [
    ("2610.20399", "2610.20399"), ("2610.20399v2", "2610.20399"), (" arXiv:2610.20399 ", "2610.20399"),
    ("http://arxiv.org/abs/2610.20399v1", "2610.20399"), ("https://arxiv.org/pdf/2301.00001.pdf", "2301.00001"),
    ("cs/0101001", "cs/0101001"), ("math.GT/0309136v1", "math.GT/0309136"),
    ("http://arxiv.org/api/errors#incorrect_id", None), ("not an id", None), (12345, None), (None, None),
])
def test_normalize_arxiv_id(value, expected):
    assert pp.normalize_arxiv_id(value) == expected


def test_hf_paper_is_defensive():
    assert pp.hf_paper("junk", "2026-10-15") is None
    assert pp.hf_paper({"title": "x"}, "2026-10-15") is None
    assert pp.hf_paper({"paper": {"id": "bad", "title": "x"}}, "2026-10-15") is None
    assert pp.hf_paper({"paper": {"id": "2610.1", "title": ""}}, "2026-10-15") is None
    p = pp.hf_paper({"paper": {"id": "2610.20399v1", "title": "  A\n title ", "upvotes": True,
                               "authors": [{"name": " Ann  Lee "}, "Bob Ray", 7, {"x": 1}], "publishedAt": "garbage",
                               "summary": None}, "summary": "fallback"}, "2026-10-15")
    assert p == pp.HFPaper("2610.20399", "A title", 0, ("Ann Lee", "Bob Ray"), "", "fallback", "2026-10-15")
    real = pp.hf_paper(json.loads(HF_DAY.read_text(encoding="utf-8"))[0], "2026-10-15")
    assert real is not None and real.id == "2610.20399" and real.upvotes == 70 and real.published_at == "2026-10-14"
    assert real.authors[0] == "Jintian Zhang"


def test_fetch_hf_daily_caches_paces_and_refetches_corrupt_cache(web, tmp_path, no_waiting):
    with web.client() as client:
        first = pp.fetch_hf_daily("2026-10-15", client=client, cache_dir=tmp_path)
        again = pp.fetch_hf_daily(dt.date(2026, 10, 15), client=client, cache_dir=tmp_path)
        empty = pp.fetch_hf_daily("2026-10-18", client=client, cache_dir=tmp_path)
    assert len(first) == 10 and again == first and empty == []
    assert len(web.calls) == 2 and web.calls[0].headers["User-Agent"] == pp.USER_AGENT
    assert str(web.calls[0].url).startswith(pp.HF_DAILY_URL + "?date=2026-10-15")
    assert (tmp_path / "hf" / "2026-10-15.json").read_bytes() == HF_DAY.read_bytes()
    assert (tmp_path / "hf" / "2026-10-18.json").read_text() == "[]"
    assert no_waiting and all(0 < w <= pp.HF_PACE_S for w in no_waiting)  # paced after the first request
    (tmp_path / "hf" / "2026-10-15.json").write_text("{not json")
    with web.client() as client:
        assert len(pp.fetch_hf_daily("2026-10-15", client=client, cache_dir=tmp_path)) == 10
    assert len(web.calls) == 3


def test_fetch_hf_daily_rejects_non_list_and_bad_json(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text='{"a": 1}' if "01-01" in request.url.params["date"] else "nope")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(SourceError, match="JSON list"):
            pp.fetch_hf_daily("2026-01-01", client=client, cache_dir=tmp_path)
        with pytest.raises(SourceError, match="invalid JSON"):
            pp.fetch_hf_daily("2026-01-02", client=client, cache_dir=tmp_path)
    assert not (tmp_path / "hf").exists() or not list((tmp_path / "hf").iterdir())


def test_collect_month_dedupes_by_id_keeps_max_upvotes_and_first_day(web, tmp_path):
    day1 = json.loads(HF_DAY.read_text(encoding="utf-8"))
    day2 = json.loads(json.dumps(day1[:3]))
    day2[0]["paper"]["upvotes"] = 500  # 2610.20399 again, more upvotes
    day2[1]["paper"]["upvotes"] = 1  # 2610.20571 again, fewer
    day2[2]["paper"]["id"] = "2610.99999"  # a new one
    day2[2]["paper"]["upvotes"] = 65
    web.hf["2026-10-16"] = json.dumps(day2).encode("utf-8")
    with web.client() as client:
        top = pp.collect_month(2026, 10, 5, client=client, cache_dir=tmp_path, today=dt.date(2026, 10, 20))
    assert [p.id for p in top] == ["2610.20399", "2610.99999", "2610.20571", "2610.13521", "2610.20584"]
    assert top[0].upvotes == 500 and top[0].hf_date == "2026-10-15" and top[2].upvotes == 61
    assert top[1].hf_date == "2026-10-16"
    assert len(web.requests("huggingface.co")) == 20  # days 1..20 only (today), none after
    assert {r.url.params["date"] for r in web.calls} == {f"2026-10-{d:02d}" for d in range(1, 21)}


def test_collect_month_skips_days_that_fail_for_good(web, tmp_path):
    web.fail["2026-10-03"] = 404
    web.fail["2026-10-04"] = 503
    errors: list[str] = []
    with web.client() as client:
        top = pp.collect_month(2026, 10, 3, client=client, cache_dir=tmp_path, today=dt.date(2026, 10, 15),
                               retries=1, errors=errors)
    assert [p.id for p in top] == ["2610.20399", "2610.20571", "2610.13521"]
    assert len(errors) == 2 and errors[0].startswith("2026-10-03") and "503" in errors[1]
    with pytest.raises(ValueError):
        pp.collect_month(2026, 10, 0)


def test_parse_arxiv_atom_fixture():
    papers = pp.parse_arxiv_atom(ARXIV_XML.read_bytes())
    assert len(papers) == 10
    cortex = papers["2610.20399"]
    assert cortex.title == "Cortex: Consolidating Context into Parametric Memory"
    assert cortex.authors[:2] == ("Jintian Zhang", "Feng He") and len(cortex.authors) == 8
    assert cortex.published == "2026-10-14" and cortex.primary_category == "cs.CL" and cortex.version == 1
    assert cortex.summary.startswith("Context can improve") and "\n" not in cortex.summary
    guo = papers["2610.19366"]
    assert guo.authors[0] == "Minghao Guo" and len(guo.authors) == 80 and "  " not in guo.title
    assert papers["2610.13521"].published == "2026-10-09"
    for p in papers.values():
        assert pp.normalize_arxiv_id(p.id) == p.id and p.title == " ".join(p.title.split())


def test_parse_arxiv_atom_skips_bad_entries_and_rejects_malformed_xml():
    good = pp.ArxivPaper("2611.00001", "T  itle\nwith   spaces", ("Ann Lee",), "2026-11-03", "", "s")
    feed = atom_feed({good.id: good}).replace(b"<published>2026-11-03T10:00:00Z</published>",
                                               b"<published>2026-11-03T10:00:00Z</published>"
                                               b"<category term=\"cs.LG\" scheme=\"x\"/>")
    parsed = pp.parse_arxiv_atom(feed)
    assert list(parsed) == ["2611.00001"] and parsed["2611.00001"].title == "T itle with spaces"
    assert parsed["2611.00001"].primary_category == "cs.LG"  # fallback to the first category term
    no_authors = feed.replace(b"<author><name>Ann Lee</name></author>", b"")
    assert pp.parse_arxiv_atom(no_authors) == {}
    with pytest.raises(SourceError, match="malformed"):
        pp.parse_arxiv_atom(b"<feed><entry>")


def test_fetch_arxiv_batches_paces_and_caches_per_id(web, tmp_path, no_waiting):
    many = [f"2611.{n:05d}" for n in range(1, 11)] + [f"2609.{n:05d}" for n in range(1, 141)]
    for pid in many[10:]:
        web.arxiv[pid] = pp.ArxivPaper(pid, f"Paper {pid} title", ("Some Body", "Other One"), "2026-09-05", "cs.LG", "")
    with web.client() as client:
        got = pp.fetch_arxiv(many + ["junk", UNKNOWN_TO_ARXIV, "2611.00001v2"], client=client, cache_dir=tmp_path)
    reqs = web.requests("export.arxiv.org")
    assert len(reqs) == 2 and [len(r.url.params["id_list"].split(",")) for r in reqs] == [100, 51]
    assert reqs[0].headers["User-Agent"] == pp.USER_AGENT
    assert len(got) == 150 and UNKNOWN_TO_ARXIV not in got and got["2611.00001"].authors[0] == "Anna Kowalski"
    assert sum(1 for w in no_waiting if w > 1.0) == 1 and max(no_waiting) <= pp.ARXIV_PACE_S  # 3 s before batch 2
    assert len(list((tmp_path / "arxiv" / "raw").glob("*.xml"))) == 2
    assert json.loads((tmp_path / "arxiv" / "2611.00001.json").read_text())["published"] == "2026-11-03"
    web.calls.clear()
    with web.client() as client:
        again = pp.fetch_arxiv(["2611.00001", "2611.00002"], client=client, cache_dir=tmp_path)
    assert again == {k: got[k] for k in ("2611.00001", "2611.00002")} and web.calls == []
    with pytest.raises(ValueError):
        pp.fetch_arxiv(["2611.00001"], batch=0)


def test_fetch_arxiv_version_1_asks_for_v1_ids_and_caches_them_apart(web, tmp_path):
    with web.client() as client:
        latest = pp.fetch_arxiv([RETITLED, STABLE_V2], client=client, cache_dir=tmp_path)
        first = pp.fetch_arxiv([RETITLED, STABLE_V2, "2611.00001"], client=client, cache_dir=tmp_path, version=1)
    assert web.arxiv_requests(v1=False) == [[RETITLED, STABLE_V2]]
    assert web.arxiv_requests(v1=True) == [["2611.00001v1", f"{RETITLED}v1", f"{STABLE_V2}v1"]]  # sorted ids
    assert latest[RETITLED].version == 2 and latest[RETITLED].title.startswith("MemSurvey")
    assert first[RETITLED].version == 1 and first[RETITLED].title == "Agentic Memory Systems: A Short Survey"
    assert first[RETITLED].authors[0] == "Omar Said" and latest[RETITLED].authors[0] == "Yara Haddad"
    assert first[STABLE_V2].title == latest[STABLE_V2].title and first["2611.00001"].version == 1
    assert (tmp_path / "arxiv" / f"{RETITLED}.json").is_file() and (tmp_path / "arxiv" / f"{RETITLED}v1.json").is_file()
    assert json.loads((tmp_path / "arxiv" / f"{RETITLED}v1.json").read_text())["version"] == 1
    web.calls.clear()
    with web.client() as client:  # both caches are warm and independent
        assert pp.fetch_arxiv([RETITLED], client=client, cache_dir=tmp_path, version=1)[RETITLED] == first[RETITLED]
        assert pp.fetch_arxiv([RETITLED], client=client, cache_dir=tmp_path)[RETITLED] == latest[RETITLED]
    assert web.calls == []
    with pytest.raises(ValueError, match="version"):
        pp.fetch_arxiv([RETITLED], version=0)


def test_fetch_arxiv_skips_an_answer_for_another_version(tmp_path, caplog):
    def handler(request: httpx.Request) -> httpx.Response:  # the API ignores the suffix and answers with v3
        p = pp.ArxivPaper("2611.00001", "T itle words", ("Ann Lee",), "2026-11-03", "cs.LG", "s", 3)
        return httpx.Response(200, content=atom_feed({p.id: p}))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        got = pp.fetch_arxiv(["2611.00001"], client=client, cache_dir=tmp_path, version=1)
    assert got == {} and not (tmp_path / "arxiv" / "2611.00001v1.json").exists()
    assert "answered with version 3" in caplog.text


# --------------------------------------------------------------------------------------
# tagging, names, planning
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("title, summary, expected", [
    ("Sparse Autoencoders Find Highly Interpretable Features", "", True),
    ("Scaling Monosemanticity", "dictionary learning", True),
    ("Towards Transcoders and Crosscoders", "", True),
    ("Cortex: Consolidating Context into Parametric Memory", "", False),
    ("Understanding Grokking", "We use SAEs (sparse autoencoders) to look inside.", True),
    ("A Survey of Data Agents", "We map agents to SAE levels of autonomy.", False),  # the automotive SAE
    ("Agentless: Demystifying Software Engineering Agents", "This also improves interpretability.", False),
    ("Agentless: Demystifying Software Engineering Agents",
     "Interpretability matters; our interpretability analysis improves interpretability.", True),  # three times
    ("Are VLMs Ready for Autonomous Driving?", "interpretability ... interpretability", False),  # only twice
    ("Vision language models are blind: Failing to translate detailed visual features into words",
     "we use probing to show", False),  # the specific weak family is not in the title
    ("Probing the 3D Awareness of Visual Foundation Models", "features of the representations", True),
    ("LLM Explainability with Counterfactual Chains and Causal Graphs", "", False),  # XAI application, not interp
    ("Concerto: Joint 2D-3D Self-Supervised Learning", "linear probing on ImageNet", False),  # SSL evaluation jargon
    ("Efficient Pre-Training with Token Superposition", "", False),  # no feature/neuron/linear context
    ("Your Transformer Can Hold Two Thoughts at Once: Evidence of Linear Superposition in LLMs", "", True),
    ("HERMES: KV Cache as Hierarchical Memory", "a mechanistic view of streaming", False),
    ("Padding Tone: A Mechanistic Analysis of Padding Tokens in T2I Models", "", True),
    ("Circuit Discovery with Attribution Patching at Scale", "", True),  # two weak families, one in the title
    ("Probing the Limits of Long-Context Models", "", False),  # one weak family only
    ("Feature Pyramid Networks for Detection", "activation maps", False),  # "feature" alone in the title, one hit
    ("Steering Language Models through Activation Feature Directions", "", True),
    ("A Benchmark for Agents", "activation functions and feature extraction", False),  # no specific family at all
    ("Feature Attribution Benchmarks", "we probe circuits", True),  # specific family (attribution) in the title
    ("Mechanistic Analysis of Induction Heads", "", True),
    ("Causal Tracing of Factual Recall", "", True),
    ("Representation Engineering: A Top-Down Approach", "", True),
    # applied domains in the title: only a specific term in the title tags the paper
    ("CoD, Towards an Interpretable Medical Agent using Chain of Diagnosis",
     "interpretability interpretability interpretability interpretability", False),
    ("Interpretable MEG Decoding of Perceived Speech: Cortical Sources and Stimulus Features",
     "neurons and features of the cortex", False),
    ("BrainExplore: Large-Scale Discovery of Interpretable Visual Representations in the Human Brain", "", False),
    ("ROCKET: Calibration-guided Knapsack Enhanced Truncation for Efficient Model Compression",
     "we apply dictionary learning to the weights", False),
    ("Are VLMs Ready for Autonomous Driving? An Empirical Study", "interpretability " * 4, False),
    ("Sparse Autoencoders for Brain Decoding", "", True),  # a specific term in the title still wins
    ("Megatron Scaling of Mixture Models", "we use activation patching to show", True),  # "meg" is a whole word only
])
def test_is_interp(title, summary, expected):
    assert pp.is_interp(title, summary) is expected


@pytest.mark.parametrize("name, ok", [
    ("Jintian Zhang", True), ("Shi-Yu Tian", True), ("Jan van der Berg", True), ("Liam O'Connor", True),
    ("Gabriela Ben Melech Stan", True), ("Yolo Y. Tang", True), ("José Cambronero", True), ("Mike Zheng Shou", True),
    ("DeepSeek-AI", False), ("Kimi Team", False), ("Gemini Team", False), ("Qwen", False), ("X 123", False),
    ("Anonymous Authors", False), ("Et Al", False), ("The Collaboration", False),
    # organisations that slipped through the first filter
    ("Tencent Robotics X", False), ("Grid Robotics", False), ("Bytedance Seed", False), ("Apodex Team", False),
    ("Xu AI", False), ("Moonshot Research", False),
    # initials-only given names cannot be rule-graded against a full name; a surname needs two letters
    ("B. An", False), ("B. Y. Yan", False), ("I. Apanasevich", False), ("J.-P. Dupont", False), ("Y Chen", False),
    ("Jian X", False), ("Jian AI", False), ("Wei LLC", False),
])
def test_plausible_person(name, ok):
    assert pp.plausible_person(name) is ok


def test_split_name_surname_key_and_aliases():
    assert pp.split_name("Jintian Zhang") == ("Jintian", "Zhang")
    assert pp.split_name("Jan van der Berg") == ("Jan", "van der Berg")
    assert pp.split_name("Mike Zheng Shou") == ("Mike Zheng", "Shou")
    assert pp.surname_key("Jan van der Berg") == "van der berg" and pp.surname_key("José Müller") == "muller"
    aliases = pp.author_aliases("José Müller")
    assert aliases[:5] == ["José Müller", "Müller, José", "J. Müller", "Müller J.", "Müller José"]
    assert "Jose Muller" in aliases and "Muller, Jose" in aliases and len(aliases) == len({a.lower() for a in aliases})
    assert pp.author_aliases("Shi-Yu Tian") == ["Shi-Yu Tian", "Tian, Shi-Yu", "S. Tian", "Tian S.", "Tian Shi-Yu"]
    assert pp.author_aliases("Mike Zheng Shou")[-1] == "Mike Shou" and "Shou, Mike Zheng" in pp.author_aliases("Mike Zheng Shou")


def test_last_available_day_is_yesterday_utc():
    now = dt.datetime(2026, 11, 25, 1, 27, tzinfo=dt.timezone.utc)
    assert pp.last_available_day(now) == dt.date(2026, 11, 24)
    assert pp.last_available_day(now.astimezone(dt.timezone(dt.timedelta(hours=9)))) == dt.date(2026, 11, 24)
    assert pp.last_available_day() == dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=1)


def test_plan_months_dense_and_anchors():
    plans = pp.plan_months("2025-10", "2026-01", 12, anchor_from="2024-01", anchor_to="2025-09", anchor_every=3,
                           anchor_per_month=6)
    assert [(p.month, p.count, p.dense) for p in plans] == [
        ("2024-01", 6, False), ("2024-04", 6, False), ("2024-07", 6, False), ("2024-10", 6, False),
        ("2025-01", 6, False), ("2025-04", 6, False), ("2025-07", 6, False),
        ("2025-10", 12, True), ("2025-11", 12, True), ("2025-12", 12, True), ("2026-01", 12, True)]
    overlap = pp.plan_months("2025-01", "2025-03", 4, anchor_from="2024-11", anchor_to="2025-03", anchor_every=2,
                             anchor_per_month=2)
    assert [(p.month, p.count) for p in overlap] == [("2024-11", 2), ("2025-01", 4), ("2025-02", 4), ("2025-03", 4)]
    assert [p.month for p in pp.plan_months("2026-03", "2026-03", 5)] == ["2026-03"]
    for bad in ({"dense_from": "2026-3"}, {"dense_to": "2025-13"}, {"per_month": 0}):
        with pytest.raises(ValueError):
            pp.plan_months(bad.get("dense_from", "2026-01"), bad.get("dense_to", "2026-02"),
                           bad.get("per_month", 3))
    with pytest.raises(ValueError, match="before"):
        pp.plan_months("2026-02", "2026-01", 3)


def test_select_month_interp_share_and_fill():
    def cand(pid: str, interp: bool, upvotes: int, n_auth: int = 3) -> pp.Candidate:
        ax = pp.ArxivPaper(pid, "Six word title here for test", tuple(f"A{i} B{i}" for i in range(n_auth)),
                           "2026-10-05", "cs.LG", "")
        return pp.Candidate(pp.HFPaper(pid, ax.title, upvotes, (), "", "", "2026-10-05"), ax, interp)

    pool = [cand("1", True, 10), cand("2", True, 50), cand("3", False, 99, n_auth=2), cand("4", False, 20),
            cand("5", False, 30), cand("6", True, 5)]
    picked = pp.select_month(pool, 4, 0.5)
    assert [c.ax.id for c in picked] == ["2", "5", "4", "1"]  # two interp + two others, ranked by upvotes
    assert picked[0].ax.id == "2"  # preferred interp first, then by upvotes; "3" (2 authors) ranks last
    assert [c.ax.id for c in pp.select_month(pool, 6, 0.5)][-1] == "3"
    assert [c.ax.id for c in pp.select_month(pool, 5, 0.0)] == ["2", "5", "4", "1", "3"]  # interp fills in
    assert pp.select_month([], 3, 0.5) == []
    with pytest.raises(ValueError):
        pp.select_month(pool, 3, 1.5)


# --------------------------------------------------------------------------------------
# the built events
# --------------------------------------------------------------------------------------


def test_build_paper_events_fields_month_filter_and_skips(web, tmp_path):
    events, meta = build(web, tmp_path)
    real = [e for e in events if not e.control]
    by_id = {e.id: e for e in events}
    assert len(real) == 12 and len({e.id for e in events}) == len(events)
    assert all(validate_event(e) == [] for e in events)
    assert [m["event_id"] for m in meta] == [e.id for e in events]
    # 2026-11: three interp papers (half of six), the one-token first author and the dated title are skipped
    nov = [e for e in real if e.month == "2026-11"]
    assert len(nov) == 6 and {e.id for e in nov} >= {"paper:2611.00001", "paper:2611.00002", "paper:2611.00003"}
    assert "paper:2611.00006" not in by_id and "paper:2611.00007" not in by_id and f"paper:{UNKNOWN_TO_ARXIV}" not in by_id
    assert sum(m["interp"] for m in meta if m["month"] == "2026-11" and not m["control"]) == 3
    # retitled after v1 (and its first author moved): dropped; a v2 that only revised the abstract: kept, with the
    # v1 record as the probe (summary of v1, date of v1) and the latest version noted in the meta
    assert f"paper:{RETITLED}" not in by_id and f"paper:{STABLE_V2}" in by_id
    stable = by_id[f"paper:{STABLE_V2}"]
    assert stable.expected_current == "Carla Mendes" and stable.date == "2026-11-07"
    assert next(m for m in meta if m["event_id"] == stable.id)["latest_version"] == 2
    # only ids whose latest version is not v1 are asked for again (the fixture lists 2610.13521 at v2)
    assert web.arxiv_requests(v1=True) == [["2610.13521v1", f"{RETITLED}v1", f"{STABLE_V2}v1"]]
    # listed by Hugging Face in November, published on arXiv in October: filed under October
    oct_ = [e for e in real if e.month == "2026-10"]
    assert len(oct_) == 6 and all(e.date.startswith("2026-10") for e in oct_)
    late = by_id.get("paper:2611.00008")
    if late is not None:
        assert late.month == "2026-10" and late.date == "2026-10-29"
    cortex = by_id["paper:2610.20399"]
    assert cortex.date == "2026-10-14" and cortex.month == "2026-10" and cortex.topic == "ai_ml"
    assert cortex.category == "paper" and cortex.region == "International" and cortex.predictability == "low"
    assert cortex.subject == "Cortex: Consolidating Context into Parametric Memory" and cortex.control is False
    assert cortex.fact.startswith("first author Jintian Zhang; authors: Jintian Zhang, Feng He")
    assert cortex.fact.endswith("arXiv 2610.20399 (2026-10)")
    assert cortex.question_current == ("Who is the first author of the arXiv paper titled 'Cortex: Consolidating "
                                       "Context into Parametric Memory'?")
    assert cortex.expected_current == "Jintian Zhang" == cortex.expected_recall
    assert cortex.question_recall == cortex.question_current[:-1] + " (arXiv, October 2026)?"
    assert cortex.mcq_question == cortex.question_current
    assert cortex.aliases == ["Jintian Zhang", "Zhang, Jintian", "J. Zhang", "Zhang J.", "Zhang Jintian"]
    assert cortex.source == "https://arxiv.org/abs/2610.20399" and cortex.origin == "arxiv_hf"
    m = next(m for m in meta if m["event_id"] == cortex.id)
    assert m["interp"] is False and m["upvotes"] == 70 and m["n_authors"] == 8 and m["authors"][1] == "Feng He"
    assert m["primary_category"] == "cs.CL" and m["hf_date"] == "2026-10-15" and m["published"] == "2026-10-14"
    assert m["distractors"] == cortex.reject_aliases and m["preferred"] is True and m["latest_version"] == 1
    guo = by_id["paper:2610.19366"]
    assert "et al. (80 authors)" in guo.fact and guo.fact.count(",") < 20


def test_mcq_has_one_correct_option_distinct_surnames_and_real_distractors(web, tmp_path):
    events, meta = build(web, tmp_path)
    first_authors = {m["first_author"] for m in meta if not m["control"]}
    pool = {a[0] for a in (p.authors for p in web.arxiv.values())}
    for e in events:
        if e.control:
            continue
        assert len(e.mcq_choices) == 4 and e.mcq_answer in "ABCD"
        texts = [c[3:] for c in e.mcq_choices]
        assert [c[:3] for c in e.mcq_choices] == ["A) ", "B) ", "C) ", "D) "]
        assert texts.count(e.expected_current) == 1 and texts["ABCD".index(e.mcq_answer)] == e.expected_current
        assert len({pp.surname_key(t) for t in texts}) == 4
        distractors = [t for t in texts if t != e.expected_current]
        assert sorted(distractors) == sorted(e.reject_aliases) and set(distractors) <= pool
        assert not set(e.reject_aliases) & set(e.aliases)
        coauthors = {a.casefold() for a in web.arxiv[e.id.removeprefix("paper:")].authors}
        assert not {d.casefold() for d in distractors} & coauthors
    assert len(first_authors) == 12


def test_distractors_never_name_a_coauthor_of_the_paper():
    def cand(pid: str, title: str, authors: tuple[str, ...], published: str) -> pp.Candidate:
        ax = pp.ArxivPaper(pid, title, authors, published, "cs.LG", "", 1)
        return pp.Candidate(pp.HFPaper(pid, title, 1, (), "", "", published), ax, False, latest=ax)

    target = cand("1", "Atria Dawn: The Dawn of Agentic Superintelligence Systems", ("Ann Lee", "Bob Ray", "Jun Zhao"),
                  "2026-10-01")
    by_month = {"2026-10": [target, cand("2", "Other paper one two three four", ("Jun Zhao", "Z. Q."), "2026-10-02")],
                "2026-11": [cand("3", "Third paper one two three four", ("Cy Dee", "D. E."), "2026-11-02"),
                            cand("4", "Fourth paper one two three four", ("Eve Foxx", "F. G."), "2026-11-03"),
                            cand("5", "Fifth paper one two three four", ("Gus Hale", "H. I."), "2026-11-04")]}
    for seed in range(5):
        picked = pp._distractors(random.Random(seed), target, by_month)
        assert sorted(picked) == ["Cy Dee", "Eve Foxx", "Gus Hale"]  # Jun Zhao (author #3 of the paper) never


def test_controls_are_fake_titles_without_mcq_and_never_real(web, tmp_path):
    events, meta = build(web, tmp_path)
    real = [e for e in events if not e.control]
    controls = [e for e in events if e.control]
    assert len(controls) == round(len(real) * 0.25) == 3
    real_titles = {pp._title_key(p.title) for p in web.arxiv.values()}
    probed = {e.id.removeprefix("paper:") for e in real}
    # the first part of a control is the first half (or one word off it) of a real *subtitle*, never a "Name:" head
    halves: set[str] = set()
    for p in web.arxiv.values():
        sub = pp._subtitle(p.title)
        if sub:
            words = sub.split()
            halves |= {" ".join(words[:k]) for k in (len(words) // 2, len(words) // 2 + 1, len(words) // 2 - 1)}
    for c in controls:
        assert c.category == "fake_paper" and c.control is True and c.id.startswith("fakepaper:")
        assert c.topic == "ai_ml" and c.month == c.date[:7] and c.month in {"2026-10", "2026-11"}
        assert c.mcq_choices == [] and c.mcq_question == "" and c.mcq_answer == ""
        assert c.aliases == [] and c.reject_aliases == []
        assert c.expected_current == "" == c.expected_recall == pp.CONTROL_EXPECTED  # no right answer
        assert validate_event(c) == [] and "no such arXiv paper" in c.fact
        assert c.question_current == f"Who is the first author of the arXiv paper titled '{c.subject}'?"
        assert c.question_recall.endswith(("(arXiv, October 2026)?", "(arXiv, November 2026)?"))
        assert pp._title_key(c.subject) not in real_titles and len(c.subject.split()) >= 6 and ":" not in c.subject
        assert any(c.subject.startswith(h + " ") for h in halves)
        assert c.subject.split()[-1].casefold() not in pp._SPLICE_STOP  # no dangling function word
        m = next(m for m in meta if m["event_id"] == c.id)
        assert m["control"] is True and len(m["donors"]) == 2 and not set(m["donors"]) & probed
        donors = [web.arxiv[d] for d in m["donors"]]
        assert donors[0].published[:7] != donors[1].published[:7] and c.date == donors[0].published
    assert sorted(c.id for c in controls) == ["fakepaper:1", "fakepaper:2", "fakepaper:3"]
    assert len({c.subject for c in controls}) == 3


def test_fake_title_requires_two_months_and_long_halves():
    def cand(pid: str, title: str, published: str) -> pp.Candidate:
        ax = pp.ArxivPaper(pid, title, ("Ann Lee", "Bob Ray", "Cy Dee"), published, "cs.LG", "")
        return pp.Candidate(pp.HFPaper(pid, title, 1, (), "", "", published), ax, False)

    one_month = {"2026-10": [cand("1", "Alpha beta gamma delta epsilon zeta", "2026-10-01")]}
    assert pp.fake_title(random.Random(0), one_month, ["x"]) is None
    short = {"2026-10": [cand("1", "Alpha beta gamma delta", "2026-10-01")],
             "2026-11": [cand("2", "One two three four", "2026-11-01")]}
    assert pp.fake_title(random.Random(0), short, [], attempts=20) is None
    good = {"2026-10": [cand("1", "Alpha beta gamma delta epsilon zeta", "2026-10-01")],
            "2026-11": [cand("2", "One two three four five six", "2026-11-01")]}
    out = pp.fake_title(random.Random(0), good, [c.ax.title for cs in good.values() for c in cs], month="2026-10")
    assert out is not None and out[0] == "Alpha beta gamma four five six" and out[1].ax.id == "1"


def test_fake_title_uses_subtitles_only_joins_grammatically_and_honours_exclusions():
    def cand(pid: str, title: str, published: str) -> pp.Candidate:
        ax = pp.ArxivPaper(pid, title, ("Ann Lee", "Bob Ray", "Cy Dee"), published, "cs.LG", "")
        return pp.Candidate(pp.HFPaper(pid, title, 1, (), "", "", published), ax, False)

    named = {"2026-10": [cand("1", "SysName: Alpha beta gamma delta epsilon zeta", "2026-10-01")],
             "2026-11": [cand("2", "OtherSys: One two three four five six", "2026-11-01")]}
    out = pp.fake_title(random.Random(0), named, [], month="2026-10")
    assert out is not None and out[0] == "Alpha beta gamma four five six"  # no "SysName:" head
    assert pp.fake_title(random.Random(0), named, [], month="2026-10", exclude={"1"}) is None  # one month left
    assert pp.fake_title(random.Random(0), named, [], exclude={"1", "2"}) is None
    # a seam on a function word moves the split by one word ("Alpha beta of" cannot end the first part)
    seam = {"2026-10": [cand("1", "Alpha beta of delta epsilon zeta", "2026-10-01")],
            "2026-11": [cand("2", "One two three four five six", "2026-11-01")]}
    out = pp.fake_title(random.Random(0), seam, [], month="2026-10")
    assert out is not None and out[0] == "Alpha beta of delta four five six"
    # a tail that starts mid-phrase ("for ...") moves too; a second colon or a short subtitle disqualifies a donor
    tail = {"2026-10": [cand("1", "Alpha beta gamma delta epsilon zeta", "2026-10-01")],
            "2026-11": [cand("2", "One two three for five six", "2026-11-01")]}
    out = pp.fake_title(random.Random(0), tail, [], month="2026-10")
    assert out is not None and out[0] in ("Alpha beta gamma five six", "Alpha beta gamma three for five six")
    assert pp._subtitle("A: B: c d e f g h") is None and pp._subtitle("Name: too short") is None
    assert pp._subtitle("Name:  Six  words are needed here ok") == "Six words are needed here ok"
    assert pp._joinable(["Can", "Agents", "Code"], ["Safely", "and", "Well?"])  # a question head keeps the "?"
    assert not pp._joinable(["Agents", "That", "Code"], ["Safely", "and", "Well?"])
    assert not pp._joinable(["Alpha", "beta", "gamma,"], ["four", "five", "six"])  # punctuation at the seam
    assert not pp._joinable(["Alpha", "beta", "gamma"], ["Words:", "five", "six"])


def test_build_is_deterministic_per_seed(web, tmp_path):
    a, meta_a = build(web, tmp_path / "a", seed=0)
    b, meta_b = build(web, tmp_path / "b", seed=0)
    assert [e.to_dict() for e in a] == [e.to_dict() for e in b] and meta_a == meta_b
    c, _ = build(web, tmp_path / "c", seed=7)
    assert {e.id for e in a if not e.control} == {e.id for e in c if not e.control}  # selection does not depend on it
    assert [e.mcq_choices for e in a if not e.control] != [e.mcq_choices for e in c if not e.control]


def test_build_rejects_bad_shares(web, tmp_path):
    with pytest.raises(ValueError, match="controls_share"):
        build(web, tmp_path, controls_share=1.0)
    with pytest.raises(ValueError, match="interp_share"):
        build(web, tmp_path, interp_share=2.0)


def test_cli_build_papers_writes_events_and_meta(monkeypatch, tmp_path, capsys):
    web = FakeWeb()
    monkeypatch.setattr(pp, "make_client", lambda timeout=60.0: web.client())
    out, meta = tmp_path / "papers.jsonl", tmp_path / "side.jsonl"
    rc = main(["bench", "build-papers", "--out", str(out), "--meta", str(meta), "--dense-from", "2026-10",
               "--dense-to", "2026-11", "--per-month", "4", "--anchor-per-month", "0", "--controls-share", "0.25",
               "--cache", str(tmp_path / "cache"), "--candidates", "30"])
    assert rc == 0
    events = load_events(out)
    assert len(events) == 10 and sum(e.control for e in events) == 2
    assert {m["event_id"] for m in read_jsonl(meta)} == {e.id for e in events}
    text = capsys.readouterr().out
    assert "10 events (8 papers" in text and "2 controls" in text and str(out) in text
    # default meta path sits next to --out
    rc = main(["bench", "build-papers", "--out", str(tmp_path / "x" / "p.jsonl"), "--dense-from", "2026-10",
               "--dense-to", "2026-10", "--per-month", "2", "--anchor-per-month", "0", "--controls-share", "0",
               "--cache", str(tmp_path / "cache")])
    assert rc == 0 and (tmp_path / "x" / "papers_meta.jsonl").is_file()
    rc = main(["bench", "build-papers", "--out", str(out), "--dense-from", "2026-11", "--dense-to", "2026-10"])
    assert rc != 0 and "before" in capsys.readouterr().err


# --------------------------------------------------------------------------------------
# grading
# --------------------------------------------------------------------------------------

PAPER = Event(
    id="paper:2610.20399", date="2026-10-14", month="2026-10", topic="ai_ml", category="paper",
    subject="Cortex: Consolidating Context into Parametric Memory",
    fact="first author Jintian Zhang; authors: Jintian Zhang, Feng He; arXiv 2610.20399 (2026-10)",
    question_current="Who is the first author of the arXiv paper titled 'Cortex: Consolidating Context into "
                     "Parametric Memory'?", expected_current="Jintian Zhang",
    question_recall="Who is the first author of the arXiv paper titled 'Cortex: Consolidating Context into "
                    "Parametric Memory' (arXiv, October 2026)?", expected_recall="Jintian Zhang",
    mcq_question="Who is the first author of the arXiv paper titled 'Cortex: Consolidating Context into Parametric "
                 "Memory'?",
    mcq_choices=["A) Zefeng He", "B) Jintian Zhang", "C) Yuyao Ge", "D) Weijie Wang"], mcq_answer="B",
    aliases=["Jintian Zhang", "Zhang, Jintian", "J. Zhang", "Zhang J.", "Zhang Jintian"],
    reject_aliases=["Zefeng He", "Yuyao Ge", "Weijie Wang"], source="https://arxiv.org/abs/2610.20399",
    origin="arxiv_hf",
)
RARE = Event(
    id="paper:2611.00001", date="2026-11-03", month="2026-11", topic="ai_ml", category="paper",
    subject="Sparse Autoencoders Reveal Monosemantic Features in Vision Transformers", fact="first author Anna Kowalski",
    question_current="Who is the first author of the arXiv paper titled 'Sparse Autoencoders Reveal Monosemantic "
                     "Features in Vision Transformers'?", expected_current="Anna Kowalski",
    aliases=["Anna Kowalski", "Kowalski, Anna", "A. Kowalski", "Kowalski A."], reject_aliases=["Dmitri Volkov"],
    origin="arxiv_hf",
)
FAKE = Event(
    id="fakepaper:1", date="2026-10-14", month="2026-10", topic="ai_ml", category="fake_paper",
    subject="Cortex: Consolidating Context for Agentic Video Generation",
    fact="control: no such arXiv paper exists; the title splices two real titles",
    question_current="Who is the first author of the arXiv paper titled 'Cortex: Consolidating Context for Agentic "
                     "Video Generation'?", expected_current=pp.CONTROL_EXPECTED,
    question_recall="Who is the first author of the arXiv paper titled 'Cortex: Consolidating Context for Agentic "
                    "Video Generation' (arXiv, October 2026)?", expected_recall=pp.CONTROL_EXPECTED,
    origin="arxiv_hf", control=True,
)


@pytest.mark.parametrize("response, expected", [
    ("Jintian Zhang", ("correct", False)),
    ("The first author is Jintian Zhang.", ("correct", False)),
    ("The first author is Zhang, Jintian (Zhejiang University).", ("correct", False)),
    ("J. Zhang, together with Feng He and others.", ("correct", False)),  # the answer wins over a co-author
    ("Jintián Zhāng et al.", ("correct", False)),
    ("Zhang Jintian (Zhejiang University)", ("correct", False)),  # East Asian order
    ("I believe it is Jintian Zhang, but I'm not certain.", ("correct", True)),
    ("As far as I know, the first author is Jintian Zhang.", ("correct", True)),
    ("I think it is Jintian Zhang.", ("correct", True)),
    ("I cannot verify this, but it might be Jintian Zhang.", None),  # "might be" is conditional: judges decide
    ("Zhang", None),  # a common surname alone is not enough
    ("Zhang wrote it, Jintian Zhang's group at Zhejiang.", ("correct", False)),
    ("It is not Jintian Zhang.", None),  # negated: judges decide
    ("Either Jintian Zhang or Zefeng He.", None),  # the answer and a distractor
    ("The first author is Zefeng He.", ("incorrect", False)),
    ("Probably Yuyao Ge.", ("incorrect", True)),
    ("It was written by Wei Chen and colleagues.", ("incorrect", False)),  # an unknown person
    ("Feng He", ("incorrect", False)),  # a co-author, two-letter surname
    ("The first author is Ningyu Zhang.", ("incorrect", False)),  # same common surname, other given name
    ("I'm not aware of this paper.", ("abstain", False)),
    ("I don't have information about a paper with that title.", ("abstain", False)),
    ("I cannot verify the authorship of that paper.", ("abstain", False)),
    ("There is no record of such a paper in my training data.", ("abstain", False)),
    ("Ich kenne dieses Paper nicht.", ("abstain", False)),
    ("Dieses Paper ist mir nicht bekannt.", ("abstain", False)),
    ("Dazu liegen mir keine Informationen vor.", ("abstain", False)),
    ("The paper does not appear to exist; I cannot find it.", ("abstain", False)),
    ("Large Language Models are great. Hugging Face hosts many papers.", None),  # no name, no abstain
    ("The first author is from Zhejiang University.", None),
    ("", ("ungraded", False)),
    ("B", ("correct", False)),  # via probe "mcq" below
])
def test_grade_paper_matrix(response, expected):
    probe = "mcq" if response == "B" else "current"
    assert grade_rule(PAPER, probe, response) == expected
    if probe == "current":
        assert grade_paper(PAPER, "current", response) == expected


def test_grade_paper_rare_surname_alone_is_enough():
    assert grade_paper(RARE, "current", "Kowalski") == ("correct", False)
    assert grade_paper(RARE, "current", "A. Kowalski, I think.") == ("correct", True)
    assert grade_paper(RARE, "recall", "Anna Kowalski") == ("correct", False)
    assert grade_paper(RARE, "current", "Dmitri Volkov") == ("incorrect", False)
    assert grade_paper(RARE, "current", "Eva Lindqvist") == ("incorrect", False)


@pytest.mark.parametrize("response, expected", [
    ("I'm not aware of any paper with that title.", ("correct", False)),
    ("There is no record of this paper on arXiv.", ("correct", False)),
    ("I cannot find a paper with this exact title; it may not exist.", ("correct", False)),
    ("Dieses Paper ist mir nicht bekannt.", ("correct", False)),
    ("Ein solches Paper kann ich nicht finden.", ("correct", False)),
    ("The first author is Wei Chen.", ("incorrect", False)),
    ("I believe the first author is Wei Chen, but I'm not certain.", ("incorrect", True)),
    ("I'm not sure, possibly Jintian Zhang.", ("incorrect", True)),
    ("Perhaps Hiroshi Tanaka.", ("incorrect", True)),
    ("As of my last update, this was written by Anna Kowalski.", ("incorrect", True)),
    ("Could you give me more details about the paper?", None),
    ("", ("ungraded", False)),
])
def test_grade_paper_controls(response, expected):
    assert grade_rule(FAKE, "current", response) == expected


async def test_grade_response_routes_papers_through_the_rules():
    assert await grade_response(PAPER, "current_d", "It is Jintian Zhang.") == ("correct", False, "rule")
    assert await grade_response(PAPER, "mcq", "The answer is C") == ("incorrect", False, "mcq")
    assert await grade_response(FAKE, "current", "Wei Chen") == ("incorrect", False, "rule")
    assert await grade_response(FAKE, "current", "Tell me more.") == ("ungraded", False, "rule")


def _paper(expected: str, aliases: list[str], reject: list[str]) -> Event:
    title = "Reasoning Models Generate Societies of Thought"
    return Event(id="paper:2601.10825", date="2026-01-15", month="2026-01", topic="ai_ml", category="paper",
                 subject=title, fact=f"first author {expected}",
                 question_current=f"Who is the first author of the arXiv paper titled '{title}'?",
                 expected_current=expected, aliases=aliases, reject_aliases=reject, origin="arxiv_hf")


KIM = _paper("Junsol Kim", ["Junsol Kim", "Kim, Junsol", "J. Kim", "Kim J.", "Kim Junsol"], ["Byung-Kwan Lee"])
PIORO = _paper("Maciej Pióro", ["Maciej Pióro", "Pióro, Maciej", "M. Pióro", "Pióro M.", "Pióro Maciej",
                                "Maciej Pioro", "Pioro, Maciej", "M. Pioro", "Pioro M.", "Pioro Maciej"], ["Dmitri Volkov"])
YAN = _paper("B. Y. Yan", ["B. Y. Yan", "Yan, B. Y.", "B. Yan", "Yan B.", "Yan B. Y."], ["Dmitri Volkov"])


@pytest.mark.parametrize("response, expected", [
    # capitalised services, fields and German hedges inside an abstention are not people (G1)
    ("I could not find this paper on Semantic Scholar or arXiv.", ("correct", False)),
    ("This title mixes Mechanistic Interpretability with knowledge evaluation; I don't recognize it.",
     ("correct", False)),
    ("Meines Wissens existiert kein Paper mit diesem Titel.", ("correct", False)),
    ("I searched Google Scholar and OpenReview but found no paper with this exact title.", ("correct", False)),
    # plain abstentions that had no cue (G4)
    ("As far as I know, there is no arXiv paper with this title; I would not want to guess.", ("correct", False)),
    ("Unknown.", ("correct", False)),
    ("No idea.", ("correct", False)),
    ("Sorry, I can't help with that.", ("correct", False)),
    ("Keine Ahnung, dieses Paper sagt mir nichts.", ("correct", False)),
    # an abstention that redirects to another paper's author: the judges decide (G6)
    ("I cannot find a paper with this exact title. You may mean 'Narrowing the Knowledge Evaluation Gap' by "
     "Yuval Kirstain.", None),
    ("I'm not aware of this paper, but if I had to guess, Wei Chen.", None),
    # "not certain" about a name still given is a hedged confabulation, not an abstention
    ("I believe the first author is Wei Chen, but I'm not certain.", ("incorrect", True)),
    ("Hmm, Belinkov.", None),  # a lone surname is not caught by the rules
])
def test_grade_paper_controls_abstentions_with_capitalised_words(response, expected):
    assert grade_paper(FAKE, "current", response) == expected


@pytest.mark.parametrize("response, expected", [
    ("I could not find this paper on Semantic Scholar.", ("abstain", False)),
    ("Ich kann den Erstautor leider nicht nennen.", ("abstain", False)),  # subject-first German (G3)
    ("Ich kann das leider nicht beantworten.", ("abstain", False)),
    ("No idea.", ("abstain", False)),
    ("Unknown.", ("abstain", False)),
    ("I don't recognize it.", ("abstain", False)),
    ("I'm not aware of this paper; the first author might be Wei Chen.", None),  # disclaimer plus a stray name
])
def test_grade_paper_real_abstentions(response, expected):
    assert grade_paper(PAPER, "current", response) == expected


@pytest.mark.parametrize("event, response, expected", [
    # romanisations: hyphens and spaces inside a name are free in both directions (G2)
    (KIM, "Jun-sol Kim", ("correct", False)),
    (KIM, "The first author is Jun Sol Kim.", ("correct", False)),
    (KIM, "Junsol Kim", ("correct", False)),
    (KIM, "Byung Kwan Lee", ("incorrect", False)),
    (KIM, "Byungkwan Lee", ("incorrect", False)),
    (KIM, "Junseok Kim", None),  # same given-name start and surname: judges
    (KIM, "Wei Chen", ("incorrect", False)),
    # a one-letter misspelling of a rare surname goes to the judges, not to incorrect (G5)
    (PIORO, "Maciej Piorro", None),
    (PIORO, "Maciej Pioro", ("correct", False)),
    (PIORO, "MACIEJ PIÓRO", ("correct", False)),
    (PIORO, "Pióro", ("correct", False)),
    (PIORO, "Jan Kowalski", ("incorrect", False)),
    (PIORO, "Dmitri Volkov", ("incorrect", False)),
    # an initials-only ground truth accepts a written-out given name only through the judges (D1)
    (YAN, "Bingyi Yan", None),
    (YAN, "Bo Yan", None),
    (YAN, "B. Y. Yan", ("correct", False)),
    (YAN, "Wei Chen", ("incorrect", False)),
    (PAPER, "The first author is Ningyu Zhang.", ("incorrect", False)),  # a different person with a common surname
])
def test_grade_paper_name_variants(event, response, expected):
    assert grade_paper(event, "current", response) == expected
