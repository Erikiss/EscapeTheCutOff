"""Key, not value (docs/DESIGN_keys.md): masking, the plan on the real paper data and on fixtures, the cost estimate,
the runner with a scripted backend, the statistics and the report with its prediction verdicts."""

from __future__ import annotations

import asyncio
import json
import math
import re
from pathlib import Path
from typing import Any

import pytest

from cutoffmix.bench.io import load_events, read_jsonl
from cutoffmix.bench.keys import (
    ARMS_BY_SET,
    KEY_USER_TEMPLATE,
    SETS,
    TITLE,
    WITHHELD,
    _contains_answer,
    _title_spans,
    asserted_name,
    build_plan,
    build_report,
    estimate_cost,
    fisher_exact,
    mask_note,
    mask_title,
    mcnemar_exact,
    render_report,
    row_cost,
    run_plan,
    wilson,
)
from cutoffmix.bench.prompts import probe_messages
from cutoffmix.types import ChatResult, Event, ModelSpec

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "data" / "events" / "papers_events.jsonl"
RESULTS = ROOT / "runs" / "papers_results_final.jsonl"
FABLE, SOL, NEON = "claude-fable-5.2", "gpt-6.2-sol", "gemini-4-neon"
TODAY = "2026-11-27"


def forms(name: str) -> list[str]:
    given, surname = name.rsplit(" ", 1)
    return [name, f"{surname}, {given}", f"{given[0]}. {surname}", f"{surname} {given[0]}."]


def paper(i: int, name: str = "", *, subject: str | None = None, control: bool = False,
          rejects: tuple[str, ...] = (), aliases: list[str] | None = None) -> Event:
    subject = subject or f"Widget Paper {i}: Scaling Sparse Widgets Further"
    return Event(id=f"paper:{i}", date="2026-03-02", month="2026-03", topic="ai_ml",
                 category="fake_paper" if control else "paper", subject=subject, fact="",
                 question_current=f"Who is the first author of the arXiv paper titled '{subject}'?",
                 expected_current="" if control else name,
                 aliases=list(aliases if aliases is not None else ([] if control else forms(name))),
                 reject_aliases=list(rejects), control=control)


def result(event_id: str, model: str, grade: str, response: str = "", probe: str = "current_d",
           usage: dict | None = None) -> dict[str, Any]:
    return {"event_id": event_id, "model": model, "probe": probe, "response": response, "grade": grade,
            "hedged": False, "grader": "rule", "latency_s": 1.0, "usage": usage, "error": None}


# ---- masking ----------------------------------------------------------------------------------------------------

PIORO = paper(1, "Maciej Pióro", subject="MoE-Mamba: Efficient Selective State Space Models with Mixture of Experts")


@pytest.mark.parametrize("text", [
    "The first author is Maciej Pióro.",
    "The first author is Maciej Pioro.",                     # accent missing in the note
    "The first author is MACIEJ PIÓRO.",
    "The first author is Pióro, Maciej.",
    "The first author is M. Pióro.",
    "The first author is Pióro M.",
])
def test_every_form_of_the_answer_is_withheld(text):
    masked, leaked = mask_note(text, PIORO)
    assert masked == "The first author is [withheld]." and not leaked


def test_surname_and_given_name_alone_are_withheld_and_the_rest_stays():
    masked, leaked = mask_note("Pióro's group (with Kamil Ciebiera) wrote it; Maciej led the work.", PIORO)
    assert masked == "[withheld]'s group (with Kamil Ciebiera) wrote it; [withheld] led the work." and not leaked
    accented = paper(2, "Andres Marafioti")
    assert mask_note("Andrés Marafioti (of Hugging Face)", accented)[0] == "[withheld] (of Hugging Face)"


def test_hyphen_and_space_variants_of_a_name():
    kim = paper(3, "Jun-sol Kim")
    for text in ("Junsol Kim wrote it.", "Jun Sol Kim wrote it.", "Jun-sol Kim wrote it.", "JUN-SOL KIM wrote it."):
        assert mask_note(text, kim) == ("[withheld] wrote it.", False), text
    gur = paper(4, "Yoav Gur-Arieh")
    assert mask_note("Yoav Gur Arieh and Mor Geva; Gur-Arieh led it.", gur)[0] == "[withheld] and Mor Geva; [withheld] led it."


def test_short_name_tokens_keep_their_case_so_pronouns_stay():
    he = paper(5, "Feng He")
    masked, leaked = mask_note("Feng He led it; he also wrote the code. He is at Tsinghua.", he)
    assert masked == "[withheld] led it; he also wrote the code. [withheld] is at Tsinghua." and not leaked
    lu = paper(6, "Wei Lu")
    assert mask_note("Wei Lu, with Luo and Lu's students.", lu)[0] == "[withheld], with Luo and [withheld]'s students."


def test_title_is_masked_whole_and_by_its_parts():
    assert mask_title("On MoE-Mamba: Efficient Selective State Space Models with Mixture of Experts, see", PIORO) \
        == "On [title], see"
    assert mask_title("the moe mamba paper", PIORO) == "the [title] paper"
    assert mask_title("about efficient selective state space models with mixture of experts.", PIORO) \
        == "about [title]."
    assert mask_title("no title here", PIORO) == "no title here"
    short = paper(7, "Ann Bee", subject="Go: A Tiny Game")
    assert mask_title("I like Go and a tiny game", short) == "I like Go and [title]"   # 'Go' is too short alone


def test_adjacent_placeholders_collapse():
    masked, _ = mask_note("Maciej - Pióro and M. Pióro Maciej", PIORO)
    assert WITHHELD + " " + WITHHELD not in masked and masked.count(WITHHELD) <= 2


def test_a_form_the_mask_misses_is_reported_as_a_leak():
    masked, leaked = mask_note("The first author is Maciej **Pió**ro.", PIORO)   # markdown inside the name
    assert leaked and "**Pió**ro" in masked
    assert _contains_answer("The first author is [withheld].", PIORO) is False
    assert _contains_answer("The first author is Pioro.", PIORO) is True


def test_a_control_has_no_answer_to_mask():
    fake = paper(8, control=True, subject="Quantum Llamas: Teaching Alpacas to Tunnel")
    assert mask_note("The first author is Sho Maeoki.", fake) == ("The first author is Sho Maeoki.", False)


def test_asserted_name():
    event = paper(9, "Tamar Rott Shaham", rejects=("Sarah Schwettmann",))
    assert asserted_name("The first author is Sarah Schwettmann.", event) == "Sarah Schwettmann"
    assert asserted_name("My best guess is Jan Chorowski, but I can't verify it.", event) == "Jan Chorowski"
    quoted = ('I am not sure. The original "Vidu: a Highly Consistent, Dynamic and Skilled Text-to-Video Generator" '
              'paper had Fan Bao as first author.')
    assert asserted_name(quoted, event) == "Fan Bao"
    assert asserted_name("I'm not sure and won't guess a name.", event) is None
    assert asserted_name("", event) is None


# ---- plan on the real paper data ------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_plan():
    events = load_events(EVENTS)
    results = read_jsonl(RESULTS)
    rows, summary = build_plan(events, results, today=TODAY)
    return events, results, rows, summary


def test_real_plan_counts(real_plan):
    _, _, rows, summary = real_plan
    pairs = summary["pairs"]
    assert pairs[f"{FABLE}->{SOL}"]["dark_candidates"] == 96 and pairs[f"{FABLE}->{SOL}"]["dark"] == 96
    assert pairs[f"{FABLE}->{NEON}"]["dark"] == 50 and pairs[f"{NEON}->{SOL}"]["dark"] == 47
    for pair in (f"{SOL}->{FABLE}", f"{SOL}->{NEON}", f"{NEON}->{FABLE}"):
        assert pairs[pair]["dark_candidates"] == 1 and pairs[pair]["dark"] == 0 and pairs[pair]["keyed"] is False
    assert sorted((e["donor"], e["recipient"], e["dark"]) for e in summary["excluded"]) == sorted(
        [(SOL, FABLE, 1), (SOL, NEON, 1), (NEON, FABLE, 1)])
    assert all(pairs[p]["both_right"] == 10 for p in (f"{FABLE}->{SOL}", f"{FABLE}->{NEON}", f"{NEON}->{SOL}"))
    assert pairs[f"{SOL}->{FABLE}"]["control"] == 4 and pairs[f"{SOL}->{NEON}"]["control"] == 4
    assert sum(p["donor_wrong"] for p in pairs.values()) == 45 and summary["leaks"] == 0
    assert summary["calls_by_set_arm"]["dark"] == {arm: 193 for arm in ARMS_BY_SET["dark"]}
    assert summary["calls"] == len(rows) == 193 * 4 + 45 * 2 + 30 * 2 + 8 * 2
    assert summary["today"] == TODAY and summary["seed"] == 0


def test_real_plan_rows_are_well_formed(real_plan):
    events, _, rows, _ = real_plan
    by_id = {e.id: e for e in events}
    fields = {"id", "set", "donor", "recipient", "event_id", "arm", "recognised", "donor_grade", "donor_answer", "note",
              "note_event", "messages"}
    assert all(set(r) == fields for r in rows)
    assert len({r["id"] for r in rows}) == len(rows) and all(re.fullmatch(r"[0-9a-f]{16}", r["id"]) for r in rows)
    items: dict[tuple, set] = {}
    for r in rows:
        items.setdefault((r["set"], r["donor"], r["recipient"], r["event_id"]), set()).add(r["arm"])
    assert all(arms == set(ARMS_BY_SET[key[0]]) for key, arms in items.items())
    for r in rows:
        event = by_id[r["event_id"]]
        system, user = r["messages"]
        assert system["role"] == "system" and f"Today's date: {TODAY}." in system["content"]
        if r["arm"] == "alone":
            assert r["note"] is None and r["messages"] == probe_messages(event, "current_d", today=TODAY)
        else:
            assert user["content"] == KEY_USER_TEMPLATE.format(question=event.question_current, note=r["note"])
        if r["arm"] in ("key", "wrong_key"):
            assert not _contains_answer(r["note"], event), r["id"]
            assert not _title_spans(r["note"], event), r["id"]                 # neither its answer nor its title
        if r["arm"] == "wrong_key":
            other = by_id[r["note_event"]]
            assert other.id != event.id and other.expected_current != event.expected_current
            assert not _contains_answer(r["note"], other) and WITHHELD in r["note"]
        else:
            assert r["note_event"] is None
        if r["arm"] == "full" and r["set"] in ("dark", "both_right"):
            assert event.expected_current in r["note"] or _contains_answer(r["note"], event)
    dark = [r for r in rows if r["set"] == "dark"]
    assert all(r["donor_grade"] == "correct" and r["donor_answer"] for r in dark)
    assert all(isinstance(r["recognised"], bool) for r in dark)
    assert all(r["donor_grade"] == "incorrect" and r["donor_answer"] for r in rows if r["set"] in ("donor_wrong", "control"))


def test_real_plan_is_deterministic_and_seeded(real_plan):
    events, results, rows, summary = real_plan
    again, summary2 = build_plan(events, results, today=TODAY)
    assert again == rows and summary2 == summary
    other, _ = build_plan(events, results, today=TODAY, seed=1)
    wrong = {r["id"]: r["note_event"] for r in rows if r["arm"] == "wrong_key"}
    wrong1 = {r["id"]: r["note_event"] for r in other if r["arm"] == "wrong_key"}
    assert wrong.keys() == wrong1.keys() and wrong != wrong1                      # another seed draws other wrong keys


def test_real_plan_cost_estimate(real_plan):
    _, results, rows, _ = real_plan
    cost = estimate_cost(rows, results)
    assert cost["calls"] == len(rows) and set(cost["by_model"]) == {FABLE, SOL, NEON}
    assert 1.0 < cost["total_usd"] < 5.0
    assert math.isclose(sum(m["estimate_usd"] for m in cost["by_model"].values()), cost["total_usd"], abs_tol=1e-3)


# ---- plan on fixtures -------------------------------------------------------------------------------------------


NAMES = ["Alba Quist", "Bodo Rinne", "Cora Sell", "Dario Tamm", "Edda Ulm", "Femke Vos", "Gil Wendt"]


def fixture_data():
    events = [paper(i + 1, name) for i, name in enumerate(NAMES)]
    events.append(paper(20, control=True, subject="Quantum Llamas: Teaching Alpacas to Tunnel"))
    results = []
    for i, name in enumerate(NAMES[:6]):
        results.append(result(f"paper:{i + 1}", "a", "correct", f"The first author is {name}, with Zed Quillon."))
    for i in range(4):
        results.append(result(f"paper:{i + 1}", "b", "abstain", "I don't know."))
    for i in (4, 5):
        results.append(result(f"paper:{i + 1}", "b", "correct", f"It is {NAMES[i]}."))
    results.append(result("paper:7", "a", "incorrect", "My best guess is Wrong Person."))
    results.append(result("paper:7", "b", "abstain", "No idea."))
    results.append(result("paper:20", "a", "incorrect", "I believe the first author is Made Upname."))
    results.append(result("paper:20", "b", "correct", "I cannot find that paper."))
    results += [result(f"paper:{i}", "b", "correct" if i % 2 else "incorrect", "A", probe="mcq") for i in range(1, 5)]
    return events, results


def test_fixture_plan_sets_pairs_and_sampling():
    events, results = fixture_data()
    rows, summary = build_plan(events, results, pairs=[("a", "b"), ("b", "a")], both_right=1, min_items=3,
                               today=TODAY)
    ab = summary["pairs"]["a->b"]
    assert (ab["dark"], ab["donor_wrong"], ab["both_right"], ab["control"]) == (4, 1, 1, 1) and ab["keyed"]
    assert summary["pairs"]["b->a"]["keyed"] is False and summary["excluded"] == [{"donor": "b", "recipient": "a",
                                                                                   "dark": 0}]
    assert len(rows) == 4 * 4 + 2 + 2 + 2 and {r["donor"] for r in rows} == {"a"}
    dark = [r for r in rows if r["set"] == "dark"]
    assert {r["event_id"]: r["recognised"] for r in dark} == {"paper:1": True, "paper:2": False, "paper:3": True,
                                                               "paper:4": False}
    wrong = [r for r in dark if r["arm"] == "wrong_key"]
    assert all(r["note_event"] != r["event_id"] and r["note_event"] in {f"paper:{i}" for i in range(1, 7)}
               for r in wrong)
    assert [r["donor_answer"] for r in rows if r["set"] == "donor_wrong"] == ["Wrong Person"] * 2
    assert [r["donor_answer"] for r in rows if r["set"] == "control"] == ["Made Upname"] * 2
    picks = {build_plan(events, results, pairs=[("a", "b")], both_right=1, min_items=3, seed=s,
                        today=TODAY)[0][-3]["event_id"] for s in range(12)}
    assert picks == {"paper:5", "paper:6"}                                     # the seed decides which pair is sampled
    every, _ = build_plan(events, results, pairs=[("a", "b")], both_right=10, min_items=3, today=TODAY)
    assert sum(1 for r in every if r["set"] == "both_right") == 4              # all two candidates, two arms each
    none, summary_none = build_plan(events, results, pairs=[("a", "b")], min_items=5, today=TODAY)
    assert {r["set"] for r in none} == {"donor_wrong", "control"} and summary_none["excluded"][0]["dark"] == 4


def test_fixture_plan_excludes_a_leaking_key_and_validates_input():
    events, results = fixture_data()
    results[0] = result("paper:1", "a", "correct", "The first author is Alba **Qu**ist.")      # the mask misses it
    rows, summary = build_plan(events, results, pairs=[("a", "b")], min_items=3, today=TODAY)
    assert summary["pairs"]["a->b"]["leaks"] == 1 and summary["pairs"]["a->b"]["dark"] == 3
    assert "paper:1" not in {r["event_id"] for r in rows if r["set"] == "dark"}
    assert all(r["note_event"] != "paper:1" for r in rows if r["arm"] == "wrong_key")    # nor as a wrong key
    with pytest.raises(ValueError, match="without stored results"):
        build_plan(events, results, pairs=[("a", "zzz")])
    with pytest.raises(ValueError, match="two different"):
        build_plan(events, results, pairs=[("a", "a")])
    with pytest.raises(ValueError):
        build_plan(events, results, today="27.11.2026")


def test_estimate_cost_scales_the_prompt_part():
    events, results = fixture_data()
    usage = {"cost": 0.01, "cost_details": {"upstream_inference_cost": 0.01, "upstream_inference_prompt_cost": 0.002}}
    byok = {"cost": 0, "is_byok": True, "cost_details": {"upstream_inference_cost": 0.004,
                                                           "upstream_inference_prompt_cost": 0.001}}
    results = [dict(r, usage=usage if r["model"] == "b" else byok) for r in results]
    rows, _ = build_plan(events, results, pairs=[("a", "b")], min_items=3, today=TODAY)
    cost = estimate_cost(rows, results)
    entry = cost["by_model"]["b"]
    assert entry["calls"] == len(rows) and entry["mean_stored_cost_usd"] == 0.01
    alone = [r for r in rows if r["arm"] == "alone"]
    keyed = [r for r in rows if r["arm"] != "alone"]
    assert estimate_cost(alone, results)["total_usd"] == pytest.approx(0.01 * len(alone), abs=1e-4)
    assert estimate_cost(keyed, results)["total_usd"] > 0.01 * len(keyed)       # the notes make the prompt longer
    assert row_cost({"usage": byok}) == 0.004 and row_cost({"usage": None}) == 0.0 and row_cost(None) == 0.0


# ---- run --------------------------------------------------------------------------------------------------------


class ScriptedBackend:
    """``chat`` answers from ``reply(model, req)``: a ChatResult, a string (answer text) or an exception to raise."""

    def __init__(self, reply):
        self.reply = reply
        self.calls: list[tuple[str, Any]] = []

    async def chat(self, model, req):
        self.calls.append((model.key, req))
        out = self.reply(model, req)
        if isinstance(out, BaseException):
            raise out
        if isinstance(out, str):
            return ChatResult(text=out, usage={"cost": 0.01}, finish_reason="stop", latency_s=0.5)
        return out

    def stream_continuation(self, model, req):  # pragma: no cover - not used here
        raise NotImplementedError


MODELS = [ModelSpec(key=k, provider_id=f"x/{k}") for k in ("a", "b")]


def fixture_plan():
    events, results = fixture_data()
    rows, _ = build_plan(events, results, pairs=[("a", "b")], min_items=3, today=TODAY)
    return events, rows


def run(backend, rows, out, events, **kw):
    return asyncio.run(run_plan(backend, MODELS, rows, out, events=events, **kw))


def test_run_sends_grades_and_resumes(tmp_path):
    events, rows = fixture_plan()
    by_id = {e.id: e for e in events}

    def reply(model, req):
        event = by_id[req.metadata["event_id"]]
        if req.metadata["set"] in ("donor_wrong", "control"):
            return f"I think it is {req.metadata['donor_answer']}." if req.metadata["arm"] == "full" else "No idea."
        if req.metadata["arm"] in ("key", "full"):
            return f"The first author is {event.expected_current}."
        return "I don't know."
    backend = ScriptedBackend(reply)
    out = tmp_path / "keys.jsonl"
    got = run(backend, rows, out, events, concurrency=3)
    assert len(got) == len(rows) == len(backend.calls) and [r["id"] for r in got] == [r["id"] for r in rows]
    model, req = backend.calls[0]
    assert model == "b" and req.temperature == 0.0 and req.max_tokens == 4096
    assert set(req.metadata) == {"event_id", "probe", "model", "arm", "set", "donor", "item", "donor_answer"}
    assert req.metadata["probe"] == "current_d" and req.metadata["model"] == "b" and req.metadata["donor"] == "a"
    by_arm = {(r["set"], r["arm"]): r for r in got}
    assert by_arm[("dark", "key")]["grade"] == "correct" and by_arm[("dark", "alone")]["grade"] == "abstain"
    assert by_arm[("donor_wrong", "full")]["adopted"] is True and by_arm[("donor_wrong", "alone")]["adopted"] is False
    assert by_arm[("control", "full")]["adopted"] is True and by_arm[("dark", "key")]["adopted"] is None
    assert len(read_jsonl(out)) == len(rows)
    again = run(ScriptedBackend(reply), rows, out, events)
    assert len(again) == len(rows) and len(read_jsonl(out)) == len(rows)       # nothing sent twice


def test_run_records_errors_and_refusals_and_retries_errors(tmp_path):
    events, rows = fixture_plan()

    def reply(model, req):
        if req.metadata["arm"] == "wrong_key":
            return RuntimeError("upstream 503")
        if req.metadata["arm"] == "full":
            return ChatResult(text="", finish_reason="content_filter", usage={"cost": 0.0})
        return "I don't know."
    out = tmp_path / "keys.jsonl"
    got = run(ScriptedBackend(reply), rows, out, events)
    errors = [r for r in got if r["error"]]
    assert errors and all(r["arm"] == "wrong_key" and r["grade"] == "ungraded" and "503" in r["error"]
                          for r in errors)
    refused = [r for r in got if r["grader"] == "refusal"]
    assert refused and all(r["arm"] == "full" and r["grade"] == "ungraded" for r in refused)
    second = ScriptedBackend(lambda model, req: "I don't know.")
    run(second, rows, out, events)
    assert {req.metadata["arm"] for _, req in second.calls} == {"wrong_key"}    # only the failed calls again
    last = {r["id"]: r for r in read_jsonl(out)}
    assert not any(r["error"] for r in last.values())


def test_run_honours_limit_and_cost_cap(tmp_path):
    events, rows = fixture_plan()
    backend = ScriptedBackend(lambda model, req: "I don't know.")
    run(backend, rows, tmp_path / "limit.jsonl", events, limit=2)
    assert len(backend.calls) == 2
    capped = ScriptedBackend(lambda model, req: "I don't know.")
    run(capped, rows, tmp_path / "cap.jsonl", events, concurrency=1, max_cost=0.025)
    assert len(capped.calls) == 3                                               # 0.01 each: stops once 0.03 >= 0.025
    resumed = ScriptedBackend(lambda model, req: "I don't know.")
    run(resumed, rows, tmp_path / "cap.jsonl", events, concurrency=1, max_cost=0.025)
    assert resumed.calls == []                                                  # the earlier rows count toward the cap
    with pytest.raises(ValueError, match="not configured"):
        asyncio.run(run_plan(backend, MODELS[:1], rows, tmp_path / "x.jsonl", events=events))
    with pytest.raises(ValueError, match="not in the events file"):
        asyncio.run(run_plan(backend, MODELS, rows, tmp_path / "x.jsonl", events=events[:2]))


def test_run_cuts_an_interrupted_last_line(tmp_path):
    events, rows = fixture_plan()
    out = tmp_path / "keys.jsonl"
    run(ScriptedBackend(lambda model, req: "I don't know."), rows[:2], out, events)
    with out.open("a", encoding="utf-8") as f:
        f.write('{"id": "' + rows[2]["id"] + '", "grade": "abst')               # a crash in the middle of a write
    backend = ScriptedBackend(lambda model, req: "I don't know.")
    run(backend, rows[:3], out, events)
    assert len(backend.calls) == 1 and len(read_jsonl(out)) == 3


# ---- statistics ---------------------------------------------------------------------------------------------------


def test_wilson_known_values():
    assert wilson(8, 10) == pytest.approx((0.4902, 0.9433), abs=1e-4)
    assert wilson(0, 10) == pytest.approx((0.0, 0.2775), abs=1e-4)
    assert wilson(10, 10) == pytest.approx((0.7225, 1.0), abs=1e-4)
    assert wilson(0, 0) == (0.0, 1.0)
    with pytest.raises(ValueError):
        wilson(3, 2)


def test_exact_tests_known_values():
    assert mcnemar_exact(10, 2) == pytest.approx(0.03857, abs=1e-5)
    assert mcnemar_exact(2, 10) == mcnemar_exact(10, 2)
    assert mcnemar_exact(0, 0) == 1.0 and mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(8, 0) == pytest.approx(2 / 256)
    assert fisher_exact(3, 1, 1, 3) == pytest.approx(0.4857, abs=1e-4)
    assert fisher_exact(10, 0, 0, 10) == pytest.approx(2 / math.comb(20, 10))
    assert fisher_exact(0, 0, 0, 0) == 1.0


# ---- report -----------------------------------------------------------------------------------------------------


def plan_row(set_: str, pair: tuple[str, str], event: str, arm: str, recognised: bool | None = None) -> dict:
    return {"id": f"{set_}|{pair[0]}|{pair[1]}|{event}|{arm}", "set": set_, "donor": pair[0], "recipient": pair[1],
            "event_id": event, "arm": arm, "recognised": recognised, "donor_grade": "correct", "donor_answer": "X Y",
            "note": None, "note_event": None, "messages": []}


def result_row(row: dict, grade: str, adopted: bool | None = None, error: str | None = None) -> dict:
    return {"id": row["id"], "set": row["set"], "donor": row["donor"], "recipient": row["recipient"],
            "event_id": row["event_id"], "arm": row["arm"], "response": "" if error else "text", "grade": grade,
            "hedged": False, "grader": "rule", "adopted": adopted, "latency_s": 1.0,
            "usage": {"cost": 0.001}, "error": error}


def synthetic(dark_items: int = 20, adopt_items: int = 8, harm_items: int = 10):
    """Dark items: key right on 9 of 10 recognised and 1 of 10 unrecognised, alone right on 1, wrong key never,
    full on 18; donor_wrong: adopted under full on every item, never alone; both_right: key never breaks recall."""
    plan, results = [], []
    pair = ("a", "b")
    for i in range(dark_items):
        rec = i < dark_items // 2
        outcome = {"alone": i == 0, "key": (i < 9) if rec else (i == dark_items // 2), "wrong_key": False,
                   "full": i < dark_items - 2}
        for arm in ARMS_BY_SET["dark"]:
            row = plan_row("dark", pair, f"e{i}", arm, rec)
            plan.append(row)
            results.append(result_row(row, "correct" if outcome[arm] else "abstain"))
    for i in range(adopt_items):
        for arm in ARMS_BY_SET["donor_wrong"]:
            row = plan_row("donor_wrong", pair, f"w{i}", arm)
            plan.append(row)
            results.append(result_row(row, "incorrect" if arm == "full" else "abstain", adopted=arm == "full"))
    for i in range(harm_items):
        for arm in ARMS_BY_SET["both_right"]:
            row = plan_row("both_right", pair, f"r{i}", arm)
            plan.append(row)
            results.append(result_row(row, "correct"))
    return plan, results


def test_report_numbers_and_verdicts():
    plan, results = synthetic()
    results.append(result_row(plan[0], "ungraded", error="boom"))              # a later failed retry: last row counts
    results.append(result_row(plan[0], "abstain"))
    report = build_report(plan, results)
    dark = report["pairs"]["a->b"]["dark"]
    assert dark["items"] == 20 and dark["arms"]["key"]["k"] == 10 and dark["arms"]["key"]["n"] == 20
    assert dark["arms"]["alone"]["k"] == 0                                    # the last row of e0/alone is a miss
    assert report["paired"]["a->b"]["key_vs_wrong_key"] == {"b": 10, "c": 0, "p": round(mcnemar_exact(10, 0), 6),
                                                            "n_items": 20}
    rec = report["recognised"]["a->b"]
    assert rec["recognised"]["key"]["k"] == 9 and rec["unrecognised"]["key"]["k"] == 1
    assert report["adoption"]["a->b"]["donor_wrong"]["full"]["k"] == 8
    assert report["harm"]["a->b"]["key"]["k"] == 0 and report["totals"]["planned"] == len(plan)
    verdicts = {p["id"]: p["verdict"] for p in report["predictions"]}
    assert verdicts == {1: "supported", 2: "supported", 3: "supported", 4: "supported"}
    md = render_report(report)
    assert "# Key, not value: report" in md and "**supported**" in md and "a->b" in md
    json.dumps(report)                                                           # plain data


def test_report_verdicts_when_the_key_does_nothing_or_n_is_small():
    plan, results = synthetic()
    flat = [dict(r, grade="abstain") if r["arm"] == "key" else r for r in results]
    verdicts = {p["id"]: p["verdict"] for p in build_report(plan, flat)["predictions"]}
    # every key answer is now a miss: no recovery, nothing to tell apart, and the key breaks the correct recalls
    assert verdicts == {1: "not supported", 2: "not supported", 3: "supported", 4: "not supported"}
    small_plan, small_results = synthetic(dark_items=6, adopt_items=2, harm_items=3)
    small = {p["id"]: p["verdict"] for p in build_report(small_plan, small_results)["predictions"]}
    assert small == {1: "not testable (n too small)", 2: "not testable (n too small)",
                     3: "not testable (n too small)", 4: "not testable (n too small)"}
    empty = build_report(plan, [])
    assert empty["totals"]["rows"] == 0 and empty["totals"]["missing"] == len(plan)
    assert all(p["verdict"].startswith("not testable") for p in empty["predictions"])
    render_report(empty)


def test_report_leaves_ungraded_and_errors_out_of_n():
    plan, results = synthetic()
    broken = [dict(r, grade="ungraded") if r["arm"] == "wrong_key" and r["event_id"] == "e3" else r for r in results]
    report = build_report(plan, broken)
    arms = report["pairs"]["a->b"]["dark"]["arms"]
    assert arms["wrong_key"]["n"] == 19 and arms["wrong_key"]["ungraded"] == 1
    assert report["paired"]["a->b"]["n_items"] == 19                           # e3 lacks a graded wrong key
    assert report["totals"]["ungraded"] == 1


def test_sets_and_arms_constants():
    assert SETS == ("dark", "donor_wrong", "both_right", "control")
    assert ARMS_BY_SET["dark"] == ("alone", "key", "full", "wrong_key")
    assert "{question}" in KEY_USER_TEMPLATE and "{note}" in KEY_USER_TEMPLATE and TITLE == "[title]"
