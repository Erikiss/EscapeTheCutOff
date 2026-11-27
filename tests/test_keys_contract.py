"""Key, not value: the contract (docs/DESIGN_keys.md) states what the code does. A pre-registered rule text that changed
has its row in section 10's refinements table, and the judge-cost figures of section 9 follow from the stored data the
way section 9 says it prices a judge call (final review: prediction 1's rule text changed without a row; the judges
were priced like probe calls, which understated the worst case the owner approves the pilot on)."""

from __future__ import annotations

import re
from statistics import mean

import pytest
from keys_testlib import EVENTS, RESULTS, ROOT, TODAY

from cutoffmix.bench.grading import grade_rule
from cutoffmix.bench.io import load_events, load_results
from cutoffmix.bench.keys import build_plan, estimate_cost
from cutoffmix.bench.keys.run import adoption_judge_messages
from cutoffmix.bench.prompts import judge_messages, probe_messages

DESIGN = ROOT / "docs" / "DESIGN_keys.md"
JUDGE = "gpt-6.2-sol"  # the pilot's judge (design section 6)
PROMPT_USD, COMPLETION_USD = 2e-6, 10e-6  # per token, as section 9 states them


def section(number: int) -> str:
    """Section ``number`` of the design, with its line breaks folded into spaces."""
    text = DESIGN.read_text("utf-8")
    start = text.index(f"\n## {number}. ")
    end = text.find("\n## ", start + 1)
    return " ".join(text[start: end if end > 0 else len(text)].split())


def test_prediction_1_rule_text_has_its_refinement_row():
    assert "not supported if either is refuted; not testable if either is; else inconclusive" in section(7)
    rows = [line for line in DESIGN.read_text("utf-8").splitlines()
            if line.startswith("| prediction 1 with a comparison under the floor |")]
    assert len(rows) == 1 and "section 7 said inconclusive" in rows[0] and "not testable if either comparison is" \
        in rows[0]


@pytest.fixture(scope="module")
def priced():
    """Section 9's judge figures, computed the way it states: the judge's stored current_d usage gives the characters
    per prompt token and the mean and longest completion; a call's prompt is the judge's messages with the recipient's
    stored answer (at worst the longest stored answer or note) in them."""
    from cutoffmix.bench.keys.run import _adoption_rule  # which alone rows the rules leave to the judges

    events = load_events(EVENTS)
    results = load_results(RESULTS)
    plan, _ = build_plan(events, results, today=TODAY)
    by_id = {e.id: e for e in events}
    stored = {(r.event_id, r.model): r for r in results if r.probe == "current_d"}
    judge = [r for r in results if r.model == JUDGE and r.probe == "current_d"]
    chars = mean(sum(len(m["content"]) for m in probe_messages(by_id[r.event_id], "current_d", today="2026-11-25"))
                 / r.usage["prompt_tokens"] for r in judge)
    completion = mean(r.usage["completion_tokens"] for r in judge)
    longest_completion = max(r.usage["completion_tokens"] for r in judge)
    longest = max(max(len(r.response or "") for r in stored.values()), max(len(p["note"] or "") for p in plan))

    def usd(messages, worst: bool = False) -> float:
        tokens = sum(len(m["content"]) for m in messages) / chars
        return tokens * PROMPT_USD + (longest_completion if worst else completion) * COMPLETION_USD

    def answer(p):
        return stored[(p["event_id"], p["recipient"])].response

    def adoption(p, worst=False):
        return usd(adoption_judge_messages(by_id[p["event_id"]], p["donor_answer"], "x" * longest if worst
                                           else answer(p)), worst)

    grading = [p for p in plan if grade_rule(by_id[p["event_id"]], "current_d", answer(p)) is None]
    right = [p for p in plan if p["set"] in ("dark", "both_right") and p["arm"] == "full"]
    adopt_full = [p for p in plan if p["set"] in ("donor_wrong", "control") and p["arm"] == "full"]
    alone_due = []
    for p in (p for p in plan if p["set"] in ("donor_wrong", "control") and p["arm"] == "alone"):
        ruled = grade_rule(by_id[p["event_id"]], "current_d", answer(p))
        adopted, label = _adoption_rule(p, by_id[p["event_id"]], answer(p), ruled[0] if ruled else "ungraded")
        if label is not None and adopted is None:
            alone_due.append(p)
    figures = {
        "grading": (len(grading), sum(usd(judge_messages(by_id[p["event_id"]], "current_d", answer(p)))
                                      for p in grading),
                    sum(usd(judge_messages(by_id[p["event_id"]], "current_d", "x" * longest), True) for p in grading)),
        "right": (len(right), sum(adoption(p) for p in right), sum(adoption(p, True) for p in right)),
        "adopt_full": (len(adopt_full), sum(adoption(p) for p in adopt_full),
                       sum(adoption(p, True) for p in adopt_full)),
        "alone": (len(alone_due), sum(adoption(p) for p in alone_due), sum(adoption(p, True) for p in alone_due)),
    }
    return {"figures": figures, "chars": chars, "completion": completion, "longest_completion": longest_completion,
            "longest": longest, "recipients": estimate_cost(plan, results)["total_usd"], "calls": len(plan)}


def test_section_9_prices_every_judge_call_from_its_prompt(priced):
    text = section(9)
    f = priced["figures"]
    assert f"{priced['chars']:.2f} characters per prompt token" in text
    assert f"the mean stored completion ({priced['completion']:.0f} tokens)" in text
    assert f"longest stored answer or note ({priced['longest']} characters)" in text
    assert f"longest stored completion ({priced['longest_completion']} tokens)" in text
    n, typical, worst = f["grading"]
    assert f"leaves {n} of the {priced['calls']} answers to the judges: about {typical:.2f} USD ({worst:.2f} at worst)" \
        in text
    n, typical, worst = f["right"]
    assert n == 136 and f"at most the {n} `full` rows" in text and f"about {typical:.2f} USD ({worst:.2f} at worst)" \
        in text
    n, typical, worst = f["adopt_full"]
    assert n == 53 and f"at most the {n} `full` rows of `donor_wrong` and `control`, about {typical:.2f} USD " \
        f"({worst:.2f} at worst)" in text
    n, typical, worst = f["alone"]
    assert f"({n} of 53 in the replay, about {worst:.2f} USD at worst)" in text
    rules_only = f["grading"][1]
    every_full = sum(f[k][1] for k in ("grading", "right", "adopt_full", "alone"))
    at_worst = sum(f[k][2] for k in ("grading", "right", "adopt_full", "alone"))
    assert (f"In all: about {rules_only:.2f} USD when the rules decide every name, up to about {every_full:.2f} USD "
            f"when every `full` row goes to the judges, about {at_worst:.2f} USD at worst.") in text
    total = priced["recipients"] + at_worst
    assert total < 3.0 and f"about {total:.2f} USD at worst: under the pilot's cap of 3 USD by only about " \
        f"{3.0 - total:.2f} USD" in text
    # the README states the same figures (in German, decimal comma)
    readme = " ".join((ROOT / "README.md").read_text("utf-8").split())

    def comma(x: float) -> str:
        return f"{x:.2f}".replace(".", ",")
    assert f"bis etwa {comma(every_full)} $ für den Judge (im ungünstigsten Fall {comma(at_worst)} $, zusammen " \
        f"{comma(total)} $)" in readme
    assert re.search(rf"etwa {comma(rules_only)} \$ .* bis etwa {comma(every_full)} \$, wenn er jede", readme)
