"""Key, not value: the contract (docs/DESIGN_keys.md) states what the code does. A pre-registered rule text that changed
has its row in section 10's refinements table, and the judge-cost figures of section 9 follow from the stored data the
way section 9 says it prices a judge call (final review: prediction 1's rule text changed without a row; the judges
were priced like probe calls, which understated the worst case the owner approves the pilot on). The README's cost
paragraph, the recipients' estimate per model, the stored judge prices and the cap's overshoot are pinned too, and the
texts of the exit statuses and of the judges' scope say what the code does (review of the final review). With the
judges' scope (the judges decide every labelled answer that names the candidate), the counts follow that scope in the
replay, the overshoot is bounded at max_tokens, and the pilot's cap is 3.5 USD. Review of aafe7f9: the replay grades as
the run does (a rule grade incorrect next to the expected answer is unsure), the bound is eight of the dearest item
that can be in flight (the items in flight can belong to a neighbouring block), and the first block's own estimate,
the README's block order, section 10's overshoot row and both judges' budget are pinned too. Check of 8cc83c5: the
replay grades with the grading scope (with judges every rule grade incorrect or abstain of an answer that names the
expected answer goes to the grading judges), the overshoot's qualifiers name what has started (the 440th and the 671st
call) instead of where the cap is reached, the counts of the two-letter donor names are derived from the plan, and
section 10's rows of the grading scope, the open offers and the overshoot's position are pinned. Review of b86d67f: an
open offer names the expected answer in any form, the run and the report count the answers the judges' votes left
undecided, and the figures of sections 8 to 10 (from a mock run of the real plan with --judges, the replay with every
judge call failing, the plan and the stored data) and the README's plan command (--today) are pinned."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import itertools
import json
from statistics import mean

import pytest
from keys_testlib import EVENTS, RESULTS, ROOT, TODAY

import cutoffmix.cli
from cutoffmix.bench.grading import grade_with_judges
from cutoffmix.bench.io import load_events, load_results
from cutoffmix.bench.keys import build_plan, estimate_cost, row_cost, run_plan
from cutoffmix.bench.keys.run import _JUDGE_MAX_TOKENS, adoption_judge_messages
from cutoffmix.bench.prompts import judge_messages, probe_messages
from cutoffmix.types import ChatResult, ModelSpec

DESIGN = ROOT / "docs" / "DESIGN_keys.md"
README = ROOT / "README.md"
JUDGE = "gpt-6.2-sol"  # the pilot's judge (design section 6)
PROMPT_USD, COMPLETION_USD = 2e-6, 10e-6  # per token, as section 9 states them (checked against the stored rows)
CONCURRENCY = 8  # the default of 'bench keys run --concurrency'
MAX_TOKENS = 4096  # every planned call's max_tokens (design section 4)
CAP = 3.5  # the pilot's cap (design section 9)
SHORT = {"gpt-6.2-sol": "Sol", "gemini-4-neon": "Neon", "claude-fable-5.2": "Fable"}
# the judges' budget per call: the grading judges' (grading.grade_with_judges, called with its default) and the
# adoption judges' (run._JUDGE_MAX_TOKENS); the bound of section 9 prices both at it
BUDGET = inspect.signature(grade_with_judges).parameters["max_tokens"].default


def readme() -> str:
    return " ".join(README.read_text("utf-8").split())


def comma(x: float) -> str:
    return f"{x:.2f}".replace(".", ",")


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
    from cutoffmix.bench.keys.run import _adoption_rule, _judged  # which alone rows are in the judges' scope
    from cutoffmix.bench.keys.run import _rule_grade  # the run's grading step (review of aafe7f9)

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

    def chars_per_token(model: str) -> float:
        rows = [r for r in results if r.model == model and r.probe == "current_d"]
        return mean(sum(len(m["content"]) for m in probe_messages(by_id[r.event_id], "current_d",
                                                                   today="2026-11-25")) / r.usage["prompt_tokens"]
                    for r in rows)

    def usd(messages, worst: bool = False) -> float:
        tokens = sum(len(m["content"]) for m in messages) / chars
        return tokens * PROMPT_USD + (longest_completion if worst else completion) * COMPLETION_USD

    def answer(p):
        return stored[(p["event_id"], p["recipient"])].response

    def adoption(p, worst=False):
        return usd(adoption_judge_messages(by_id[p["event_id"]], p["donor_answer"], "x" * longest if worst
                                           else answer(p)), worst)

    # graded as the run grades with judges (run._rule_grade: a rule grade incorrect next to the expected answer is
    # unsure, and with judges every rule grade incorrect or abstain of an answer that names it is the judges'; check of
    # 8cc83c5); `before`: the grading judges' calls without that scope, as the rows written before it priced them
    grading = [p for p in plan if _rule_grade(by_id[p["event_id"]], answer(p), judges=True) is None]
    before = [p for p in plan if _rule_grade(by_id[p["event_id"]], answer(p)) is None]
    extra = [p for p in grading if p not in before]
    right = [p for p in plan if p["set"] in ("dark", "both_right") and p["arm"] == "full"]
    adopt_full = [p for p in plan if p["set"] in ("donor_wrong", "control") and p["arm"] == "full"]
    alone_due = []  # the alone rows of the replay that mention the donor's name: the judges decide them
    for p in (p for p in plan if p["set"] in ("donor_wrong", "control") and p["arm"] == "alone"):
        ruled = _rule_grade(by_id[p["event_id"]], answer(p), judges=True)
        _, label = _adoption_rule(p, by_id[p["event_id"]], answer(p), ruled[0] if ruled else "ungraded")
        if _judged(p, by_id[p["event_id"]], answer(p), label):
            alone_due.append(p)
    def graded(rows):
        return (len(rows), sum(usd(judge_messages(by_id[p["event_id"]], "current_d", answer(p))) for p in rows),
                sum(usd(judge_messages(by_id[p["event_id"]], "current_d", "x" * longest), True) for p in rows))
    figures = {
        "grading": graded(grading), "extra": graded(extra), "before": graded(before),
        "right": (len(right), sum(adoption(p) for p in right), sum(adoption(p, True) for p in right)),
        "adopt_full": (len(adopt_full), sum(adoption(p) for p in adopt_full),
                       sum(adoption(p, True) for p in adopt_full)),
        "alone": (len(alone_due), sum(adoption(p) for p in alone_due), sum(adoption(p, True) for p in alone_due)),
    }
    # the cap's overshoot: up to --concurrency items in flight, each the most expensive stored call of a recipient of
    # the plan and two judge calls (grading and adoption) at their worst
    worst_judge = max(max(usd(judge_messages(by_id[p["event_id"]], "current_d", "x" * longest), True),
                          usd(adoption_judge_messages(by_id[p["event_id"]], p["donor_answer"] or "", "x" * longest),
                              True)) for p in plan)
    recipients = {p["recipient"] for p in plan}
    worst_call = max(row_cost({"usage": r.usage}) for r in results if r.probe == "current_d" and r.model in recipients)
    estimate = estimate_cost(plan, results)
    # the overshoot bound at max_tokens (review of 863ca47: the stored maxima are no bound): per block of one recipient
    # in the plan's order, --concurrency items, each the recipient's call at max_tokens completion tokens with the
    # block's longest planned prompt, at the stored prices per token, and two judge calls at their own max_tokens with
    # an answer of max_tokens tokens in the prompt
    prices = {}
    for model in recipients:
        rows = [r for r in results if r.model == model and r.probe == "current_d"]
        prompt = {r.usage["cost_details"]["upstream_inference_prompt_cost"] / r.usage["prompt_tokens"] for r in rows}
        done = {r.usage["cost_details"]["upstream_inference_completions_cost"] / r.usage["completion_tokens"]
                for r in rows}
        prices[model] = (max(prompt), max(done), chars_per_token(model), min(prompt), min(done))
    template = max(max(sum(len(m["content"]) for m in judge_messages(by_id[p["event_id"]], "current_d", "")),
                       sum(len(m["content"]) for m in adoption_judge_messages(by_id[p["event_id"]],
                                                                              p["donor_answer"] or "", "")))
                   for p in plan)
    judge_bound = (template / chars + MAX_TOKENS) * PROMPT_USD + BUDGET * COMPLETION_USD

    def item_bound(p) -> float:
        prompt_usd, completion_usd, cpt, _, _ = prices[p["recipient"]]
        tokens = sum(len(m["content"]) for m in p["messages"]) / cpt
        return tokens * prompt_usd + MAX_TOKENS * completion_usd + 2 * judge_bound

    blocks = [(model, len(rows), max(item_bound(p) for p in rows))
              for model, rows in ((k, list(g)) for k, g in itertools.groupby(plan, key=lambda p: p["recipient"]))]
    first = plan[: blocks[0][1]]
    first_ids = {p["id"] for p in first}
    # the items in flight when the cap is reached were started before, in the plan's order: the call that reaches it
    # can have up to --concurrency - 1 later calls in flight with it, so from this call on (1-based) a Fable call can be
    # among them
    first_fable = next(i for i, p in enumerate(plan) if p["recipient"] == "claude-fable-5.2")
    first_neon = next(i for i, p in enumerate(plan) if p["recipient"] == "gemini-4-neon")
    # calls start in the plan's order: while no call of a dearer recipient has started, none can be in flight (check of
    # 8cc83c5: the positions of the call that reaches the cap bounded nothing)
    first_judges = sum(usd(judge_messages(by_id[p["event_id"]], "current_d", "x" * longest), True)
                       for p in grading if p["id"] in first_ids)
    first_judges += sum(adoption(p, True) for p in right + adopt_full + alone_due if p["id"] in first_ids)
    return {"figures": figures, "chars": chars, "completion": completion, "longest_completion": longest_completion,
            "longest": longest, "recipients": estimate["total_usd"], "by_model": estimate["by_model"],
            "calls": len(plan), "judge_rows": judge, "dark_full": sum(1 for p in right if p["set"] == "dark"),
            "both_right_full": sum(1 for p in right if p["set"] == "both_right"), "worst_call": worst_call,
            "overshoot": CONCURRENCY * (worst_call + 2 * worst_judge), "prices": prices, "blocks": blocks,
            "bound": {SHORT[m]: CONCURRENCY * max(b for k, _, b in blocks if k == m) for m in recipients},
            "item": {SHORT[m]: max(b for k, _, b in blocks if k == m) for m in recipients},
            "first_estimate": estimate_cost(first, results)["total_usd"],
            "first": estimate_cost(first, results)["total_usd"] + first_judges,
            "first_neon_call": first_neon + 1, "first_fable_call": first_fable + 1,
            "plan": plan, "events": by_id, "stored": stored}


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
    # the grading scope's share of them (check of 8cc83c5): three dark items on five arms in the replay
    n_extra, typical, worst = f["extra"]
    assert (n_extra, n - n_extra) == (15, 71) and f"{n_extra} of them come from the grading scope (section 6)" in text
    assert f"on all five arms: about {typical:.2f} USD ({worst:.2f} at worst)" in text
    # the judges' scope (review of 863ca47): every correct right-name row and every donor row that names the donor's
    # name goes to the judges; the replay counts the alone rows that mention it (14 of 53; the rules left 10 open)
    n, typical, worst = f["right"]
    assert n == 136 and "every correct `full` row of `dark` and `both_right`, so at most the 136 `full` rows" in text
    assert f"(20): about {typical:.2f} USD ({worst:.2f} at worst)" in text
    n, typical, worst = f["adopt_full"]
    assert n == 53 and f"at most the {n} `full` rows of `donor_wrong` and `control`, about {typical:.2f} USD " \
        f"({worst:.2f} at worst)" in text
    n, typical, worst = f["alone"]
    assert n == 14 and f"the `alone` rows that mention the donor's name ({n} of 53 in the replay, about {worst:.2f} " \
        "USD at worst)" in text
    every = sum(f[k][1] for k in ("grading", "right", "adopt_full", "alone"))
    at_worst = sum(f[k][2] for k in ("grading", "right", "adopt_full", "alone"))
    assert f"In all: up to about {every:.2f} USD, about {at_worst:.2f} USD at worst" in text
    assert "when the rules decide every name" not in text and "in a way the rules leave to the judges" not in text
    total = priced["recipients"] + at_worst
    # the total is the recipients' mean estimate plus the judges at their worst: no worst case, no bound (review of the
    # final review: it read "about 2.94 USD at worst"); under the old cap of 3 too little room is left (863ca47)
    assert 3.0 - 0.1 < total < CAP and f"Added to the recipients' estimate that is about {total:.2f} USD, about " \
        f"{CAP - total:.2f} USD under the pilot's cap of 3.5 USD. It is no bound:" in text
    assert f"Under the cap of 3 USD of the earlier drafts the judges at their worst would leave about " \
        f"{3.0 - total:.3f} USD" in text
    assert f"which still leaves about {CAP - total:.2f} USD with the grading scope" in text
    assert "USD at worst: under the pilot's cap" not in text
    # the README states the same figures (in German, decimal comma)
    text_de = readme()
    assert f"bis etwa {comma(every)} $ für den Judge (im ungünstigsten Fall {comma(at_worst)} $; zusammen " \
        f"geschätzt {comma(total)} $, eine Schätzung, keine Obergrenze: die setzt `--max-cost`, freigegeben werden " \
        "sollen 3,5 $)" in text_de
    n_right, n_adopt, n_alone = f["right"][0], f["adopt_full"][0], f["alone"][0]
    cents = round(100 * (CAP - total))
    assert f"etwa {comma(f['grading'][1])} $ für die Antworten, die die Regeln nicht bewerten können oder gegen die " \
        f"genannte richtige Antwort werten (in der Nachrechnung {f['grading'][0]}, davon {f['extra'][0]} durch die " \
        f"neue Regel), und, da er jede Antwort mit dem Namen entscheidet, bis etwa {comma(every)} $ (bis zu " \
        f"{n_right} Aufrufe für den richtigen Namen, {n_adopt} für den falschen in den `full`-Antworten, in der " \
        f"Nachrechnung {n_alone} beim erneuten Fragen), im ungünstigsten Fall etwa {comma(at_worst)} $. Mit der " \
        f"Schätzung für die Empfänger sind das etwa {comma(total)} $, also etwa {cents} Cent unter `--max-cost " \
        f"3.5`. Unter dem früheren Deckel von 3 $ bliebe nur etwa {f'{100 * (3.0 - total):.1f}'.replace('.', ',')} " \
        "Cent" in text_de
    assert "im ungünstigsten Fall etwa 2" not in text_de and "Zusammen sind das im ungünstigsten Fall" not in text_de
    assert "unter `--max-cost 3`" not in text_de


def test_the_recipients_estimate_and_the_judge_prices_are_the_stored_ones(priced):
    """The figures of section 9 and of the README that no other test pins (review of the final review): the
    recipients' estimate per model, the number of stored judge rows, the split of the right-name rows, and the prices
    per token, read from the stored judge rows' upstream costs."""
    neon, sol, fable = (priced["by_model"][k]["estimate_usd"] for k in ("gemini-4-neon", "gpt-6.2-sol",
                                                                       "claude-fable-5.2"))
    split = f"(Neon {neon:.2f}, Sol {sol:.2f}, Fable {fable:.2f})"
    assert f"**{priced['recipients']:.2f} USD** {split}" in section(9)
    assert f"Geschätzt aus den gespeicherten Kosten (Mittelwerte): {comma(priced['recipients'])} $ " \
        f"{split.replace('.', ',')}" in readme()
    assert f"Der Pilot hat {priced['calls']} Aufrufe, kostet geschätzt {comma(priced['recipients'])} $" in readme()
    rows = priced["judge_rows"]
    prompt = {r.usage["cost_details"]["upstream_inference_prompt_cost"] / r.usage["prompt_tokens"] for r in rows}
    completion = {r.usage["cost_details"]["upstream_inference_completions_cost"] / r.usage["completion_tokens"]
                  for r in rows}
    assert max(prompt) - min(prompt) < 1e-15 and max(completion) - min(completion) < 1e-15
    assert PROMPT_USD == pytest.approx(min(prompt), rel=1e-9) and COMPLETION_USD == pytest.approx(min(completion),
                                                                                                   rel=1e-9)
    assert f"`{JUDGE}` ({len(rows)} calls; {PROMPT_USD * 1e6:g} USD per million prompt tokens, " \
        f"{COMPLETION_USD * 1e6:g} per million completion tokens" in section(9)
    assert f"at most the {priced['dark_full'] + priced['both_right_full']} `full` rows of `dark` " \
        f"({priced['dark_full']}) and `both_right` ({priced['both_right_full']})" in section(9)


def test_the_cap_stops_new_calls_and_the_calls_in_flight_can_pass_it(priced):
    """The cap is checked before an item starts and the items in flight finish with their judge calls (section 6), so
    the run can end above it: by up to --concurrency items (review of the final review: section 9 and the README said
    the run cannot spend more). The stored maxima give an estimate only; every call may use max_tokens, so section 9
    bounds the overshoot at max_tokens (review of 863ca47: "at most about 0.23 USD" was no bound). The items in flight
    were started before, in the plan's order, so near a block's end they belong to the next block: the bound is eight
    of the dearest item that can be in flight (review of aafe7f9: the figures per block failed there, also for the last
    4 Sol calls). A slow call keeps its slot while later calls start and finish, so the position of the call that
    reaches the cap bounds nothing; calls start in the plan's order, so what has started does: eight Sol items while no
    Neon call has started, eight Neon items while no Fable call has (check of 8cc83c5)."""
    text, text_de = section(9), readme()
    overshoot, worst_call = priced["overshoot"], priced["worst_call"]
    assert f"From the stored maxima that is about {overshoot:.2f} USD at the default of {CONCURRENCY} (each item the " \
        f"most expensive stored recipient call, {worst_call:.3f} USD, and two judge calls at their worst): an " \
        "estimate, no bound." in text
    assert f"at the default of {CONCURRENCY} at most about" not in text and "um höchstens etwa 0,23" not in text_de
    prices = {SHORT[m]: v for m, v in priced["prices"].items()}
    for prompt, completion, _, low_prompt, low_completion in prices.values():  # one price per token and model
        assert prompt == pytest.approx(low_prompt, rel=1e-9) and completion == pytest.approx(low_completion, rel=1e-9)
    per_million = {name: (f"{prompt * 1e6:g}", f"{completion * 1e6:g}") for name, (prompt, completion, *_) in
                   prices.items()}
    judge_prices = f"{PROMPT_USD * 1e6:g} and {COMPLETION_USD * 1e6:g}"
    assert (f"Sol {' and '.join(per_million['Sol'])} USD, Neon {' and '.join(per_million['Neon'])}, Fable "
            f"{' and '.join(per_million['Fable'])}, the judge {judge_prices}") in text
    item, bound = priced["item"], priced["bound"]
    assert f"costs at most about {item['Sol']:.2f} USD (Sol), {item['Neon']:.2f} (Neon) or {item['Fable']:.2f} " \
        "(Fable)" in text
    # the bound: eight of the dearest item that can be in flight, by what has started (calls start in the plan's order)
    top = max(bound.values())
    assert top == bound["Fable"] and bound["Sol"] < bound["Neon"] < top
    neon, fable = priced["first_neon_call"], priced["first_fable_call"]
    assert (neon, fable) == (priced["blocks"][0][1] + 1, sum(n for _, n, _ in priced["blocks"][:2]) + 1) == (440, 671)
    assert f"So at the default of {CONCURRENCY} the run ends at most about {bound['Sol']:.2f} USD above the cap when " \
        f"the cap is reached before the first Neon call has started (before the {neon}th call: eight Sol items), at " \
        f"most about {bound['Neon']:.2f} when it is reached before the first Fable call has started (before the " \
        f"{fable}st: eight Neon items), and at most about {top:.2f} USD otherwise (eight Fable items); a lower " \
        "`--concurrency` lowers each in proportion." in text
    last = priced["blocks"][-1]
    assert f"{top:.2f} USD bounds the overshoot ({bound['Neon']:.2f} if the cap is reached before the first Fable " \
        "call has started)" in text
    for gone in ("the last 7 calls of", "664", "away from a dearer block", "when it is reached in a Sol block",
                 "(or the last 4 Sol calls)"):
        assert gone not in text, gone
    order = ", then ".join(f"{SHORT[m]} {n}" for m, n, _ in priced["blocks"][:2])
    rest = ", ".join(f"{SHORT[m]} {n}" for m, n, _ in priced["blocks"][2:-1])
    assert f"blocks of one recipient: {order.replace(' 439', ' 439 first')}, {rest} and {SHORT[last[0]]} {last[1]}" \
        in text
    # the first block's own estimate (review of aafe7f9: the test read the estimate of all Sol calls)
    assert priced["first_estimate"] != priced["by_model"]["gpt-6.2-sol"]["estimate_usd"]
    assert f"first Sol block costs about {priced['first_estimate']:.2f} USD, about {priced['first']:.2f} with its " \
        "judges at their worst" in text
    # both judges' budget, as section 9 and the README state it
    assert BUDGET == _JUDGE_MAX_TOKENS == 1024
    assert f"every judge call {BUDGET} (the grading judges' budget and the adoption judges' alike)" in text
    assert f"Jeder Aufruf darf `max_tokens` {MAX_TOKENS} ausschöpfen, ein Judge-Aufruf {BUDGET}." in text_de
    # the README: the estimate, the bound, the block order with the first block's estimate, where a Fable call can be
    # in flight
    assert f"Aus den längsten gespeicherten Antworten geschätzt sind das bei der Voreinstellung {CONCURRENCY} etwa " \
        f"{comma(overshoot)} $, aber das ist keine Grenze" in text_de
    assert f"Zu den gespeicherten Preisen sind es dann höchstens etwa {comma(top)} $ (acht Fable-Aufrufe)" in text_de
    blocks_de = [(SHORT[m], n) for m, n, _ in priced["blocks"]]
    tail = ", ".join(f"{n} an {name}" for name, n in blocks_de[1:-1]) + f" und {blocks_de[-1][1]} an {blocks_de[-1][0]}"
    assert f"Der Plan schickt erst {blocks_de[0][1]} Aufrufe an {blocks_de[0][0]} (geschätzt etwa " \
        f"{comma(priced['first_estimate'])} $, mit dem Judge im ungünstigsten Fall etwa {comma(priced['first'])} $), " \
        f"dann {tail}." in text_de
    assert f"Höchstens etwa {comma(bound['Sol'])} $ über dem Deckel endet der Lauf, wenn der Deckel erreicht wird, " \
        f"bevor der erste Neon-Aufruf gestartet ist (vor dem {neon}. Aufruf: acht Sol-Aufrufe), höchstens etwa " \
        f"{comma(bound['Neon'])} $, wenn er erreicht wird, bevor der erste Fable-Aufruf gestartet ist (vor dem " \
        f"{fable}.: acht Neon-Aufrufe), sonst höchstens etwa {comma(top)} $." in text_de
    assert f"und dann gilt die Grenze von {comma(top)} $ ({comma(bound['Neon'])} $, solange kein Fable-Aufruf " \
        "gestartet ist)" in text_de
    assert "aber ab dem 664. der 746 Aufrufe kann ein Fable-Aufruf mitlaufen" not in text_de     # the old qualifiers
    assert "(„vor den letzten 7 Aufrufen eines Blocks“, „ab dem 664. Aufruf“) keine Grenzen" in text_de
    assert "wenn der Deckel während der Sol-Aufrufe erreicht wird" not in text_de
    assert "cannot spend more" not in text and "mehr wird also nicht ausgegeben" not in text_de
    assert "so a run can end above the cap by what the items in flight cost" in section(6)
    assert _keys_run_parser().get_default("concurrency") == CONCURRENCY           # the default the figure assumes
    max_cost = next(a for a in _keys_run_parser()._actions if "--max-cost" in a.option_strings)
    help_text = " ".join((max_cost.help or "").split())
    assert f"at max_tokens and the default of {CONCURRENCY} up to about {top:.2f} USD (eight Fable items: the items " \
        f"in flight can belong to a later block of the plan), {bound['Sol']:.2f} if the cap is reached before the " \
        f"first Neon call has started and {bound['Neon']:.2f} before the first Fable call has" in help_text
    assert "the pilot asks for 3.5" in help_text
    assert "in a Sol block of the plan" not in help_text


class _AtTheBound:
    """A recipient backend for the boundary cases of the cap's overshoot: every call costs the per-item bound of its
    model (section 9) and takes the latency the case gives its item; no judges."""

    def __init__(self, cost, latency):
        self.cost, self.latency = cost, latency

    async def chat(self, model, req):
        await asyncio.sleep(self.latency(req.metadata["item"], model.key))
        return ChatResult(text="I don't know.", usage={"cost": self.cost[model.key]}, finish_reason="stop")

    def stream_continuation(self, model, req):  # pragma: no cover - unused
        raise NotImplementedError


@pytest.mark.parametrize("case", ["neon_block_end", "last_sol_calls"])
def test_the_items_in_flight_at_a_block_boundary_can_belong_to_a_dearer_block(tmp_path, priced, case):
    """Section 9's bound is eight of the dearest item that can be in flight, not eight of the block that reaches the
    cap (review of aafe7f9): the real run_plan on two windows of the real plan, every call at its model's per-item
    bound. At the end of the Neon 231 block a slow Neon call reaches the cap while the first 7 Fable calls are in
    flight; among the last 4 Sol calls, the last Neon and the 2 Fable calls are still in flight. Either way the run
    ends above the cap by more than eight items of the block that reached it, and by at most eight Fable items."""
    plan, item = priced["plan"], priced["item"]
    cost = {model: item[name] for model, name in SHORT.items()}
    first_fable = next(i for i, p in enumerate(plan) if p["recipient"] == "claude-fable-5.2")
    last_sol = len(plan) - priced["blocks"][-1][1]
    if case == "neon_block_end":                          # a slow Neon call (the block's last) reaches the cap
        lo, hi, slow, crossing = first_fable - 20, first_fable + 30, {first_fable - 1: 0.12}, "gemini-4-neon"
        latency = {"claude-fable-5.2": 0.25}
    else:                                                 # a Sol call of the last block reaches it
        lo, hi, slow, crossing = len(plan) - 20, len(plan), dict.fromkeys(range(last_sol - 4, last_sol), 0.25), \
            "gpt-6.2-sol"
        latency = {"gpt-6.2-sol": 0.08}
    window = plan[lo:hi]
    slow_ids = {plan[i]["id"]: t for i, t in slow.items()}
    first_slow = min(slow)
    assert plan[first_slow]["recipient"] == "gemini-4-neon" and plan[first_fable - 1]["recipient"] == "gemini-4-neon"
    cap = sum(cost[p["recipient"]] for p in plan[lo:first_slow]) + 0.01      # crossed by the next call to finish
    backend = _AtTheBound(cost, lambda item_id, model: slow_ids.get(item_id, latency.get(model, 0.002)))
    models = [ModelSpec(key=k, provider_id=f"x/{k}") for k in SHORT]
    out = tmp_path / "keys.jsonl"
    got = asyncio.run(run_plan(backend, models, window, out, events=priced["events"], max_cost=cap,
                               concurrency=CONCURRENCY, backend_name="scripted"))
    assert got.stopped_by == ("max_cost" if case == "neon_block_end" else None)
    spent = 0.0
    for row in (json.loads(line) for line in out.read_text("utf-8").splitlines() if line.strip()):  # as written
        spent += row_cost(row)
        if spent >= cap:
            assert row["recipient"] == crossing                                  # the call that reached the cap
            break
    else:
        pytest.fail("the cap was never reached")
    top = CONCURRENCY * max(cost.values())
    assert top == pytest.approx(priced["bound"]["Fable"])
    assert CONCURRENCY * cost[crossing] < got.cost_usd - cap <= top        # beyond its block's figure, within the bound


class _SlowCall:
    """A recipient backend for one slow call: it returns only once every other call of its block has returned and the
    other ``CONCURRENCY - 1`` slots hold calls of the next block, which return only after it; every call costs the
    per-item bound of its model. It records, in one sequence, when each call starts and returns (its row is written
    right after, with no judges, so the cap's spending follows the returns)."""

    def __init__(self, cost, slow_id, others, next_block):
        self.cost, self.slow_id, self.others, self.next_block = cost, slow_id, others, next_block
        self.events, self.next_started = [], 0
        self.others_done, self.next_ready, self.slow_done = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def chat(self, model, req):
        item = req.metadata["item"]
        self.events.append(("start", item, model.key))
        if item == self.slow_id:
            await self.others_done.wait()
            await self.next_ready.wait()
        elif model.key == self.next_block:
            self.next_started += 1
            if self.next_started == CONCURRENCY - 1:
                self.next_ready.set()
            await self.slow_done.wait()
        else:
            await asyncio.sleep(0)
        self.events.append(("finish", item, model.key))
        if item == self.slow_id:
            self.slow_done.set()
        elif model.key != self.next_block:
            self.others -= 1
            if self.others == 0:
                self.others_done.set()
        return ChatResult(text="I don't know.", usage={"cost": self.cost[model.key]}, finish_reason="stop")

    def stream_continuation(self, model, req):  # pragma: no cover - unused
        raise NotImplementedError


@pytest.mark.parametrize("case", ["sol_101st", "neon_601st"])
def test_what_has_started_bounds_the_overshoot_and_the_position_of_the_crossing_call_does_not(tmp_path, priced, case):
    """Check of 8cc83c5: section 9 qualified the bound by the position of the call that reaches the cap ("0.65 while
    the cap is reached before the last 7 calls of the first Sol block", "0.81 if the cap is reached before the 664th
    call"). A slow call keeps its slot while later calls start and finish, so the 101st call (a Sol call) can reach the
    cap with the first 7 Neon calls in flight, and the 601st (a Neon call) with the first 7 Fable calls: above the old
    figure. What has started bounds it: eight Neon items while no Fable call has started, eight Fable items
    otherwise. The real run_plan on windows of the real plan, every call at its model's per-item bound; section 10's
    row quotes the two overshoots."""
    plan, item, bound = priced["plan"], priced["item"], priced["bound"]
    cost = {model: item[name] for model, name in SHORT.items()}
    if case == "sol_101st":   # the 101st call, Sol, slow; Sol 102-439 finish, Neon 440-446 start and stay in flight
        lo, hi, slow, next_block, old = 90, 460, 100, "gemini-4-neon", bound["Sol"]
    else:                     # the 601st call, Neon, slow; Neon 602-670 finish, Fable 671-677 start and stay
        lo, hi, slow, next_block, old = 580, 690, 600, "claude-fable-5.2", bound["Neon"]
    crossing = plan[slow]["recipient"]
    assert plan[slow - 1]["recipient"] == plan[slow + 1]["recipient"] == crossing != next_block
    window = plan[lo:hi]
    slow_id = plan[slow]["id"]
    # every call of the crossing call's block but the slow one finishes first; the cap is crossed by the slow one
    cap = sum(cost[p["recipient"]] for i, p in enumerate(window, lo) if i != slow and p["recipient"] == crossing) + 0.01
    others = sum(1 for i, p in enumerate(window, lo) if i != slow and p["recipient"] == crossing)
    assert all(p["recipient"] in (crossing, next_block) for p in window)
    backend = _SlowCall(cost, slow_id, others, next_block)
    models = [ModelSpec(key=k, provider_id=f"x/{k}") for k in SHORT]
    got = asyncio.run(run_plan(backend, models, window, tmp_path / "keys.jsonl", events=priced["events"],
                               max_cost=cap, concurrency=CONCURRENCY, backend_name="scripted"))
    assert got.stopped_by == "max_cost"
    spent, started = 0.0, set()
    for kind, item_id, model in backend.events:            # the call whose row reaches the cap, and what had started
        if kind == "start":
            started.add(model)
            continue
        spent += cost[model]
        if spent >= cap:
            assert item_id == slow_id
            break
    over = got.cost_usd - cap
    assert next_block in started and over > old                                 # the old qualifier was no bound
    fable_started = "claude-fable-5.2" in started
    assert fable_started == (case == "neon_601st")
    assert over <= (bound["Fable"] if fable_started else bound["Neon"])          # what has started bounds it
    quoted = {"sol_101st": "a Sol call, the 101st, reaching the cap with 7 Neon calls in flight: "
                           f"{over:.2f} USD above it",
              "neon_601st": f"a Neon call, the 601st, with 7 Fable calls in flight: {over:.2f})"}[case]
    text = DESIGN.read_text("utf-8")
    table = text[text.index("### Refinements during the implementation review"):]
    row = next(line for line in table.splitlines() if line.startswith("| the position of the cap's overshoot |"))
    assert quoted in row


def test_the_two_letter_donor_names_are_counted_from_the_plan(priced):
    """The V1 counts of section 6, its section 10 row and the README (check of 8cc83c5: they were not pinned): the donor
    rows of donor_wrong and control whose donor name has a capitalised part of two letters, by name, and how many of
    them are full rows."""
    from cutoffmix.bench.keys.text import _short_tokens

    plan = priced["plan"]
    donor = [p for p in plan if p["set"] in ("donor_wrong", "control")]
    short = [p for p in donor if _short_tokens(p["donor_answer"])]
    by_name = {name: sum(1 for p in short if p["donor_answer"] == name) for name in sorted({p["donor_answer"]
                                                                                            for p in short})}
    full = sum(1 for p in short if p["arm"] == "full")
    assert (len(donor), len(short), full) == (106, 10, 5)
    assert by_name == {"Jiabo Ye": 4, "Kunchang Li": 2, "Xiaoyang Wu": 4}
    six = section(6)
    assert f"so \"Wu et al.\", \"Dr. Li\" and \"Ye's group\" mention Xiaoyang Wu, Kunchang Li and Jiabo Ye (the " \
        f"three donor names of the real plan with such a surname, {len(short)} of its {len(donor)} donor rows)" in six
    text = DESIGN.read_text("utf-8")
    table = text[text.index("### Refinements during the implementation review"):]
    row = next(line for line in table.splitlines() if line.startswith("| a two-letter surname alone |"))
    assert f"on the {len(short)} donor rows of Xiaoyang Wu, Kunchang Li and Jiabo Ye ({full} of them `full` rows " \
        "that 3b counts)" in row
    assert f"(Xiaoyang Wu, Kunchang Li, Jiabo Ye; {len(short)} der {len(donor)} Spender-Zeilen)" in readme()


def test_the_grading_scope_and_the_open_offers_have_their_figures(priced):
    """The figures of the grading scope and the open offers in sections 6 and 10 and the README (check of 8cc83c5): on
    the 136 right-name rows a correction with the surname alone, or the given name alone, stayed incorrect by rule
    with no judge asked (105 and 134); with the grading scope every one of them goes to the grading judges; the replay
    adds 15 grading-judge calls; and of the 57 stored answers the rules leave undecided none offers the expected
    answer (the one correct among them is a fake paper's abstention)."""
    from cutoffmix.bench.grading import grade_rule
    from cutoffmix.bench.keys.run import _rule_grade
    from cutoffmix.bench.keys.text import offers_answer

    plan, events, f = priced["plan"], priced["events"], priced["figures"]
    right = [p for p in plan if p["set"] in ("dark", "both_right") and p["arm"] == "full"]
    final = {}
    for part in ("surname", "given"):
        def text(p, part=part):
            name = p["donor_answer"].split()
            return f"It is not Sanjay Ramaswamy, it is {name[-1] if part == 'surname' else ' '.join(name[:-1])}."
        ruled = [_rule_grade(events[p["event_id"]], text(p)) for p in right]
        final[part] = sum(1 for r in ruled if r is not None and r[0] == "incorrect")     # without the scope: final
        assert all(_rule_grade(events[p["event_id"]], text(p), judges=True) is None for p in right)
    assert (len(right), final["surname"], final["given"]) == (136, 105, 134)
    six = section(6)
    assert f"(on the {len(right)} right-name rows of the real plan, {final['surname']} such corrections with the " \
        f"surname alone and {final['given']} with the given name alone)" in six
    total = priced["recipients"] + sum(f[k][2] for k in ("grading", "right", "adopt_full", "alone"))
    n, n_extra = f["grading"][0], f["extra"][0]
    assert f"in the replay of section 9, 3 `dark` items whose stored Sol answer names another person of the same " \
        f"surname (\"Yutong Mei\" for Lingrui Mei), {n_extra} calls" in six
    assert n_extra == 3 * len(ARMS_DARK) and {p["set"] for p in plan if _rule_grade(
        events[p["event_id"]], stored_answer(priced, p), judges=True) is None and _rule_grade(
        events[p["event_id"]], stored_answer(priced, p)) is not None} == {"dark"}
    # the stored answers the rules leave undecided: none offers the expected answer
    results = [r for r in load_results(RESULTS) if r.probe == "current_d"]
    undecided = [r for r in results if grade_rule(events[r.event_id], "current_d", r.response or "") is None]
    offering = [r for r in undecided if offers_answer(r.response or "", events[r.event_id])]
    correct = [r for r in undecided if r.grade == "correct"]
    assert (len(undecided), len(offering), len(correct)) == (57, 0, 1) and events[correct[0].event_id].control
    assert f"of the {len(undecided)} `current_d` answers the rules leave undecided, all of them `alone` answers, " \
        f"{len(correct)} is correct, a fake paper's abstention, and none offers the expected answer" in six
    text = DESIGN.read_text("utf-8")
    table = text[text.index("### Refinements during the implementation review"):]
    rows = {line.split(" | ")[0].removeprefix("| "): line for line in table.splitlines() if line.startswith("| ")}
    scope = rows["the grading scope"]
    assert f"(on the {len(right)} right-name rows {final['surname']} with the surname alone, {final['given']} with " \
        "the given name alone)" in scope
    assert f"in the replay {n_extra} more grading-judge calls ({n} in all: three `dark` items" in scope
    at_worst = sum(f[k][2] for k in ("grading", "right", "adopt_full", "alone"))
    assert f"the judges about {at_worst:.2f} USD at worst and the pilot about {total:.2f} USD with the recipients' " \
        f"estimate, {CAP - total:.2f} under the cap of 3.5 (under 3 about {3.0 - total:.3f} would be left)" in scope
    offers = rows["open offers"]
    assert f"of the {len(undecided)} stored answers the rules leave undecided none is an open offer" in offers
    assert "`offers_expected`" in offers and "the confound check of 1 alike" in offers
    text_de = readme()
    assert f"(bei {final['surname']} bzw. {final['given']} der {len(right)} richtigen Namen)" in text_de
    assert f"In der Nachrechnung kostet das {n_extra} Judge-Aufrufe mehr" in text_de
    assert f"Von den {len(undecided)} gespeicherten Antworten, die die Regeln offen lassen, bietet keine die " \
        "richtige Antwort an." in text_de
    assert "nach neun Prüfrunden" in text_de                                  # check of 8739803: nine rounds


ARMS_DARK = ("alone", "key", "full", "wrong_key", "blank")


def stored_answer(priced, p):
    return priced["stored"][(p["event_id"], p["recipient"])].response


def test_the_pilot_asks_for_a_cap_of_3_5_usd():
    """The judges' worst case left about 5 cents under the cap of 3 USD, so the pilot's command asks the owner to
    approve 3.5 USD (review of 863ca47), in the design and the README alike."""
    assert "`--judges gpt-6.2-sol --max-cost 3.5`, the cap the owner is asked to approve (the owner may choose " \
        "another cap" in section(9)
    assert "--out runs/keys_results.jsonl --judges gpt-6.2-sol --max-cost 3.5   # echte Modelle" in \
        README.read_text("utf-8")
    assert "`--max-cost 3`" not in DESIGN.read_text("utf-8") and "--max-cost 3 " not in README.read_text("utf-8")
    assert "the cap the owner is asked to approve is 3.5 dollars" in " ".join(DESIGN.read_text("utf-8")[:600].split())


def _keys_run_parser() -> argparse.ArgumentParser:
    parser = cutoffmix.cli.build_parser()
    for name in ("bench", "keys", "run"):
        sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
        parser = sub.choices[name]
    return parser


def test_the_exit_statuses_are_those_of_the_first_stop_everywhere():
    """Section 6 gives the status of the stop that first kept planned work from starting; the README, the module
    docstring and the epilog gave 130 after any Ctrl-C and 4 for any run with failed calls, and status 2 only before
    any call (review of the final review)."""
    epilog = " ".join((_keys_run_parser().epilog or "").split())
    assert epilog.startswith("exit status: that of the stop that first kept planned calls from starting: 3 --max-cost")
    assert "stops met at one check rank Ctrl-C, then the error, then the cap, and a later Ctrl-C or error keeps the " \
        "first stop's status" in epilog
    assert "2: refused before any call, or --out could not take a row during the run" in epilog
    doc = " ".join((cutoffmix.cli.__doc__ or "").split())
    assert "Its exit status is that of the stop that first kept planned calls from starting" in doc
    assert "or a results file that cannot take a row during the run" in doc
    assert "the first Ctrl-C lets the calls in flight finish and exits with status 130" not in doc
    text_de = readme()
    assert "Der Exit-Code ist der des Halts, der als erster geplante Aufrufe am Start hinderte" in text_de
    assert "gilt Strg-C vor dem Fehler vor dem Deckel; ein späteres Strg-C oder ein späterer Fehler ändert den Code " \
        "nicht" in text_de
    assert "Exit-Code 2 heißt: Eingaben oder Optionen abgewiesen, vor jedem Aufruf, oder die Ergebnisdatei nahm " \
        "während des Laufs keine Zeile mehr an" in text_de
    assert "und mit 130 nach Strg-C, wobei" not in text_de
    assert "Status 2 also ends a run whose results file cannot take a row once calls were sent" in section(6)


def test_the_judges_scope_is_stated_for_each_name():
    """With --judges the judges decide every labelled answer that names the candidate, the right name and the donor's
    name alike, and the rules only the donor rows that do not name it (review of 863ca47: the rules decided by rule
    phrasings that four review rounds kept finding). The README, the --judges help and the design say so; the rules'
    own decision without judges is still stated for each name."""
    text_de = readme()
    assert "Läuft der Judge mit (`--judges`), entscheidet er jede bewertete Antwort, die den Namen nennt, in welcher " \
        "Form auch immer" in text_de
    assert "Die Regeln entscheiden allein nur noch die Antworten, die den Namen gar nicht nennen (nicht übernommen)." \
        in text_de
    assert "Beim falschen Namen des Spenders (3b) gilt eine Antwort nach den Regeln als übernommen, wenn sie sich " \
        "den Namen zu eigen macht" in text_de
    judges = next(a for a in _keys_run_parser()._actions if "--judges" in a.option_strings)
    help_text = " ".join((judges.help or "").split())
    assert "every labelled answer that names the candidate in any form, whatever the rules would decide" in help_text
    # the grading scope (check of 8cc83c5): the grade too, wherever the rules would grade against the named answer
    assert "for every rule grade incorrect or abstain of an answer that names the expected answer in any form (every " \
        "arm and set)" in help_text
    assert "the rules decide alone only a grade correct and the answers that do not name the expected answer, and " \
        "for adoption the answers that do not name the donor's name" in help_text
    assert "the report gives no verdict that the open offers, ungraded answers that name the expected answer, could " \
        "overturn" in help_text
    # review of b86d67f: a reply without a verdict leaves the answer undecided too, and the run counts such answers
    assert "A failed or tied vote, or a reply without a verdict, leaves the answer undecided, never to the rules " \
        "(the run prints how many are)" in help_text
    assert "for the right name also another spelling or a doubt" not in help_text
    six = section(6)
    assert "**The judges' scope** (with `--judges`). The adoption judges decide every labelled answer that mentions " \
        "the name it is measured against, whatever the rules would decide" in six
    assert "`mentions(text, name, event)` reads widely" in six and "rule_adopted" in six
    seven = section(7)
    assert "The gates, the 20 % limit and the checks against the undecided rows and the open offers are the same " \
        "whoever decides the rows." in seven
    assert "an answered row is undecided, and an open offer, only when the judges' votes failed, tied or gave no " \
        "verdict" in seven
    # the grading scope (check of 8cc83c5): the same principle for the grade
    assert "**The grading scope** (with `--judges`). The principle of the judges' scope (next point) holds for the " \
        "grade as well: the grading judges decide every answer that names the expected answer wherever the rules " \
        "would grade it against that answer" in six
    assert "what they leave open stays undecided for the report's gates" not in six


def test_section_10_records_the_judges_scope_and_its_refinements(priced):
    """The check of 863ca47 added four rows to the refinements table: the judges' scope, the rules without judges, the
    3b effect of the wider note references (the row on the note in more words named only the right name) and the
    cap's overshoot; the intro counts the rows each review added. The review of aafe7f9 added four more (a two-letter
    surname alone, a correction of another name, a judges' decision on resume, the rules-versus-judges table) and
    corrected two of the check's rows in place: the dash clause and the further references to the note in "the rules
    without judges", and the bound in "the cap's overshoot", whose figures are pinned here. The check of 8cc83c5 added
    the last three (the grading scope, open offers, the position of the cap's overshoot) and corrected two rows in
    place: the correction's claim about 3a, and the overshoot's qualifiers. The review of b86d67f added two (an open
    offer in any form, answers the judges left undecided), the check of 8739803 the last (another person's name)."""
    text = DESIGN.read_text("utf-8")
    table = text[text.index("### Refinements during the implementation review"):]
    titles = [line.split(" | ")[0].removeprefix("| ") for line in table.splitlines() if line.startswith("| ")][1:]
    check = ["the judges' scope", "the rules without judges", "the wider note references on the donor's name",
             "the cap's overshoot"]
    review = ["a two-letter surname alone", "a correction of another name, on every arm",
              "a judges' decision on resume", "the rules-versus-judges table"]
    latest = ["the grading scope", "open offers", "the position of the cap's overshoot"]
    newest = ["an open offer in any form", "answers the judges left undecided"]          # review of b86d67f
    person = ["another person's name"]                                                  # check of 8739803
    assert titles[-14:] == check + review + latest + newest + person
    seven = titles[titles.index("another role or a rejection in other words"):titles.index("the judges' scope")]
    assert len(seven) == 7 and seven[-1] == "the pilot's total and the cap"
    intro = " ".join(table[: table.index("| issue | before | now |")].split())
    assert "added the seven rows from \"another role or a rejection in other words\" to \"the pilot's total and the " \
        "cap\"" in intro
    assert "added the four rows from \"the judges' scope\" to \"the cap's overshoot\"" in intro
    assert "A review of that change added the four rows from \"a two-letter surname alone\" to \"the " \
        "rules-versus-judges table\"" in intro
    assert "corrected two rows of the check in place: \"the rules without judges\"" in intro
    assert "A check of that review's changes added the three rows from \"the grading scope\" to \"the position of " \
        "the cap's overshoot\"" in intro
    assert "A review of that check's changes added the two rows \"an open offer in any form\" and \"answers " \
        "the judges left undecided\"" in intro
    assert "It corrected two rows of the check in place: \"the grading scope\" (a second run with the same " \
        "`--judges` sends nothing only once the judges have decided every row in their scope) and \"open offers\" " \
        "(what `offers_expected` holds)" in intro
    assert "A check of that review's changes added the last row, \"another person's name\": a name the paper grader " \
        "cannot place that writes the expected answer with a title or a group word, or in a near spelling the paper " \
        "rules leave to the judges as a possible misspelling, is no other person's" in intro
    assert "It corrected two rows in place as well: \"a correction of another name, on every arm\"" in intro
    assert "and \"the cap's overshoot\" (its qualifiers)" in intro
    rows = {line.split(" | ")[0].removeprefix("| "): line for line in table.splitlines() if line.startswith("| ")}
    assert "\"Supposedly X.\"" in rows["the wider note references on the donor's name"]
    assert "adopted by rule on `donor_wrong` and `control`" in rows["the wider note references on the donor's name"]
    assert "`rule_adopted`" in rows["the judges' scope"] and "`mentions`" in rows["the judges' scope"]
    without = rows["the rules without judges"]
    assert "the plan is unchanged byte for byte" in without
    assert "a comma or dash clause that asserts anew" in without and "\"It's not Y - it's X.\"" in without
    for form in ("\"the summary\"", "\"you've provided\"", "\"you share\"", "\"as mentioned\" (indicated, shown,"):
        assert form in without, form
    assert "an answer's own citations, \"as shown on the arXiv listing\", \"the paper's summary\"" in without
    # the cap's overshoot: the figures of section 9 (review of aafe7f9: the row's figures were not pinned)
    over = rows["the cap's overshoot"]
    bound, f = priced["bound"], priced["figures"]
    # the row's room under 3 USD is the one its review computed: the grading judges' calls before the grading scope
    earlier = priced["recipients"] + sum(f[k][2] for k in ("before", "right", "adopt_full", "alone"))
    assert "0.23 USD is an estimate from the stored maxima" in over
    assert f"(4096 per call, {BUDGET} per judge call)" in over
    assert f"up to about {max(bound.values()):.2f} USD above the cap at the default of {CONCURRENCY} (eight Fable " \
        f"items: the items in flight can belong to a later block, so eight Sol items, {bound['Sol']:.2f}, bound it " \
        f"only while no Neon call has started, and eight Neon items, {bound['Neon']:.2f}, only while no Fable call " \
        "has)" in over
    assert f"the pilot's cap is 3.5 USD (under 3 the judges at their worst left about {3.0 - earlier:.2f} USD)" in over
    assert "in a Sol block, 0.81 in a Neon block" not in over and "664" not in over
    # the position of the overshoot (check of 8cc83c5): the old qualifiers quoted, the new bounds by what has started
    position = rows["the position of the cap's overshoot"]
    later, top = CONCURRENCY - 1, max(bound.values())
    old_fable = priced["first_fable_call"] - CONCURRENCY + 1                    # where a Fable call could be in flight
    assert f"(\"{bound['Sol']:.2f} while the cap is reached before the last {later} calls of the first Sol block, " \
        f"{bound['Neon']:.2f} before the last {later} calls of the Neon {priced['blocks'][1][1]} block\", \"from the " \
        f"{old_fable}th of the {priced['calls']} calls on a Fable call can be in flight\", " \
        f"\"{bound['Neon']:.2f} if the cap is reached before the {old_fable}th call\")" in position
    assert f"at the default of {CONCURRENCY} the run ends at most about {bound['Sol']:.2f} USD above the cap when " \
        f"the cap is reached before the first Neon call has started (the {priced['first_neon_call']}th), " \
        f"{bound['Neon']:.2f} before the first Fable call has started (the {priced['first_fable_call']}st), and " \
        f"{top:.2f} otherwise" in position
    added = {k: rows[k] for k in review}
    assert "`mentions` found a part of the name on its own only with three letters or more" in \
        added["a two-letter surname alone"]
    correction = added["a correction of another name, on every arm"]
    assert "`offers_answer`" in correction and f"the replay's {f['before'][0]} grading-judge calls stay" in correction
    assert "(on `full` alone it would favour 3a; on every arm too, without judges, as long as an ungraded `key` row " \
        "counted as a miss in 3a: the row \"open offers\" ends that)" in correction
    assert "`adoption_name`" in added["a judges' decision on resume"]
    assert "a column of their own" in added["the rules-versus-judges table"]


# ---- review of b86d67f ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mock_judged(tmp_path_factory, priced):
    """A --mock run of the real plan with --judges through the CLI, then the same command again (section 8): the mock's
    judges give no verdict, so every row they are asked about stays undecided and the second run asks again."""
    from cutoffmix.bench.io import read_jsonl, write_jsonl
    from cutoffmix.bench.keys import build_report

    tmp = tmp_path_factory.mktemp("mock_judged")
    plan_file, out = tmp / "plan.jsonl", tmp / "results.jsonl"
    write_jsonl(plan_file, priced["plan"])
    argv = ["bench", "keys", "run", "--plan", str(plan_file), "--events", str(EVENTS), "--out", str(out), "--mock",
            "--judges", JUDGE]
    assert cutoffmix.cli.main(argv) == 0
    first = read_jsonl(out)
    assert cutoffmix.cli.main(argv) == 0
    rows = read_jsonl(out)
    last = {r["id"]: r for r in rows}
    return {"first": first, "again": len(rows) - len(first), "last": last,
            "report": build_report(priced["plan"], list(last.values()))}


def test_the_mock_run_with_judges_has_the_figures_of_section_8(priced, mock_judged):
    """Section 8's figures (review of b86d67f: none of them was pinned): the one row whose rule grade is in the grading
    scope, the open offers of the full arms, and the answers the judges' votes left undecided, which the same command
    asks about again (a second run with --judges sends nothing only once the judges decided every row)."""
    from cutoffmix.bench.keys.common import judges_left_open
    from cutoffmix.bench.keys.run import _rule_grade

    events, first = priced["events"], mock_judged["first"]
    scoped = [r for r in first if _rule_grade(events[r["event_id"]], r["response"], judges=True) is None
              and _rule_grade(events[r["event_id"]], r["response"]) is not None]
    assert [(r["set"], r["arm"], r["response"]) for r in scoped] == [("donor_wrong", "full", "Kunchang Li.")]
    assert scoped[0]["offers_expected"] is False and events[scoped[0]["event_id"]].expected_current == "Xinhao Li"
    totals = mock_judged["report"]["totals"]
    full = sum(1 for r in first
               if r["arm"] == "full" and r["set"] in ("dark", "both_right") and r["grade"] == "ungraded")
    undecided = sum(1 for r in first if judges_left_open(r))
    assert totals["open_offers"] == full and totals["judges_undecided"] == undecided == mock_judged["again"]
    eight = section(8)
    assert "(one row in the mock run of the real plan: the donor's \"Kunchang Li.\" beside the right answer Xinhao " \
        "Li, which names Xinhao Li only inside another person's name and so is no open offer)" in eight
    assert f"the undecided right names are open offers of the `full` arms ({full} in that run)" in eight
    assert f"the answers the judges' votes left undecided ({undecided} in that run; the same command run again asks " \
        "about them again)" in eight
    six = section(6)
    assert "Once the judges have decided every row in their scope, a second run with the same `--judges` sends " \
        "nothing; a failed or tied vote, or a reply without a verdict, is asked again by every later run with " \
        "`--judges` (and paid again)." in six
    assert "and a second run with the same `--judges` sends nothing. A stored" not in six
    text = DESIGN.read_text("utf-8")
    table = text[text.index("### Refinements during the implementation review"):]
    scope = next(line for line in table.splitlines() if line.startswith("| the grading scope |"))
    assert "once the judges have decided every row in their scope a second run with the same `--judges` sends " \
        "nothing (a failed or tied vote, or a reply without a verdict, is asked again)" in scope
    assert "a grade of the grading judges stands, and a second run with the same `--judges` sends nothing;" \
        not in scope


def test_the_figures_of_sections_9_and_10_that_follow_from_the_plan_and_the_stored_data(priced):
    """Figures that no test pinned (review of b86d67f): the items in flight with a slow Sol call (one slot fewer than
    --concurrency), the dark key rows of the open-offer row, its quote of the stored undecided answers, the README's
    count of items the grading scope adds; and that the grading scope adds no second call to an item."""
    from cutoffmix.bench.grading import grade_rule

    nine = section(9)
    assert f"(a slow Sol call, the 101st, can reach the cap while the first {CONCURRENCY - 1} Neon calls are in " \
        "flight)" in nine
    extra = priced["figures"]["extra"][0]
    assert "so the grading scope adds no second call to an item (the per-item bound stands); it asks the grading " \
        f"judges about items they were not asked about before ({extra} in the replay, below)" in nine
    assert "so the grading scope adds no call to an item." not in nine
    plan, events = priced["plan"], priced["events"]
    dark_key = sum(1 for p in plan if p["set"] == "dark" and p["arm"] == "key")
    text = DESIGN.read_text("utf-8")
    table = text[text.index("### Refinements during the implementation review"):]
    offers = next(line for line in table.splitlines() if line.startswith("| open offers |"))
    assert dark_key == 116 and f"3a read supported on {dark_key} such `dark` key rows" in offers
    results = [r for r in load_results(RESULTS) if r.probe == "current_d"]
    undecided = [r for r in results if grade_rule(events[r.event_id], "current_d", r.response or "") is None]
    correct = sum(1 for r in undecided if r.grade == "correct")
    assert f"(stored results: {correct} correct answer among {len(undecided)} undecided ones)" in offers
    # the README's three items: the distinct events among the grading scope's extra calls of the replay
    from cutoffmix.bench.keys.run import _rule_grade
    added = {p["event_id"] for p in plan if _rule_grade(events[p["event_id"]], stored_answer(priced, p), judges=True)
             is None and _rule_grade(events[p["event_id"]], stored_answer(priced, p)) is not None}
    words = {2: "zwei", 3: "drei", 4: "vier", 5: "fünf"}
    assert f"In der Nachrechnung kostet das {extra} Judge-Aufrufe mehr ({words[len(added)]} Fragen, bei denen Sol " \
        "eine andere Person desselben Nachnamens nennt)" in readme()


@pytest.fixture(scope="module")
def failing_replay(tmp_path_factory, priced):
    """The offline replay of section 9 through the real run_plan with --judges, every judge call failing."""
    from cutoffmix.backends.base import BackendError
    from cutoffmix.bench.keys import build_report

    class Stored:
        async def chat(self, model, req):
            if req.metadata.get("purpose"):
                raise BackendError("judge down", status=500)
            p = {"event_id": req.metadata["event_id"], "recipient": model.key}
            return ChatResult(text=stored_answer(priced, p), usage={"cost": 0.0}, finish_reason="stop")

        def stream_continuation(self, model, req):  # pragma: no cover - unused
            raise NotImplementedError
    models = [ModelSpec(key=k, provider_id=f"x/{k}") for k in SHORT]
    out = tmp_path_factory.mktemp("failing_replay") / "keys.jsonl"
    got = asyncio.run(run_plan(Stored(), models, priced["plan"], out, events=priced["events"],
                               judges=[ModelSpec(key=JUDGE, provider_id="judge/x")], concurrency=16,
                               backend_name="scripted"))
    return build_report(priced["plan"], got.rows)


def test_an_open_offer_is_an_ungraded_answer_that_names_the_expected_answer(priced, failing_replay):
    """Review of b86d67f: an open offer was an answer offers_answer reads as offering the expected answer, so a lost
    grading vote on "Dr. Zhang", "Zhang et al.", a given name or a common surname alone, or "X?" counted as a miss with
    no check, and without judges so did such an answer the paper rules cannot grade. The docs state the wider rule,
    the figures of its row, and how a row that does not name the expected answer counts in each prediction; the
    stored replay keeps its open offers."""
    from cutoffmix.bench.grading import grade_rule

    events = [priced["events"][e] for e in sorted({p["event_id"] for p in priced["plan"]})]
    real = [e for e in events if not e.control]
    open_et_al = sum(1 for e in real if grade_rule(e, "current_d", f"The first author is "
                                                                   f"{e.expected_current.split()[-1]} et al.") is None)
    open_question = sum(1 for e in real if grade_rule(e, "current_d", f"I'm not sure. {e.expected_current}?") is None)
    text = DESIGN.read_text("utf-8")
    table = text[text.index("### Refinements during the implementation review"):]
    row = next(line for line in table.splitlines() if line.startswith("| an open offer in any form |"))
    assert (len(real), open_et_al, open_question) == (107, 81, 107)
    assert f"(without judges the paper rules leave \"Zhang et al.\" open on {open_et_al} of the {len(real)} events " \
        f"with an expected answer and the question on all {open_question})" in row
    kept = failing_replay["totals"]["open_offers"]
    assert f"the stored replay keeps its open offers ({kept} with every judge call failing)" in row and kept == 20
    assert "`offers_expected` is `offers_answer` or `names_answer`" in row
    undecided = next(line for line in table.splitlines() if line.startswith("| answers the judges left undecided |"))
    assert "the exit statuses are unchanged" in undecided
    six, seven = section(6), section(7)
    assert "`names_answer(text, event)` (the open offers, primary analysis below): `mentions_answer` of the answer " \
        "once every name in it that the paper grader cannot place, and every distractor of the event, is blanked " \
        "where it writes another person" in six
    assert "It does not hold for an **open offer**: an answered row that is `ungraded` and whose answer names the " \
        "expected answer (`offers_expected` true:" in six
    # an ungraded row that does not name the expected answer: close to exact, and counted the way each prediction
    # reads it (the review of b86d67f: section 6 called it conservative)
    assert "close to exact rather than conservative: in 1, 1b and 2 an ungraded `key` row counts against the key and " \
        "one of the other arm for it, in 3a an ungraded `key` row counts for `full`, and in 4 an ungraded `key` row " \
        "counts as harm and an ungraded `alone` row against it" in six
    assert "(conservative), those of the other arm" not in six
    assert "Without `--judges` the rules decide as above, and the labels they leave open are undecided: the gates " \
        "and checks of 3a and 3b (section 7) keep any verdict from resting on them." in six
    assert "the report's gates and its check of the open offers (below, and section 7) keep any verdict" not in six
    assert "An open offer is an answered `ungraded` row whose answer names the expected answer (`offers_expected`, " \
        "section 6, primary analysis: it offers it, or names it in any other form other than inside another " \
        "person's name)" in seven
    assert "while an ungraded `key` row that does not name it, or names it only inside another person's name, counts " \
        "as not correct and so for `full`, as in the primary analysis" in seven
    text_de = readme()
    assert "Kein Urteil stützt sich auf unbewertete Antworten, die die richtige Antwort nennen, auch mit Titel " \
        "(„Professor Guo“) oder im vollen Namen leicht verschrieben („Daya Guoo“), ob die Regeln sie offen lassen " \
        "oder die Stimme des Judge ausfiel; nur eine Nennung im Namen einer anderen Person zählt nicht („Yutong Mei“ " \
        "bei Lingrui Mei)." in text_de                                        # check of 8739803: those forms too
    assert "obwohl sie die richtige Antwort anbieten. Der Pilot" not in text_de
    assert "unbewertete Antworten, die die richtige Antwort nennen (unten)" in text_de
    assert "Jetzt ist jede unbewertete Antwort, die die richtige Antwort in irgendeiner Form nennt, ein offenes " \
        "Angebot, außer sie nennt sie nur im Namen einer anderen Person („Yutong Mei“ bei Lingrui Mei)" in text_de
    doc = " ".join((cutoffmix.cli.__doc__ or "").split())
    assert "no verdict stands that the open offers (ungraded answers that name the expected answer) could overturn" \
        in doc


def test_with_judges_a_rule_grade_correct_stays_the_rules(priced):
    """Review of b86d67f: the README's status line and the CLI's module docs said that with --judges the judge
    decides every answer that names the name, the grade as well; a rule grade correct stays the rules' decision, as
    section 6 says, and so does the grade of an answer that does not name the expected answer."""
    from cutoffmix.bench.keys.run import _rule_grade

    event = priced["events"]["paper:2507.13334"]
    assert _rule_grade(event, "The first author is Lingrui Mei.", judges=True) == ("correct", False)
    assert "A rule grade `correct` stays a rule decision" in section(6)
    text_de = readme()
    assert "entscheidet mit `--judges` der Judge bei der Übernahme jede Antwort, die den Namen nennt, und bei der " \
        "Bewertung jede, die ihn nennt und die die Regeln sonst als falsch oder als Enthaltung werten würden; die " \
        "festen Regeln entscheiden nur noch „richtig“, die Antworten ohne den Namen und den Lauf ohne Judge." in text_de
    assert "bei der Bewertung wie bei der Übernahme" not in text_de
    doc = " ".join((cutoffmix.cli.__doc__ or "").split())
    assert "the rules only a grade correct and the answers that do not name them" in doc
    assert "the rules only the answers that do not name them" not in doc


def test_the_hedge_of_the_grading_scope_has_a_given_name(priced):
    """Review of b86d67f: section 6 gave "I think it's X, but I'm not sure." (X the full name) as an abstention the
    rules decided; with the full name the paper rules grade it correct, with the given name alone abstain."""
    from cutoffmix.bench.grading import grade_rule
    from cutoffmix.bench.keys.run import _judges_grade, _rule_grade

    event = priced["events"]["paper:2507.13334"]
    full, given = "I think it's Lingrui Mei, but I'm not sure.", "I think it's Lingrui, but I'm not sure."
    assert grade_rule(event, "current_d", full)[0] == "correct" and _rule_grade(event, full, judges=True)[0] == \
        "correct"
    assert grade_rule(event, "current_d", given)[0] == "abstain" and _rule_grade(event, given, judges=True) is None
    assert "\"I think it's Lingrui, but I'm not sure.\" with the given name only (`abstain` by the paper rules, " \
        "which grade the same answer with the full name `correct`)" in section(6)
    assert "\"I think it's X, but I'm not sure.\" (`abstain`" not in " ".join(DESIGN.read_text("utf-8").split())
    assert "\"I think it's <given name>, but I'm not sure.\"" in " ".join((_judges_grade.__doc__ or "").split())


def test_the_readme_plans_with_the_pre_registered_date(priced):
    """Review of b86d67f (pre-existing): the README's plan command had no --today, so a plan made on the pilot day
    differs from the pre-registered one in every system prompt (the ids and the estimate stay)."""
    events = load_events(EVENTS)
    later, _ = build_plan(events, load_results(RESULTS), today="2026-11-28")
    plan = priced["plan"]
    assert [p["id"] for p in later] == [p["id"] for p in plan]
    assert sum(a["messages"] != b["messages"] for a, b in zip(later, plan, strict=True)) == len(plan) == 746
    assert TODAY == "2026-11-27" and f"The version 2 plan (seed 0, `--today {TODAY}`) has **746 calls**" in section(9)
    text_de = readme()
    assert f"cutoffmix bench keys plan --events data/events/papers_events.jsonl --today {TODAY} \\ --results " \
        "data/results/papers_probes_2026-11-25.jsonl --out runs/keys_plan.jsonl" in text_de
    assert f"das Datum des vorab festgelegten Plans, `--today {TODAY}` (ohne die Option nimmt `plan` das heutige " \
        f"Datum, und alle {len(plan)} Prompts wären andere)" in text_de


# ---- check of 8739803 ----------------------------------------------------------------------------------------------


def test_another_persons_name_has_its_figures(priced, failing_replay, mock_judged):
    """The check of 8739803: a name the paper grader cannot place that writes the expected answer with a title or a
    group word, or in a near spelling the paper rules leave to the judges, was another person's, so a lost vote (or,
    without judges, a near spelling the rules leave open) counted as a miss with no check. The figures of its row in
    section 10 and of the README follow from the plan, the paper rules and the stored data; the stored replay and the
    mock run keep their open offers; section 6 lists the words the code drops and its similarity is the grader's."""
    import difflib
    import re

    from cutoffmix.bench.grading import _NEAR_NAME_RATIO, grade_rule
    from cutoffmix.bench.keys.run import _offers_expected, _rule_grade
    from cutoffmix.bench.keys.text import _HONORIFICS, names_answer

    plan, events = priced["plan"], priced["events"]
    daya = sum(1 for p in plan
               if p["set"] == "both_right" and p["arm"] == "key" and p["event_id"] == "paper:2401.14196")
    both_key = sum(1 for p in plan if p["set"] == "both_right" and p["arm"] == "key")
    dark_key = [p for p in plan if p["set"] == "dark" and p["arm"] == "key"]
    titled = sum(1 for p in dark_key if _rule_grade(events[p["event_id"]], "The first author is Professor "
                                                    f"{events[p['event_id']].expected_current.split()[-1]}.",
                                                    judges=True) is None)    # to the grading judges: lost votes
    assert (daya, both_key, len(dark_key), titled) == (2, 20, 116, 92)
    kept, mock_open = failing_replay["totals"]["open_offers"], mock_judged["report"]["totals"]["open_offers"]
    results = [r for r in load_results(RESULTS) if r.probe == "current_d"]
    undecided = [r for r in results if grade_rule(events[r.event_id], "current_d", r.response or "") is None]
    assert not any(_offers_expected(events[r.event_id], r.response or "") for r in undecided)
    text = DESIGN.read_text("utf-8")
    table = text[text.index("### Refinements during the implementation review"):]
    row = next(line for line in table.splitlines() if line.startswith("| another person's name |"))
    assert f"{daya} lost votes on `both_right` key rows \"The first author is Professor Guo.\" made 4 not supported " \
        "with 0 open offers" in row
    assert f"with every grading vote lost {titled} `dark` key rows \"The first author is Professor <surname>.\" made " \
        f"3a supported (b = {titled}, c = 0) with 0 open offers" in row
    assert f"and without judges {len(dark_key)} `dark` key rows that double the surname's last letter made 3a " \
        "supported with 0 open offers" in row
    assert f"the three scenarios read not testable, with {daya}, {titled} and {len(dark_key)} open offers" in row
    assert f"the stored replay keeps its {kept} open offers with every judge call failing and its verdicts, the mock " \
        f"runs write the same rows ({mock_open} open offers with judges), and of the {len(undecided)} stored answers " \
        "the rules leave undecided none is an open offer" in row and (kept, mock_open) == (20, 126)
    assert f"a possible misspelling by the paper grader's similarity of at least {_NEAR_NAME_RATIO} or by the same " \
        "given name with the same first four surname letters, or by its test of initials one slip away (\"Ziyo Ma\" " \
        "for Ziyu Ma)" in row
    assert "`_same_person` and the adoption rules are unchanged" in row
    six, seven = section(6), section(7)
    assert f"(`grading._near_name`) by its letters-only similarity of at least {_NEAR_NAME_RATIO} (\"Daya Guoo\", " \
        "\"Day Guo\") or by the same given name with the same first four surname letters (\"Matthew Kowalewski\" " \
        "for Matthew Kowal), or by its test of initials one slip away" in six
    from cutoffmix.bench.grading import _compact, _near_name
    for name, form, similar in (("Daya Guoo", "Daya Guo", True), ("Day Guo", "Daya Guo", True),
                                ("Matthew Kowalewski", "Matthew Kowal", False), ("Ziyo Ma", "Ziyu Ma", False)):
        ratio = difflib.SequenceMatcher(None, _compact(name), _compact(form)).ratio()
        assert _near_name(name, form) and (ratio >= _NEAR_NAME_RATIO) is similar, name
    assert "A name that is not a distractor must also pass these tests once its honorific, title and group words are " \
        "dropped, and be no near spelling of the expected answer or of an accepted form of two words or more" in six
    listed = six[six.index("The words: ") + len("The words: "): six.index(", compared on their letters")]
    words = set()
    for word in re.split(r",\s*(?:and the German\s+|and\s+)?|\s+and\s+", listed):
        base, _, tail = word.strip().partition("(")
        words |= {base, base + tail.rstrip(")")} if tail else {base}
    assert words == set(_HONORIFICS)
    assert "Without `--judges` that one difference is all, a known limit of that path: a rule grade `incorrect` of " \
        "an answer that names the expected answer in a form `offers_answer` does not accept stays final" in six
    event = events["paper:2401.14196"]
    assert grade_rule(event, "current_d", "It is not Sanjay Ramaswamy, it is Dr. Guo.")[0] == "incorrect"
    assert grade_rule(event, "current_d", "The first author is Professor Guo.")[0] == "incorrect"
    assert grade_rule(event, "current_d", "The first author is Dr. Guoo.") is None and \
        grade_rule(event, "current_d", "I am not sure, maybe Dr. Guoo.")[0] == "abstain"
    assert not names_answer("The first author is Dr. Guoo.", event)
    assert "the paper rules leave \"The first author is Dr. Guoo.\" to the judges and grade \"I am not sure, maybe " \
        "Dr. Guoo.\" `abstain`, and an ungraded row with such an answer is no open offer" in six
    assert "where a name the paper grader cannot place that writes the expected answer with a title or a group word, " \
        "or in a near spelling the paper rules leave to the judges as a possible misspelling, is no other person's " \
        "(section 6, `names_answer`)" in six
    assert "A name the paper grader cannot place that writes the expected answer with a title or a group word " \
        "(\"Professor Guo\", \"the Guo Group\"), or in a near spelling the paper rules leave to the judges as a " \
        "possible misspelling (\"Daya Guoo\"), is no other person's (section 6, `names_answer`)." in seven
    text_de = readme()
    numbers = {2: "Zwei", 3: "Drei", 4: "Vier"}
    assert f"{numbers[daya]} ausgefallene Stimmen bei „Der Erstautor ist Professor Guo.“ unter den {both_key} " \
        "`key`-Antworten von `both_right` machten Vorhersage 4 zu „nicht gestützt“" in text_de
    assert f"ohne Judge machten {len(dark_key)} `key`-Antworten bei `dark`, deren Nachname einen verdoppelten " \
        "letzten Buchstaben hat, Vorhersage 3a zu „gestützt“, beide Male ohne ein einziges offenes Angebot" in text_de
    assert "Ein anderer Vorname mit demselben Nachnamen bleibt eine andere Person („Zhaoyang Li“ bei Zhiyu Li)" in \
        text_de
    assert "Ein verschriebener Nachname allein („Dr. Guoo“) ist dagegen für die Bewertung kein Name und zählt weiter " \
        "nicht als Nennung der richtigen Antwort." in text_de
    assert "Ohne Judge bleibt eine bekannte Grenze: Ein „falsch“ der Regeln für eine Antwort, die die richtige " \
        "Antwort in einer Form nennt, die die Regel von `offers` nicht annimmt („Es ist nicht Y, es ist Dr. " \
        f"Zhang.“), bleibt endgültig; der Pilot läuft mit Judge. Bei den gespeicherten Antworten ändert sich nichts. " \
        f"Die {len(plan)} geplanten Aufrufe sind dieselben geblieben." in text_de and len(plan) == 746
