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
the README's block order, section 10's overshoot row and both judges' budget are pinned too."""

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

    # graded as the run grades (run._rule_grade: a rule grade incorrect next to the expected answer is unsure too)
    grading = [p for p in plan if _rule_grade(by_id[p["event_id"]], answer(p)) is None]
    right = [p for p in plan if p["set"] in ("dark", "both_right") and p["arm"] == "full"]
    adopt_full = [p for p in plan if p["set"] in ("donor_wrong", "control") and p["arm"] == "full"]
    alone_due = []  # the alone rows of the replay that mention the donor's name: the judges decide them
    for p in (p for p in plan if p["set"] in ("donor_wrong", "control") and p["arm"] == "alone"):
        ruled = _rule_grade(by_id[p["event_id"]], answer(p))
        _, label = _adoption_rule(p, by_id[p["event_id"]], answer(p), ruled[0] if ruled else "ungraded")
        if _judged(p, by_id[p["event_id"]], answer(p), label):
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
            "fable_from": first_fable - CONCURRENCY + 2, "neon_from": first_neon - CONCURRENCY + 2,
            "plan": plan, "events": by_id}


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
        f"{3.0 - total:.2f} USD" in text
    assert "USD at worst: under the pilot's cap" not in text
    # the README states the same figures (in German, decimal comma)
    text_de = readme()
    assert f"bis etwa {comma(every)} $ für den Judge (im ungünstigsten Fall {comma(at_worst)} $; zusammen " \
        f"geschätzt {comma(total)} $, eine Schätzung, keine Obergrenze: die setzt `--max-cost`, freigegeben werden " \
        "sollen 3,5 $)" in text_de
    n_right, n_adopt, n_alone = f["right"][0], f["adopt_full"][0], f["alone"][0]
    cents = round(100 * (CAP - total))
    assert f"etwa {comma(f['grading'][1])} $ für die Antworten, die die Regeln nicht bewerten können, und, da er " \
        f"jede Antwort mit dem Namen entscheidet, bis etwa {comma(every)} $ (bis zu {n_right} Aufrufe für den " \
        f"richtigen Namen, {n_adopt} für den falschen in den `full`-Antworten, in der Nachrechnung {n_alone} beim " \
        f"erneuten Fragen), im ungünstigsten Fall etwa {comma(at_worst)} $. Mit der Schätzung für die Empfänger " \
        f"sind das etwa {comma(total)} $, also etwa {cents} Cent unter `--max-cost 3.5`. Unter dem früheren Deckel " \
        f"von 3 $ blieben nur etwa {round(100 * (3.0 - total))} Cent" in text_de
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
    of the dearest item that can be in flight, eight Fable items from the call on that can have a Fable call in flight
    with it (review of aafe7f9: the figures per block failed there, also for the last 4 Sol calls)."""
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
    # the bound: eight of the dearest item, Fable's; eight Sol or Neon items only away from a dearer block
    top = max(bound.values())
    assert top == bound["Fable"] and bound["Sol"] < bound["Neon"] < top
    calls, later = priced["calls"], CONCURRENCY - 1
    assert f"So the run ends at most about {top:.2f} USD above the cap at the default of {CONCURRENCY} (eight Fable " \
        "items; a lower `--concurrency` lowers this in proportion)" in text
    assert f"Eight Sol items cost at most about {bound['Sol']:.2f} USD and eight Neon items {bound['Neon']:.2f}, but " \
        f"those hold only while no dearer call can be in flight: {bound['Sol']:.2f} while the cap is reached before " \
        f"the last {later} calls of the first Sol block, {bound['Neon']:.2f} before the last {later} calls of the " \
        f"Neon {priced['blocks'][1][1]} block." in text
    assert priced["neon_from"] == priced["blocks"][0][1] - later + 1                # the Sol block's last 7 calls
    last = priced["blocks"][-1]
    assert f"From the {priced['fable_from']}th of the {calls} calls on a Fable call can be in flight, also among the " \
        f"last {last[1]} {SHORT[last[0]]} calls." in text
    assert f"{top:.2f} USD bounds the overshoot ({bound['Neon']:.2f} if the cap is reached before the " \
        f"{priced['fable_from']}th call)" in text
    assert "when it is reached in a Sol block" not in text and "(or the last 4 Sol calls)" not in text
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
    assert f"Acht Sol-Aufrufe kosten höchstens etwa {comma(bound['Sol'])} $ und acht Neon-Aufrufe " \
        f"{comma(bound['Neon'])} $, aber ab dem {priced['fable_from']}. der {calls} Aufrufe kann ein Fable-Aufruf " \
        f"mitlaufen, auch bei den letzten {last[1]} Sol-Aufrufen." in text_de
    assert f"und dann gilt die Grenze von {comma(top)} $" in text_de
    assert "wenn der Deckel während der Sol-Aufrufe erreicht wird" not in text_de
    assert "cannot spend more" not in text and "mehr wird also nicht ausgegeben" not in text_de
    assert "so a run can end above the cap by what the items in flight cost" in section(6)
    assert _keys_run_parser().get_default("concurrency") == CONCURRENCY           # the default the figure assumes
    max_cost = next(a for a in _keys_run_parser()._actions if "--max-cost" in a.option_strings)
    help_text = " ".join((max_cost.help or "").split())
    assert f"at max_tokens and the default of {CONCURRENCY} up to about {top:.2f} USD (eight Fable items: the items " \
        "in flight can belong to a neighbouring block of the plan" in help_text
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
    assert "the rules decide alone only the answers that do not name the donor's name" in help_text
    assert "A failed or tied vote leaves the answer undecided, never to the rules" in help_text
    assert "for the right name also another spelling or a doubt" not in help_text
    six = section(6)
    assert "**The judges' scope** (with `--judges`). The adoption judges decide every labelled answer that mentions " \
        "the name it is measured against, whatever the rules would decide" in six
    assert "`mentions(text, name, event)` reads widely" in six and "rule_adopted" in six
    seven = section(7)
    assert "The gates, the 20 % limit and the check against the undecided rows are the same whoever decides the rows." \
        in seven and "an answered row is undecided only when the judges' votes failed or tied" in seven


def test_section_10_records_the_judges_scope_and_its_refinements(priced):
    """The check of 863ca47 added four rows to the refinements table: the judges' scope, the rules without judges, the
    3b effect of the wider note references (the row on the note in more words named only the right name) and the
    cap's overshoot; the intro counts the rows each review added. The review of aafe7f9 added four more (a two-letter
    surname alone, a correction of another name, a judges' decision on resume, the rules-versus-judges table) and
    corrected two of the check's rows in place: the dash clause and the further references to the note in "the rules
    without judges", and the bound in "the cap's overshoot", whose figures are pinned here."""
    text = DESIGN.read_text("utf-8")
    table = text[text.index("### Refinements during the implementation review"):]
    titles = [line.split(" | ")[0].removeprefix("| ") for line in table.splitlines() if line.startswith("| ")][1:]
    check = ["the judges' scope", "the rules without judges", "the wider note references on the donor's name",
             "the cap's overshoot"]
    review = ["a two-letter surname alone", "a correction of another name, on every arm",
              "a judges' decision on resume", "the rules-versus-judges table"]
    assert titles[-8:] == check + review
    seven = titles[titles.index("another role or a rejection in other words"):titles.index("the judges' scope")]
    assert len(seven) == 7 and seven[-1] == "the pilot's total and the cap"
    intro = " ".join(table[: table.index("| issue | before | now |")].split())
    assert "added the seven rows from \"another role or a rejection in other words\" to \"the pilot's total and the " \
        "cap\"" in intro
    assert "added the four rows from \"the judges' scope\" to \"the cap's overshoot\"" in intro
    assert "A review of that change added the last four rows, from \"a two-letter surname alone\" on" in intro
    assert "corrected two rows of the check in place: \"the rules without judges\"" in intro
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
    total = priced["recipients"] + sum(f[k][2] for k in ("grading", "right", "adopt_full", "alone"))
    assert "0.23 USD is an estimate from the stored maxima" in over
    assert f"(4096 per call, {BUDGET} per judge call)" in over
    assert f"up to about {max(bound.values()):.2f} USD above the cap at the default of {CONCURRENCY} (eight Fable " \
        f"items: the items in flight can belong to a neighbouring block, so eight Sol items, {bound['Sol']:.2f}, and " \
        f"eight Neon items, {bound['Neon']:.2f}, bound it only away from a dearer block, and from the " \
        f"{priced['fable_from']}th call on a Fable call can be in flight)" in over
    assert f"the pilot's cap is 3.5 USD (under 3 the judges at their worst left about {3.0 - total:.2f} USD)" in over
    assert "in a Sol block, 0.81 in a Neon block" not in over
    added = {k: rows[k] for k in review}
    assert "`mentions` found a part of the name on its own only with three letters or more" in \
        added["a two-letter surname alone"]
    correction = added["a correction of another name, on every arm"]
    assert "`offers_answer`" in correction and f"the replay's {f['grading'][0]} grading-judge calls stay" in correction
    assert "`adoption_name`" in added["a judges' decision on resume"]
    assert "a column of their own" in added["the rules-versus-judges table"]
