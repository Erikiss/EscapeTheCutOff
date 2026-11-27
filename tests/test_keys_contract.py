"""Key, not value: the contract (docs/DESIGN_keys.md) states what the code does. A pre-registered rule text that changed
has its row in section 10's refinements table, and the judge-cost figures of section 9 follow from the stored data the
way section 9 says it prices a judge call (final review: prediction 1's rule text changed without a row; the judges
were priced like probe calls, which understated the worst case the owner approves the pilot on). The README's cost
paragraph, the recipients' estimate per model, the stored judge prices and the cap's overshoot are pinned too, and the
texts of the exit statuses and of the judges' scope say what the code does (review of the final review)."""

from __future__ import annotations

import argparse
import re
from statistics import mean

import pytest
from keys_testlib import EVENTS, RESULTS, ROOT, TODAY

import cutoffmix.cli
from cutoffmix.bench.grading import grade_rule
from cutoffmix.bench.io import load_events, load_results
from cutoffmix.bench.keys import build_plan, estimate_cost, row_cost
from cutoffmix.bench.keys.run import adoption_judge_messages
from cutoffmix.bench.prompts import judge_messages, probe_messages

DESIGN = ROOT / "docs" / "DESIGN_keys.md"
README = ROOT / "README.md"
JUDGE = "gpt-6.2-sol"  # the pilot's judge (design section 6)
PROMPT_USD, COMPLETION_USD = 2e-6, 10e-6  # per token, as section 9 states them (checked against the stored rows)
CONCURRENCY = 8  # the default of 'bench keys run --concurrency'


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
    # the cap's overshoot: up to --concurrency items in flight, each the most expensive stored call of a recipient of
    # the plan and two judge calls (grading and adoption) at their worst
    worst_judge = max(max(usd(judge_messages(by_id[p["event_id"]], "current_d", "x" * longest), True),
                          usd(adoption_judge_messages(by_id[p["event_id"]], p["donor_answer"] or "", "x" * longest),
                              True)) for p in plan)
    recipients = {p["recipient"] for p in plan}
    worst_call = max(row_cost({"usage": r.usage}) for r in results if r.probe == "current_d" and r.model in recipients)
    estimate = estimate_cost(plan, results)
    return {"figures": figures, "chars": chars, "completion": completion, "longest_completion": longest_completion,
            "longest": longest, "recipients": estimate["total_usd"], "by_model": estimate["by_model"],
            "calls": len(plan), "judge_rows": judge, "dark_full": sum(1 for p in right if p["set"] == "dark"),
            "both_right_full": sum(1 for p in right if p["set"] == "both_right"), "worst_call": worst_call,
            "overshoot": CONCURRENCY * (worst_call + 2 * worst_judge)}


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
    # the total is the recipients' mean estimate plus the judges at their worst: no worst case, no bound (review of the
    # final review: it read "about 2.94 USD at worst")
    assert total < 3.0 and f"Added to the recipients' estimate that is about {total:.2f} USD, about " \
        f"{3.0 - total:.2f} USD under the pilot's cap of 3 USD. It is no bound:" in text
    assert "USD at worst: under the pilot's cap" not in text
    # the README states the same figures (in German, decimal comma)
    text_de = readme()
    assert f"bis etwa {comma(every_full)} $ für den Judge (im ungünstigsten Fall {comma(at_worst)} $; zusammen " \
        f"geschätzt {comma(total)} $, eine Schätzung, keine Obergrenze: die setzt `--max-cost`)" in text_de
    assert re.search(rf"etwa {comma(rules_only)} \$ .* bis etwa {comma(every_full)} \$, wenn er jede", text_de)
    n_right, n_adopt = f["right"][0], f["adopt_full"][0]
    cents = round(100 * (3.0 - total))
    assert f"(bis zu {n_right} Aufrufe für den richtigen Namen, {n_adopt} für den falschen), im ungünstigsten Fall " \
        f"etwa {comma(at_worst)} $. Mit der Schätzung für die Empfänger sind das etwa {comma(total)} $, also etwa " \
        f"{cents} Cent unter `--max-cost 3`. Eine Obergrenze ist das nicht" in text_de
    assert "im ungünstigsten Fall etwa 2" not in text_de and "Zusammen sind das im ungünstigsten Fall" not in text_de


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
    the run cannot spend more)."""
    text, text_de = section(9), readme()
    overshoot, worst_call = priced["overshoot"], priced["worst_call"]
    assert f"up to `--concurrency` items, at the default of {CONCURRENCY} at most about {overshoot:.2f} USD " \
        f"(each item the most expensive stored recipient call, {worst_call:.3f} USD, and two judge calls at their " \
        "worst)" in text
    assert f"bei der Voreinstellung {CONCURRENCY} um höchstens etwa {comma(overshoot)} $" in text_de
    assert "cannot spend more" not in text and "mehr wird also nicht ausgegeben" not in text_de
    assert "so a run can end above the cap by what the items in flight cost" in section(6)
    assert _keys_run_parser().get_default("concurrency") == CONCURRENCY           # the default the figure assumes


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
    """For the donor's name, another spelling, a doubt or a rejection is absent: not adopted, by rule. Only for the
    right name of a full note do these go to the judges; the README and the --judges help said so for both names
    (review of the final review)."""
    text_de = readme()
    assert "Beim richtigen Namen einer `full`-Notiz (3a) entscheidet alles andere der Judge" in text_de
    assert "Beim falschen Namen des Spenders (3b) gilt eine Antwort nach den Regeln als übernommen, wenn sie sich " \
        "den Namen zu eigen macht, die Bewertung sie als falsche Antwort liest und sie niemand anderen als Antwort " \
        "anbietet; als nicht übernommen gilt sie nach den Regeln, wenn sie den Namen nicht als Antwort anbietet" in \
        text_de
    assert "sonst entscheidet der Judge" not in text_de and "noch eine andere Person nennt" not in text_de
    judges = next(a for a in _keys_run_parser()._actions if "--judges" in a.option_strings)
    help_text = " ".join((judges.help or "").split())
    assert "for the right name also another spelling or a doubt (for the donor's name these are not adopted, by " \
        "rule)" in help_text
    assert "(the donor's name, or the right name of a full note: reported speech, another spelling" not in help_text
