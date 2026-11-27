"""Key, not value: the contract (docs/DESIGN_keys.md) states what the code does. A pre-registered rule text that changed
has its row in section 10's refinements table, and the judge-cost figures of section 9 follow from the stored data the
way section 9 says it prices a judge call (final review: prediction 1's rule text changed without a row; the judges
were priced like probe calls, which understated the worst case the owner approves the pilot on). The README's cost
paragraph, the recipients' estimate per model, the stored judge prices and the cap's overshoot are pinned too, and the
texts of the exit statuses and of the judges' scope say what the code does (review of the final review). With the
judges' scope (the judges decide every labelled answer that names the candidate), the counts follow that scope in the
replay, the overshoot is bounded at max_tokens per block of the plan's order, and the pilot's cap is 3.5 USD."""

from __future__ import annotations

import argparse
import itertools
from statistics import mean

import pytest
from keys_testlib import EVENTS, RESULTS, ROOT, TODAY

import cutoffmix.cli
from cutoffmix.bench.grading import grade_rule
from cutoffmix.bench.io import load_events, load_results
from cutoffmix.bench.keys import build_plan, estimate_cost, row_cost
from cutoffmix.bench.keys.run import _JUDGE_MAX_TOKENS, adoption_judge_messages
from cutoffmix.bench.prompts import judge_messages, probe_messages

DESIGN = ROOT / "docs" / "DESIGN_keys.md"
README = ROOT / "README.md"
JUDGE = "gpt-6.2-sol"  # the pilot's judge (design section 6)
PROMPT_USD, COMPLETION_USD = 2e-6, 10e-6  # per token, as section 9 states them (checked against the stored rows)
CONCURRENCY = 8  # the default of 'bench keys run --concurrency'
MAX_TOKENS = 4096  # every planned call's max_tokens (design section 4)
CAP = 3.5  # the pilot's cap (design section 9)
SHORT = {"gpt-6.2-sol": "Sol", "gemini-4-neon": "Neon", "claude-fable-5.2": "Fable"}


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

    grading = [p for p in plan if grade_rule(by_id[p["event_id"]], "current_d", answer(p)) is None]
    right = [p for p in plan if p["set"] in ("dark", "both_right") and p["arm"] == "full"]
    adopt_full = [p for p in plan if p["set"] in ("donor_wrong", "control") and p["arm"] == "full"]
    alone_due = []  # the alone rows of the replay that mention the donor's name: the judges decide them
    for p in (p for p in plan if p["set"] in ("donor_wrong", "control") and p["arm"] == "alone"):
        ruled = grade_rule(by_id[p["event_id"]], "current_d", answer(p))
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
    judge_bound = (template / chars + MAX_TOKENS) * PROMPT_USD + _JUDGE_MAX_TOKENS * COMPLETION_USD

    def item_bound(p) -> float:
        prompt_usd, completion_usd, cpt, _, _ = prices[p["recipient"]]
        tokens = sum(len(m["content"]) for m in p["messages"]) / cpt
        return tokens * prompt_usd + MAX_TOKENS * completion_usd + 2 * judge_bound

    blocks = [(model, len(rows), max(item_bound(p) for p in rows))
              for model, rows in ((k, list(g)) for k, g in itertools.groupby(plan, key=lambda p: p["recipient"]))]
    first = plan[: blocks[0][1]]
    first_ids = {p["id"] for p in first}
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
            "first": estimate_cost(first, results)["total_usd"] + first_judges}


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
    bounds the overshoot per block of the plan's order at max_tokens (review of 863ca47: "at most about 0.23 USD" was
    no bound)."""
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
    assert f"ends at most about {bound['Sol']:.2f} USD above the cap when it is reached in a Sol block, " \
        f"{bound['Neon']:.2f} in a Neon block and {bound['Fable']:.2f} in a Fable block, at the default of " \
        f"{CONCURRENCY}" in text
    order = ", then ".join(f"{SHORT[m]} {n}" for m, n, _ in priced["blocks"][:2])
    rest = ", ".join(f"{SHORT[m]} {n}" for m, n, _ in priced["blocks"][2:-1])
    last = priced["blocks"][-1]
    assert f"blocks of one recipient: {order.replace(' 439', ' 439 first')}, {rest} and {SHORT[last[0]]} {last[1]}" \
        in text
    assert f"first Sol block costs about {priced['by_model']['gpt-6.2-sol']['estimate_usd']:.2f} USD, about " \
        f"{priced['first']:.2f} with its judges at their worst" in text
    assert f"Aus den längsten gespeicherten Antworten geschätzt sind das bei der Voreinstellung {CONCURRENCY} etwa " \
        f"{comma(overshoot)} $, aber das ist keine Grenze" in " ".join(text_de.split())
    assert f"höchstens etwa {comma(bound['Sol'])} $, wenn der Deckel während der Sol-Aufrufe erreicht wird, " \
        f"{comma(bound['Neon'])} $ bei Neon und {comma(bound['Fable'])} $ bei Fable" in text_de
    assert f"mit dem Judge im ungünstigsten Fall etwa {comma(priced['first'])} $" in text_de
    assert "cannot spend more" not in text and "mehr wird also nicht ausgegeben" not in text_de
    assert "so a run can end above the cap by what the items in flight cost" in section(6)
    assert _keys_run_parser().get_default("concurrency") == CONCURRENCY           # the default the figure assumes
    max_cost = next(a for a in _keys_run_parser()._actions if "--max-cost" in a.option_strings)
    help_text = " ".join((max_cost.help or "").split())
    assert f"up to about {bound['Sol']:.2f} USD in a Sol block of the plan, {bound['Neon']:.2f} in a Neon block, " \
        f"{bound['Fable']:.2f} in a Fable block" in help_text and "the pilot asks for 3.5" in help_text


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


def test_section_10_records_the_judges_scope_and_its_refinements():
    """The check of 863ca47 added four rows to the refinements table: the judges' scope, the rules without judges, the
    3b effect of the wider note references (the row on the note in more words named only the right name) and the
    cap's overshoot; the intro counts the rows each review added."""
    text = DESIGN.read_text("utf-8")
    table = text[text.index("### Refinements during the implementation review"):]
    titles = [line.split(" | ")[0].removeprefix("| ") for line in table.splitlines() if line.startswith("| ")][1:]
    assert titles[-4:] == ["the judges' scope", "the rules without judges",
                           "the wider note references on the donor's name", "the cap's overshoot"]
    seven = titles[titles.index("another role or a rejection in other words"):titles.index("the judges' scope")]
    assert len(seven) == 7 and seven[-1] == "the pilot's total and the cap"
    intro = " ".join(table[: table.index("| issue | before | now |")].split())
    assert "added the seven rows from \"another role or a rejection in other words\" to \"the pilot's total and the " \
        "cap\"" in intro
    assert "added the last four rows, from \"the judges' scope\" on" in intro
    rows = {line.split(" | ")[0].removeprefix("| "): line for line in table.splitlines() if line.startswith("| ")}
    assert "\"Supposedly X.\"" in rows["the wider note references on the donor's name"]
    assert "adopted by rule on `donor_wrong` and `control`" in rows["the wider note references on the donor's name"]
    assert "0.23 USD is an estimate from the stored maxima" in rows["the cap's overshoot"]
    assert "`rule_adopted`" in rows["the judges' scope"] and "`mentions`" in rows["the judges' scope"]
    assert "the plan is unchanged byte for byte" in rows["the rules without judges"]
