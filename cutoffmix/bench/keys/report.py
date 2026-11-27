"""The report of the key experiment: comparisons and verdicts on the pre-registered predictions
(docs/DESIGN_keys.md sections 6 and 7)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from cutoffmix.bench.keys.common import ARMS_BY_SET, SETS, row_cost
from cutoffmix.bench.keys.stats import fisher_exact, mcnemar_exact, wilson

_MIN_TESTABLE_PAIRED = 10  # paired dark items a comparison needs before a prediction counts as testable
_MIN_TESTABLE_GROUP = 5  # rows per group (recognised / unrecognised, adoption items) for the other predictions
_HARM_LIMIT = 0.10  # prediction 4: harm under 10 %


# ------------------------------------------------------------------------------------------------------- report


def _usable(row: Mapping[str, Any] | None) -> bool:
    return row is not None and not row.get("error") and row.get("grade") not in (None, "ungraded")


def _rate(k: int, n: int) -> dict[str, Any]:
    lo, hi = wilson(k, n)
    return {"n": n, "k": k, "rate": (k / n) if n else None, "ci": [round(lo, 4), round(hi, 4)]}


def _paired(items: Mapping[str, Mapping[str, Mapping[str, Any]]], first: str, second: str,
            ok: Callable[[Mapping[str, Any]], bool]) -> dict[str, Any]:
    b = sum(1 for arms in items.values() if ok(arms[first]) and not ok(arms[second]))
    c = sum(1 for arms in items.values() if ok(arms[second]) and not ok(arms[first]))
    return {"b": b, "c": c, "p": round(mcnemar_exact(b, c), 6), "n_items": len(items)}


def _correct(row: Mapping[str, Any]) -> bool:
    return row.get("grade") == "correct"


def build_report(plan: Sequence[Mapping[str, Any]], results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Join plan and results (the last row per id counts) into the numbers of design section 6 and the verdicts on
    the pre-registered predictions of section 7. Rows with an ``error`` or the grade ``ungraded`` are left out of
    every ``n`` and counted separately; adoption only needs an answer (no error, a non-empty response)."""
    last: dict[str, Mapping[str, Any]] = {}
    for row in results:
        last[str(row.get("id"))] = row
    planned = {str(p["id"]): p for p in plan}
    joined = [(p, last.get(i)) for i, p in planned.items()]
    rows = [r for _, r in joined if r is not None]
    totals = {
        "planned": len(planned), "rows": len(rows),
        "missing": sum(1 for _, r in joined if r is None),
        "errors": sum(1 for r in rows if r.get("error")),
        "ungraded": sum(1 for r in rows if not r.get("error") and r.get("grade") == "ungraded"),
        "refusals": sum(1 for r in rows if r.get("grader") == "refusal"),
        "cost_usd": round(sum(row_cost(r) for r in rows), 4),
        "by_recipient": dict(Counter(str(p["recipient"]) for p, r in joined if r is not None)),
    }
    pair_names = list(dict.fromkeys(f"{p['donor']}->{p['recipient']}" for p in plan))

    def pick(pair: str, set_: str) -> list[tuple[Mapping[str, Any], Mapping[str, Any] | None]]:
        return [(p, r) for p, r in joined if f"{p['donor']}->{p['recipient']}" == pair and p["set"] == set_]

    pairs: dict[str, dict[str, Any]] = {}
    paired: dict[str, dict[str, Any]] = {}
    recognised: dict[str, dict[str, Any]] = {}
    adoption: dict[str, dict[str, Any]] = {}
    harm: dict[str, dict[str, Any]] = {}
    pooled_dark: dict[str, dict[str, Mapping[str, Any]]] = {}
    pooled_rec: dict[bool, list[Mapping[str, Any]]] = {True: [], False: []}
    pooled_adopt: dict[str, dict[str, Mapping[str, Any]]] = {}
    pooled_harm: list[Mapping[str, Any]] = []
    for pair in pair_names:
        pairs[pair] = {}
        for set_ in SETS:
            chosen = pick(pair, set_)
            if not chosen:
                continue
            arms = {}
            for arm in ARMS_BY_SET[set_]:
                at = [(p, r) for p, r in chosen if p["arm"] == arm]
                ok = [r for _, r in at if _usable(r)]
                entry = _rate(sum(1 for r in ok if _correct(r)), len(ok))
                entry.update(planned=len(at), missing=sum(1 for _, r in at if r is None),
                             errors=sum(1 for _, r in at if r is not None and r.get("error")),
                             ungraded=sum(1 for _, r in at if r is not None and not r.get("error")
                                          and r.get("grade") == "ungraded"))
                arms[arm] = entry
            pairs[pair][set_] = {"items": len({p["event_id"] for p, _ in chosen}), "arms": arms}
        # dark: paired comparisons on the items with every arm graded, and the split by recognition
        items: dict[str, dict[str, Mapping[str, Any]]] = {}
        flags: dict[str, Any] = {}
        for p, r in pick(pair, "dark"):
            items.setdefault(str(p["event_id"]), {})[str(p["arm"])] = r
            flags[str(p["event_id"])] = p.get("recognised")
        complete = {e: a for e, a in items.items()
                    if all(_usable(a.get(arm)) for arm in ARMS_BY_SET["dark"])}
        if items:
            paired[pair] = {"n_items": len(complete),
                            "key_vs_alone": _paired(complete, "key", "alone", _correct),
                            "key_vs_wrong_key": _paired(complete, "key", "wrong_key", _correct),
                            "full_vs_key": _paired(complete, "full", "key", _correct)}
            for event_id, arms_rows in complete.items():
                pooled_dark[f"{pair}|{event_id}"] = arms_rows
            groups: dict[str, dict[str, Any]] = {}
            for label, flag in (("recognised", True), ("unrecognised", False)):
                group: dict[str, Any] = {}
                for arm in ARMS_BY_SET["dark"]:
                    ok = [a[arm] for e, a in items.items() if flags.get(e) is flag and _usable(a.get(arm))]
                    group[arm] = _rate(sum(1 for r in ok if _correct(r)), len(ok))
                    if arm == "key":
                        pooled_rec[flag].extend(ok)
                groups[label] = group
            recognised[pair] = groups
        # adoption of the donor's wrong name: alone vs full, on answers (no error, a response)
        for set_ in ("donor_wrong", "control"):
            chosen = pick(pair, set_)
            if not chosen:
                continue
            entry_set: dict[str, Any] = {}
            by_item: dict[str, dict[str, Mapping[str, Any]]] = {}
            for arm in ARMS_BY_SET[set_]:
                ok = [r for p, r in chosen if p["arm"] == arm and r is not None and not r.get("error")
                      and str(r.get("response") or "").strip()]
                entry_set[arm] = _rate(sum(1 for r in ok if r.get("adopted")), len(ok))
            for p, r in chosen:
                if r is not None and not r.get("error") and str(r.get("response") or "").strip():
                    by_item.setdefault(str(p["event_id"]), {})[str(p["arm"])] = r
            for event_id, arms_rows in by_item.items():
                if all(a in arms_rows for a in ARMS_BY_SET[set_]):
                    pooled_adopt[f"{pair}|{set_}|{event_id}"] = arms_rows
            adoption.setdefault(pair, {})[set_] = entry_set
        chosen = pick(pair, "both_right")
        if chosen:
            entry_h: dict[str, Any] = {}
            for arm in ARMS_BY_SET["both_right"]:
                ok = [r for p, r in chosen if p["arm"] == arm and _usable(r)]
                entry_h[arm] = _rate(sum(1 for r in ok if not _correct(r)), len(ok))
                if arm == "key":
                    pooled_harm.extend(ok)
            harm[pair] = entry_h

    pooled = {
        "dark_items": len(pooled_dark),
        "full_vs_key": _paired(pooled_dark, "full", "key", _correct) if pooled_dark else None,
        "recognised_key": _rate(sum(1 for r in pooled_rec[True] if _correct(r)), len(pooled_rec[True])),
        "unrecognised_key": _rate(sum(1 for r in pooled_rec[False] if _correct(r)), len(pooled_rec[False])),
        "adoption_items": len(pooled_adopt),
        "adoption_full_vs_alone": _paired(pooled_adopt, "full", "alone", lambda r: bool(r.get("adopted")))
        if pooled_adopt else None,
        "harm_key": _rate(sum(1 for r in pooled_harm if not _correct(r)), len(pooled_harm)),
    }
    report = {"totals": totals, "pairs": pairs, "paired": paired, "recognised": recognised, "adoption": adoption,
              "harm": harm, "pooled": pooled}
    report["predictions"] = _predictions(report)
    return report


def _fmt_rate(entry: Mapping[str, Any] | None) -> str:
    if not entry or not entry.get("n"):
        return "n/a"
    lo, hi = entry["ci"]
    return f"{entry['k']}/{entry['n']} ({100 * entry['rate']:.1f} %, {100 * lo:.1f}-{100 * hi:.1f})"


def _fmt_test(entry: Mapping[str, Any] | None) -> str:
    if not entry:
        return "n/a"
    return f"b={entry['b']}, c={entry['c']}, p={entry['p']:.4g}"


def _predictions(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Verdicts on the four pre-registered predictions (design section 7), conservative by construction."""
    out: list[dict[str, Any]] = []
    paired = report["paired"]
    pooled = report["pooled"]
    pairs = report["pairs"]
    # 1. key beats alone and wrong_key on dark items, p < 0.05 on (at least) the largest pair
    if paired:
        largest = max(sorted(paired), key=lambda k: paired[k]["n_items"])
        entry = paired[largest]
        if entry["n_items"] < _MIN_TESTABLE_PAIRED:
            verdict = "not testable (n too small)"
        else:
            wins = all(entry[c]["b"] > entry[c]["c"] and entry[c]["p"] < 0.05
                       for c in ("key_vs_alone", "key_vs_wrong_key"))
            verdict = "supported" if wins else "not supported"
        arms = pairs.get(largest, {}).get("dark", {}).get("arms", {})
        evidence = (f"largest pair {largest}, {entry['n_items']} items with all four arms graded; key vs alone "
                    f"{_fmt_test(entry['key_vs_alone'])}; key vs wrong key {_fmt_test(entry['key_vs_wrong_key'])}; "
                    f"correct: key {_fmt_rate(arms.get('key'))}, wrong key {_fmt_rate(arms.get('wrong_key'))}, "
                    f"alone {_fmt_rate(arms.get('alone'))} (expected: key 10-30 %, wrong key under 10 %)")
    else:
        verdict, evidence = "not testable (n too small)", "no dark items in the results"
    out.append({"id": 1, "text": "On dark items, key beats alone and wrong_key (paired McNemar, p < 0.05, largest "
                                 "pair).", "verdict": verdict, "evidence": evidence})
    # 2. recovery concentrated on recognised items (pooled key arm), two-sided Fisher exact p < 0.05
    rec, unrec = pooled["recognised_key"], pooled["unrecognised_key"]
    if rec["n"] < _MIN_TESTABLE_GROUP or unrec["n"] < _MIN_TESTABLE_GROUP:
        verdict = "not testable (n too small)"
        p = None
    else:
        p = fisher_exact(rec["k"], rec["n"] - rec["k"], unrec["k"], unrec["n"] - unrec["k"])
        verdict = "supported" if rec["rate"] > unrec["rate"] and p < 0.05 else "not supported"
    out.append({"id": 2, "text": "The key recovers recognised items (the recipient passes the MCQ) more often than "
                                 "unrecognised ones.", "verdict": verdict,
                "evidence": f"key arm, all pairs: recognised {_fmt_rate(rec)}, unrecognised {_fmt_rate(unrec)}"
                            + (f", Fisher exact p={p:.4g}" if p is not None else "")})
    # 3. full beats key on dark items, and full makes the recipient adopt a wrong donor answer more often than alone
    fk, ad = pooled["full_vs_key"], pooled["adoption_full_vs_alone"]
    if not fk or fk["n_items"] < _MIN_TESTABLE_PAIRED or not ad or ad["n_items"] < _MIN_TESTABLE_GROUP:
        verdict = "not testable (n too small)"
    else:
        ok = fk["b"] > fk["c"] and fk["p"] < 0.05 and ad["b"] > ad["c"] and ad["p"] < 0.05
        verdict = "supported" if ok else "not supported"
    out.append({"id": 3, "text": "full beats key on dark items, and full makes the recipient adopt a wrong donor "
                                 "answer far more often than alone (paired, p < 0.05 each).", "verdict": verdict,
                "evidence": f"full vs key, all pairs ({fk['n_items'] if fk else 0} items): {_fmt_test(fk)}; "
                            f"adoption full vs alone ({ad['n_items'] if ad else 0} items): {_fmt_test(ad)}"})
    # 4. harm under 10 % in both_right under key
    hk = pooled["harm_key"]
    if hk["n"] < _MIN_TESTABLE_PAIRED:
        verdict = "not testable (n too small)"
    else:
        verdict = "supported" if hk["rate"] < _HARM_LIMIT else "not supported"
    out.append({"id": 4, "text": "A key rarely breaks a correct recall (both_right, harm under 10 %).",
                "verdict": verdict, "evidence": f"not correct under key, all pairs: {_fmt_rate(hk)} (the observed "
                                                f"rate must be under 10 %; the interval is reported)"})
    return out


def render_report(report: Mapping[str, Any]) -> str:
    """The report as Markdown: totals, per pair and set the correct rate per arm with its 95 % Wilson interval, the
    paired comparisons, the split by recognition, adoption, harm and the prediction verdicts."""
    t = report["totals"]
    lines = ["# Key, not value: report", "",
             f"Planned calls {t['planned']}, rows {t['rows']} (missing {t['missing']}, errors {t['errors']}, "
             f"ungraded {t['ungraded']}, refusals {t['refusals']}), cost {t['cost_usd']:.4f} USD.", ""]
    lines += ["## Correct answers per arm", "",
              "| pair | set | items | " + " | ".join(ARMS_BY_SET["dark"]) + " |",
              "|---|---|---:|" + "---|" * len(ARMS_BY_SET["dark"])]
    for pair, sets in report["pairs"].items():
        for set_, entry in sets.items():
            cells = [_fmt_rate(entry["arms"].get(arm)) if arm in entry["arms"] else "" for arm in ARMS_BY_SET["dark"]]
            lines.append(f"| {pair} | {set_} | {entry['items']} | " + " | ".join(cells) + " |")
    lines += ["", "## Paired comparisons on dark items (exact McNemar; b = first arm right and second wrong)", "",
              "| pair | items | key vs alone | key vs wrong key | full vs key |", "|---|---:|---|---|---|"]
    for pair, entry in report["paired"].items():
        lines.append(f"| {pair} | {entry['n_items']} | {_fmt_test(entry['key_vs_alone'])} | "
                     f"{_fmt_test(entry['key_vs_wrong_key'])} | {_fmt_test(entry['full_vs_key'])} |")
    lines += ["", "## Dark items by recognition (the recipient passes the MCQ)", "",
              "| pair | group | " + " | ".join(ARMS_BY_SET["dark"]) + " |",
              "|---|---|" + "---|" * len(ARMS_BY_SET["dark"])]
    for pair, groups in report["recognised"].items():
        for label, group in groups.items():
            lines.append(f"| {pair} | {label} | " + " | ".join(_fmt_rate(group.get(a)) for a in ARMS_BY_SET["dark"])
                         + " |")
    lines += ["", "## Adoption of the donor's wrong name", "", "| pair | set | alone | full |", "|---|---|---|---|"]
    for pair, sets in report["adoption"].items():
        for set_, entry in sets.items():
            lines.append(f"| {pair} | {set_} | {_fmt_rate(entry.get('alone'))} | {_fmt_rate(entry.get('full'))} |")
    lines += ["", "## Harm on items both answered correctly (share not correct)", "", "| pair | key | full |",
              "|---|---|---|"]
    for pair, entry in report["harm"].items():
        lines.append(f"| {pair} | {_fmt_rate(entry.get('key'))} | {_fmt_rate(entry.get('full'))} |")
    lines += ["", "## Pre-registered predictions (docs/DESIGN_keys.md section 7)", ""]
    for p in report["predictions"]:
        lines.append(f"{p['id']}. {p['text']} **{p['verdict']}**: {p['evidence']}")
    return "\n".join(lines) + "\n"
