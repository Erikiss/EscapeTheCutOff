"""Key, not value: does another model's cue unlock a model's own memory? (docs/DESIGN_keys.md)

A text-level analogue of Hazy Research's "Plato's Key" (25 November 2026) for models that are reached only through an
API. For a donor model A that recalled a fact and a recipient model B that missed it, B is asked again alone, with A's
note whose answer is masked (``key``), with A's whole note (``full``), and with A's masked note about another event
(``wrong_key``). Further sets measure whether B adopts a *wrong* donor answer (``donor_wrong``, ``control``) and
whether a note breaks a correct recall (``both_right``).

- :mod:`.text`: masking (:func:`mask_title`, :func:`mask_note`) and reading names in answers.
- :mod:`.plan`: :func:`build_plan` turns stored probe results into the planned calls (offline, deterministic for its
  inputs); :func:`estimate_cost` prices them from the usage stored with those results.
- :mod:`.run`: :func:`run_plan` sends and grades them (resumable, bounded concurrency, optional cost cap).
- :mod:`.stats`: Wilson intervals and exact tests.
- :mod:`.report`: :func:`build_report` and :func:`render_report` compute the comparisons and evaluate the
  pre-registered predictions of section 7 of the design.

Every name below is re-exported here, so ``from cutoffmix.bench import keys`` and ``keys.build_plan(...)`` keep working.
"""

from __future__ import annotations

# ---- shared constants and helpers (common.py)
from cutoffmix.bench.keys.common import (
    ARMS_BY_SET,
    BLANK_NOTE,
    KEY_USER_TEMPLATE,
    PROBE,
    SETS,
    TITLE,
    WITHHELD,
    judge_cost,
    prompt_sha,
    row_cost,
)

# ---- masking and names (text.py)
from cutoffmix.bench.keys.text import (
    ADOPTION_LABELS,
    adoption_label,
    asserted_name,
    contradicted,
    cue_tokens,
    disclaims,
    mask_note,
    mask_title,
    offers,
    offers_another,
)

# ---- plan and cost (plan.py)
from cutoffmix.bench.keys.plan import (
    build_plan,
    estimate_cost,
)

# ---- run (run.py)
from cutoffmix.bench.keys.run import (
    RunOutcome,
    RunStart,
    request_sha,
    run_plan,
)

# ---- statistics (stats.py)
from cutoffmix.bench.keys.stats import (
    fisher_exact,
    items_needed,
    mcnemar_exact,
    stratified_difference,
    stratified_permutation_test,
    wilson,
)

# ---- report (report.py)
from cutoffmix.bench.keys.report import (
    build_report,
    render_report,
)

__all__ = [
    "ADOPTION_LABELS",
    "ARMS_BY_SET",
    "BLANK_NOTE",
    "KEY_USER_TEMPLATE",
    "PROBE",
    "SETS",
    "TITLE",
    "WITHHELD",
    "RunOutcome",
    "RunStart",
    "adoption_label",
    "asserted_name",
    "build_plan",
    "build_report",
    "contradicted",
    "cue_tokens",
    "disclaims",
    "estimate_cost",
    "fisher_exact",
    "items_needed",
    "judge_cost",
    "mask_note",
    "mask_title",
    "mcnemar_exact",
    "offers",
    "offers_another",
    "prompt_sha",
    "render_report",
    "request_sha",
    "row_cost",
    "run_plan",
    "stratified_difference",
    "stratified_permutation_test",
    "wilson",
]
