"""Domain lenses for the explore preset (DESIGN.md §7.3): the curated fields, their distance to the question's topic
and the seeded client-side draw.

The router's 15 topics are news categories; the lenses are fields of knowledge. Each :class:`Field` has a *home
topic* (for distance) and structural concepts the lens text offers as examples. Distance comes from a fixed ordering
of four topic groups along a line, Science - Tech/Economy - Politics - Culture: same group -> ``near``, adjacent
group -> ``mid``, two or more groups apart -> ``far``. Fields are drawn without replacement within a run, from a tier
sampled with the configured weights (``far`` raised to at least 0.5 while the main line stagnates).

Everything here is pure and deterministic for a seeded :class:`random.Random`, so the relay can log and replay draws.
"""

from __future__ import annotations

import random
from collections.abc import Collection, Mapping
from dataclasses import dataclass

TIERS: tuple[str, ...] = ("mid", "far", "near")  # fixed order of the tier draw (DESIGN.md §7.3)
STAGNATION_FAR_WEIGHT = 0.50  # minimum weight of the far tier while the main line stagnates


@dataclass(frozen=True)
class Field:
    """A field of knowledge a lens may be drawn from."""

    name: str  # stable id, e.g. "materials_science"
    home_topic: str  # taxonomy topic id used for the distance tier
    concepts: tuple[str, ...]  # structural concepts offered as examples (at least three)

    @property
    def label(self) -> str:
        """Display name used in the prompts and the 'Lens (field):' prefix, e.g. "materials science"."""
        return self.name.replace("_", " ")


FIELDS: tuple[Field, ...] = (
    Field("thermodynamics", "science_research",
          ("entropy", "free energy", "phase transition", "Carnot limit", "fluctuation-dissipation", "equipartition")),
    Field("materials_science", "science_research",
          ("fatigue", "grain boundaries", "annealing", "creep", "composite laminate", "phase diagram")),
    Field("chemistry", "science_research",
          ("catalysis", "activation energy", "rate-limiting step", "equilibrium shift", "buffer", "chirality")),
    Field("evolutionary_biology", "science_research",
          ("fitness landscape", "Red Queen dynamics", "drift vs selection", "exaptation", "punctuated equilibrium")),
    Field("immunology", "health_medicine",
          ("clonal selection", "immune memory", "tolerance", "antigenic drift", "cytokine feedback", "herd immunity")),
    Field("epidemiology", "health_medicine",
          ("reproduction number", "base rate", "lead-time bias", "confounding", "dose-response",
           "screening trade-off")),
    Field("ecology", "natural_disasters_climate",
          ("carrying capacity", "niche partitioning", "keystone species", "trophic cascade", "succession",
           "r/K strategies")),
    Field("geoscience_climate", "natural_disasters_climate",
          ("tipping point", "return period", "feedback loop", "early-warning signal", "cascading failure")),
    Field("safety_engineering", "accidents",
          ("Swiss-cheese model", "normal accidents", "redundancy", "common-mode failure", "near-miss reporting",
           "root cause")),
    Field("orbital_mechanics", "space_astronomy",
          ("orbital resonance", "Lagrange point", "gravitational lensing", "redshift as a clock", "Hohmann transfer",
           "parallax")),
    Field("control_theory", "computing_software",
          ("feedback gain", "stability margin", "observability", "hysteresis", "dead time", "PID control")),
    Field("information_theory", "computing_software",
          ("channel capacity", "compression", "error-correcting code", "caching", "race condition",
           "graceful degradation")),
    Field("machine_learning", "ai_ml",
          ("overfitting", "distribution shift", "exploration vs exploitation", "regularisation", "curriculum",
           "reward hacking")),
    Field("economics", "business_economy",
          ("arbitrage", "moral hazard", "network effects", "comparative advantage", "bubble", "Goodhart's law",
           "principal-agent")),
    Field("game_theory", "international_relations",
          ("deterrence", "credible commitment", "prisoner's dilemma", "focal point", "tit-for-tat", "free-riding")),
    Field("military_logistics", "conflict_security",
          ("attrition", "supply line", "fog of war", "escalation ladder", "centre of gravity", "decision loop")),
    Field("political_institutions", "government",
          ("checks and balances", "policy lag", "succession crisis", "veto player", "path dependence")),
    Field("electoral_systems", "elections",
          ("Arrow's theorem", "median voter", "gerrymandering", "Duverger's law", "turnout effect",
           "runoff vs plurality")),
    Field("law_and_evidence", "law_crime",
          ("burden of proof", "precedent", "adversarial testing", "chain of custody", "standard of evidence")),
    Field("linguistics", "arts_entertainment",
          ("grammaticalisation", "sound change", "Zipf's law", "code-switching", "creole formation",
           "semantic drift")),
    Field("music_theory", "arts_entertainment",
          ("counterpoint", "tension and resolution", "motif development", "harmonic series", "polyrhythm")),
    Field("narrative_and_visual_arts", "arts_entertainment",
          ("narrative arc", "negative space", "foreshadowing", "remix", "genre convention")),
    Field("sports_science", "sports",
          ("home advantage", "regression to the mean", "periodisation", "doping arms race", "scouting base rate",
           "formation trade-off")),
)

# Ordered along a line; the tier is the number of groups between the home topic and the current topic.
TOPIC_GROUPS: tuple[tuple[str, ...], ...] = (
    ("science_research", "health_medicine", "space_astronomy", "natural_disasters_climate", "accidents"),
    ("ai_ml", "computing_software", "business_economy"),
    ("government", "elections", "international_relations", "conflict_security", "law_crime"),
    ("sports", "arts_entertainment"),
)
GROUP_NAMES: tuple[str, ...] = ("science", "tech_economy", "politics", "culture")

_FIELD_BY_NAME: dict[str, Field] = {f.name: f for f in FIELDS}
_GROUP_OF: dict[str, int] = {topic: i for i, group in enumerate(TOPIC_GROUPS) for topic in group}


def field_by_name(name: str) -> Field:
    """The field with this id or display name (case-insensitive, spaces or underscores); KeyError otherwise."""
    key = normalise_field(name)
    if key not in _FIELD_BY_NAME:
        raise KeyError(f"unknown lens field {name!r}; known: {', '.join(_FIELD_BY_NAME)}")
    return _FIELD_BY_NAME[key]


def normalise_field(text: str) -> str:
    """A field name as written by a model ("Materials Science", "materials-science") in the id form of :data:`FIELDS`
    ("materials_science"); unknown names are returned normalised, so the caller can compare them."""
    return "_".join(text.strip().strip("*_'\"").lower().replace("-", " ").replace("_", " ").split())


def group_of(topic: str) -> int | None:
    """Index of the topic's group in :data:`TOPIC_GROUPS`, or None for a topic outside the taxonomy."""
    return _GROUP_OF.get(topic)


def tier_of(home_topic: str, current_topic: str | None) -> str:
    """Distance tier of a field's home topic from the question's current topic: ``near`` (same group), ``mid``
    (adjacent groups) or ``far`` (two or more groups apart). An unknown or missing topic on either side gives
    ``mid`` (no distance information: neither obviously close nor obviously far)."""
    a, b = group_of(home_topic), group_of(current_topic) if current_topic is not None else None
    if a is None or b is None:
        return "mid"
    gap = abs(a - b)
    return "near" if gap == 0 else "mid" if gap == 1 else "far"


def fields_by_tier(current_topic: str | None, used_fields: Collection[str] = ()) -> dict[str, list[Field]]:
    """The unused fields grouped by their tier for ``current_topic`` (every tier present, possibly empty)."""
    used = {normalise_field(u) for u in used_fields}
    out: dict[str, list[Field]] = {tier: [] for tier in TIERS}
    for f in FIELDS:
        if f.name not in used:
            out[tier_of(f.home_topic, current_topic)].append(f)
    return out


def tier_weights(tiers: Mapping[str, float], stagnating: bool = False) -> dict[str, float]:
    """The tier weights actually used for a draw, in :data:`TIERS` order: the configured ones, or with the far
    weight raised to at least :data:`STAGNATION_FAR_WEIGHT` (the others scaled down to keep the total) when
    ``stagnating``. Missing tiers weigh 0."""
    w = {tier: max(0.0, float(tiers.get(tier, 0.0))) for tier in TIERS}
    total = sum(w.values())
    if total <= 0:
        raise ValueError("tier weights must have a positive total")
    if stagnating and w["far"] / total < STAGNATION_FAR_WEIGHT:
        rest = total - w["far"]
        target_far = STAGNATION_FAR_WEIGHT * total
        scale = (total - target_far) / rest if rest > 0 else 0.0
        w = {tier: (target_far if tier == "far" else weight * scale) for tier, weight in w.items()}
    return w


def draw_lens(rng: random.Random, current_topic: str | None, used_fields: Collection[str], tiers: Mapping[str, float],
              stagnating: bool = False) -> tuple[Field, str, tuple[str, str, str]] | None:
    """Draw ``(field, tier, three example concepts)`` for a lens note, or None when every field has been used.

    The tier is sampled in :data:`TIERS` order with :func:`tier_weights`; a tier with no unused field falls back to
    the other tiers by descending weight. The field is drawn uniformly from the unused fields of that tier and the
    concepts are a sample without replacement, so the result is a deterministic function of ``rng``'s state.
    """
    weights = tier_weights(tiers, stagnating)
    by_tier = fields_by_tier(current_topic, used_fields)
    if not any(by_tier.values()):
        return None
    tier = rng.choices(list(TIERS), weights=[weights[t] for t in TIERS])[0]
    if not by_tier[tier]:
        for other in sorted(TIERS, key=lambda t: (-weights[t], TIERS.index(t))):
            if by_tier[other]:
                tier = other
                break
    field = rng.choice(by_tier[tier])
    c1, c2, c3 = rng.sample(list(field.concepts), 3)
    return field, tier, (c1, c2, c3)
