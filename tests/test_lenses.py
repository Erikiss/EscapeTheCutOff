"""Tests for cutoffmix.lenses: the field table, distance tiers and the seeded lens draw (DESIGN.md §7.3)."""

from __future__ import annotations

import random
from collections import Counter

import pytest

from cutoffmix import lenses
from cutoffmix.lenses import (
    FIELDS,
    GROUP_NAMES,
    TIERS,
    TOPIC_GROUPS,
    Field,
    draw_lens,
    field_by_name,
    fields_by_tier,
    group_of,
    normalise_field,
    tier_of,
    tier_weights,
)
from cutoffmix.taxonomy import TOPICS

TOPIC_IDS = [t.id for t in TOPICS]
DEFAULT_TIERS = {"mid": 0.55, "far": 0.35, "near": 0.10}


def test_fields_table_matches_design():
    assert len(FIELDS) == 23 and len({f.name for f in FIELDS}) == 23
    assert all(isinstance(f, Field) and f.home_topic in TOPIC_IDS and len(f.concepts) >= 3 for f in FIELDS)
    assert all(len(set(f.concepts)) == len(f.concepts) for f in FIELDS)
    assert [f.name for f in FIELDS[:4]] == ["thermodynamics", "materials_science", "chemistry", "evolutionary_biology"]
    assert FIELDS[0].concepts == ("entropy", "free energy", "phase transition", "Carnot limit",
                                  "fluctuation-dissipation", "equipartition")
    assert field_by_name("economics").concepts[-2:] == ("Goodhart's law", "principal-agent")
    assert field_by_name("sports_science").home_topic == "sports"
    assert field_by_name("narrative_and_visual_arts").label == "narrative and visual arts"
    assert all(" " not in f.name and f.name == f.name.lower() for f in FIELDS)


def test_topic_groups_cover_the_taxonomy_once():
    flat = [t for group in TOPIC_GROUPS for t in group]
    assert sorted(flat) == sorted(TOPIC_IDS) and len(flat) == len(set(flat)) == 15
    assert len(TOPIC_GROUPS) == len(GROUP_NAMES) == 4
    assert TOPIC_GROUPS[0] == ("science_research", "health_medicine", "space_astronomy", "natural_disasters_climate",
                               "accidents")
    assert TOPIC_GROUPS[1] == ("ai_ml", "computing_software", "business_economy")
    assert TOPIC_GROUPS[3] == ("sports", "arts_entertainment")
    assert group_of("elections") == 2 and group_of("nope") is None


@pytest.mark.parametrize("home, current, tier", [
    ("science_research", "science_research", "near"),
    ("health_medicine", "accidents", "near"),
    ("computing_software", "science_research", "mid"),
    ("business_economy", "government", "mid"),
    ("government", "sports", "mid"),
    ("international_relations", "science_research", "far"),
    ("arts_entertainment", "science_research", "far"),
    ("sports", "ai_ml", "far"),
    ("science_research", "sports", "far"),
])
def test_tier_of(home, current, tier):
    assert tier_of(home, current) == tier == tier_of(current, home)


def test_tier_of_unknown_topics_is_mid():
    assert tier_of("science_research", None) == "mid"
    assert tier_of("science_research", "not_a_topic") == "mid"
    assert tier_of("not_a_topic", "science_research") == "mid"


def test_fields_by_tier_for_a_science_question_matches_design():
    by = {tier: [f.name for f in fs] for tier, fs in fields_by_tier("science_research").items()}
    assert by["near"] == ["thermodynamics", "materials_science", "chemistry", "evolutionary_biology", "immunology",
                          "epidemiology", "ecology", "geoscience_climate", "safety_engineering", "orbital_mechanics"]
    assert by["mid"] == ["control_theory", "information_theory", "machine_learning", "economics"]
    assert by["far"] == ["game_theory", "military_logistics", "political_institutions", "electoral_systems",
                         "law_and_evidence", "linguistics", "music_theory", "narrative_and_visual_arts",
                         "sports_science"]
    without = fields_by_tier("science_research", {"economics", "Thermodynamics"})
    assert "economics" not in [f.name for f in without["mid"]] and "thermodynamics" not in [f.name for f in without["near"]]
    assert set(fields_by_tier(None)) == set(TIERS) and fields_by_tier(None)["mid"] == list(FIELDS)


def test_tier_weights():
    assert tier_weights(DEFAULT_TIERS) == {"mid": 0.55, "far": 0.35, "near": 0.10}
    hot = tier_weights(DEFAULT_TIERS, stagnating=True)
    assert hot["far"] == pytest.approx(0.5) and sum(hot.values()) == pytest.approx(1.0)
    assert hot["mid"] == pytest.approx(0.55 / 0.65 * 0.5) and hot["near"] == pytest.approx(0.10 / 0.65 * 0.5)
    assert tier_weights({"far": 0.9, "mid": 0.1}, stagnating=True) == {"mid": 0.1, "far": 0.9, "near": 0.0}  # already far
    assert tier_weights({"mid": 2}) == {"mid": 2.0, "far": 0.0, "near": 0.0}
    assert tier_weights({"near": 1.0}, stagnating=True)["far"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        tier_weights({"mid": 0, "far": 0})


def test_draw_lens_is_deterministic_for_a_seed():
    a = draw_lens(random.Random("7:3"), "science_research", set(), DEFAULT_TIERS)
    b = draw_lens(random.Random("7:3"), "science_research", set(), DEFAULT_TIERS)
    assert a == b and a is not None
    field, tier, concepts = a
    assert isinstance(field, Field) and tier == tier_of(field.home_topic, "science_research")
    assert len(concepts) == 3 and len(set(concepts)) == 3 and set(concepts) <= set(field.concepts)
    # other seeds give other fields: the draw is not degenerate
    fields = {draw_lens(random.Random(f"7:{i}"), "science_research", set(), DEFAULT_TIERS)[0].name for i in range(60)}
    assert len(fields) >= 8
    # the stagnation flag changes the tier mixture, not the determinism
    assert draw_lens(random.Random(5), "science_research", set(), DEFAULT_TIERS, stagnating=True) == draw_lens(
        random.Random(5), "science_research", set(), DEFAULT_TIERS, stagnating=True)


def test_draw_lens_respects_tiers_and_used_fields():
    only_near = [draw_lens(random.Random(i), "science_research", set(), {"near": 1.0}) for i in range(30)]
    assert all(tier == "near" and group_of(f.home_topic) == 0 for f, tier, _ in only_near)
    only_far = [draw_lens(random.Random(i), "science_research", set(), {"far": 1.0}) for i in range(30)]
    assert all(tier == "far" and group_of(f.home_topic) >= 2 for f, tier, _ in only_far)
    # without replacement: drawing with every field used so far never repeats and ends with None
    used: set[str] = set()
    for i in range(len(FIELDS)):
        drawn = draw_lens(random.Random(i), "elections", used, DEFAULT_TIERS)
        assert drawn is not None and drawn[0].name not in used
        used.add(drawn[0].name)
    assert used == {f.name for f in FIELDS}
    assert draw_lens(random.Random(0), "elections", used, DEFAULT_TIERS) is None
    # a tier with no unused field falls back to the next-heaviest tier
    near_used = {f.name for f in FIELDS if tier_of(f.home_topic, "science_research") == "near"}
    for i in range(30):
        f, tier, _ = draw_lens(random.Random(i), "science_research", near_used, {"near": 1.0, "mid": 0.0, "far": 0.0})
        assert tier == "mid" and f.name not in near_used


def test_draw_lens_tier_mixture_follows_the_weights():
    n = 3000
    tiers = Counter(draw_lens(random.Random(i), "science_research", set(), DEFAULT_TIERS)[1] for i in range(n))
    assert abs(tiers["mid"] / n - 0.55) < 0.04 and abs(tiers["far"] / n - 0.35) < 0.04 and abs(tiers["near"] / n - 0.10) < 0.03
    hot = Counter(draw_lens(random.Random(i), "science_research", set(), DEFAULT_TIERS, stagnating=True)[1]
                  for i in range(n))
    assert abs(hot["far"] / n - 0.50) < 0.04


def test_field_lookup_and_normalisation():
    assert normalise_field("Materials Science") == "materials_science"
    assert normalise_field("**materials-science**") == "materials_science"
    assert normalise_field(" Narrative and Visual Arts ") == "narrative_and_visual_arts"
    assert normalise_field("Quantum Cooking") == "quantum_cooking"
    assert field_by_name("Materials Science") is field_by_name("materials_science")
    with pytest.raises(KeyError, match="unknown lens field"):
        field_by_name("quantum cooking")
    assert lenses.STAGNATION_FAR_WEIGHT == 0.5 and TIERS == ("mid", "far", "near")
