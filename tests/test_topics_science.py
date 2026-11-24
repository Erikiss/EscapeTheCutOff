"""Research and technology questions get a clear topic from the zero-training keyword classifier, news questions
keep theirs, and a flat posterior names no topic (relay, CLI header, web UI) instead of faking an argmax."""

from __future__ import annotations

import io
import random
from pathlib import Path

import pytest

from cutoffmix.cli import TerminalPrinter
from cutoffmix.lenses import FIELDS, draw_lens, field_by_name, fields_by_tier, tier_of
from cutoffmix.relay import _argmax
from cutoffmix.router.topics import KeywordTopicClassifier
from cutoffmix.taxonomy import TOPIC_IDS, argmax_topic
from cutoffmix.types import ModelSpec, RelayEvent

KW = KeywordTopicClassifier()
MIN_CLEAR = 0.35  # one full keyword hit gives 0.39 over 15 topics; two give 0.60
TIERS = {"mid": 0.55, "far": 0.35, "near": 0.10}


def posterior(text: str) -> dict[str, float]:
    return dict(zip(KW.topics, KW.posterior(text)))


def top2(text: str) -> list[tuple[str, float]]:
    return sorted(posterior(text).items(), key=lambda kv: -kv[1])[:2]


# --------------------------------------------------------------------------------------
# Research / technology questions (the project's main use): a clear topic, in German and English
# --------------------------------------------------------------------------------------

RESEARCH: list[tuple[str, str]] = [
    # batteries and electric cars (the questions that used to come out uniform)
    ("Warum verlieren Lithium-Ionen-Akkus bei Kälte an Kapazität, und was folgt daraus für Elektroautos im Winter?",
     "science_research"),
    ("Why do lithium-ion batteries lose capacity in the cold, and what does that mean for electric cars in winter?",
     "science_research"),
    # physics, chemistry, materials
    ("Wie entsteht Supraleitung bei hohen Temperaturen?", "science_research"),
    ("What is the current record critical temperature for a superconductor at ambient pressure?",
     "science_research"),
    ("Welche Rolle spielen Elektronen und Photonen in einem Supraleiter?", "science_research"),
    ("Welche Legierungen eignen sich für Turbinenschaufeln bei hohen Temperaturen?", "science_research"),
    ("Why is graphene such a good electrical conductor, and why is it hard to open a band gap in it?",
     "science_research"),
    ("Wie entsteht Entropie in einem thermodynamischen System?", "science_research"),
    ("What is the difference between nuclear fusion and fission in terms of binding energy?", "science_research"),
    ("How do catalysts lower the activation energy of a chemical reaction?", "science_research"),
    ("Welcher Katalysator spaltet Wasser per Elektrolyse in Wasserstoff und Sauerstoff?", "science_research"),
    # biology, genetics, neuroscience
    ("How does AlphaFold predict protein folding from an amino acid sequence?", "science_research"),
    ("Wie faltet sich ein Protein in der Zelle, und warum ist die Proteinfaltung so schwer zu simulieren?",
     "science_research"),
    ("Wie funktioniert die Genschere CRISPR beim Editieren des Genoms?", "science_research"),
    ("How does CRISPR-Cas9 cut DNA at a target sequence?", "science_research"),
    ("How do neurons in the hippocampus encode spatial memory?", "science_research"),
    ("Wie kommunizieren Neuronen über Synapsen im Gehirn?", "science_research"),
    ("How do bacteria in the microbiome exchange genes by horizontal transfer?", "science_research"),
    # earth science, paleontology, mathematics, research practice
    ("What does the fossil record say about the evolution of feathered dinosaurs?", "science_research"),
    ("Wie datieren Geologen Sedimentschichten mit radioaktiven Isotopen?", "science_research"),
    ("Is the Riemann hypothesis closer to a proof after the recent preprint on arXiv?", "science_research"),
    ("Welche Mathematiker haben die Vermutung in der Zahlentheorie bewiesen, und wo wurde der Beweis publiziert?",
     "science_research"),
    # space and astronomy
    ("Wie weist LIGO Gravitationswellen nach?", "space_astronomy"),
    ("How many exoplanets has the James Webb telescope characterised so far?", "space_astronomy"),
    ("Wie viele Exoplaneten hat das James-Webb-Teleskop bisher charakterisiert?", "space_astronomy"),
    ("What happens to matter that falls through a black hole's event horizon?", "space_astronomy"),
    ("Was ist Dunkle Materie, und warum sehen wir sie nicht in Galaxien?", "space_astronomy"),
    ("Which rover is NASA sending to the lunar south pole, and what will it look for?", "space_astronomy"),
    # climate and environment
    ("How reliable are climate models at predicting sea level rise from permafrost methane?",
     "natural_disasters_climate"),
    ("Wie stark steigt der Meeresspiegel bis 2100, wenn der grönländische Eisschild weiter schmilzt?",
     "natural_disasters_climate"),
    ("What is the expected temperature rise under the IPCC scenarios, and which tipping points matter?",
     "natural_disasters_climate"),
    ("How much methane escapes from thawing permafrost in the Arctic?", "natural_disasters_climate"),
    ("Wie entsteht ein Hurrikan über dem Ozean, und warum werden Stürme stärker?", "natural_disasters_climate"),
    # AI / ML
    ("How does the attention mechanism in a transformer scale with context length?", "ai_ml"),
    ("Was ist der Unterschied zwischen einem neuronalen Netz und einem Sprachmodell?", "ai_ml"),
    ("Why does reinforcement learning from human feedback make a chatbot more helpful?", "ai_ml"),
    ("Wie funktioniert maschinelles Lernen mit Embeddings und Tokens?", "ai_ml"),
    ("How does deep learning differ from classical machine learning on tabular data?", "ai_ml"),
    ("Why do large language models hallucinate, and does retrieval-augmented generation help?", "ai_ml"),
    # computing and cryptography
    ("Is RSA encryption broken by a quantum computer with a few thousand logical qubits?", "computing_software"),
    ("Wie funktioniert die Verschlüsselung in einer Datenbank, und was ändert Post-Quantum-Kryptographie daran?",
     "computing_software"),
    ("How does a compiler turn source code into machine code, and what does the algorithm optimise?",
     "computing_software"),
    ("How does a quantum computer with superconducting qubits differ from a classical CPU?", "computing_software"),
    ("Welche Sicherheitslücke erlaubte den Cyberangriff auf das Rechenzentrum, und war Malware im Spiel?",
     "computing_software"),
    # medicine
    ("Wie wirken mRNA-Impfstoffe im Immunsystem?", "health_medicine"),
    ("How effective are the new malaria vaccines in children, and what did the clinical trials show?",
     "health_medicine"),
    ("Welche Gentherapien sind gegen Sichelzellanämie zugelassen, und was kosten sie?", "health_medicine"),
    ("What causes Alzheimer's disease at the molecular level, and are amyloid antibodies a therapy?",
     "health_medicine"),
    ("Wie verändert Insulinresistenz den Stoffwechsel bei Diabetes Typ 2?", "health_medicine"),
    ("Which antibiotics still work against resistant pathogens, and what do clinical trials show?",
     "health_medicine"),
    ("Wie wirkt eine Chemotherapie auf Krebszellen, und welche Nebenwirkungen hat sie?", "health_medicine"),
]

# News questions in the taxonomy's original style: their topics must not move.
NEWS: list[tuple[str, str]] = [
    ("Who won the 2026 World Cup final?", "sports"),
    ("Der Verein gewann die Meisterschaft am letzten Spieltag.", "sports"),
    ("Wer ist aktuell Bundeskanzler?", "government"),
    ("Who is the president of France right now?", "government"),
    ("Bei der Landtagswahl in Bayern gewann die CSU.", "elections"),
    ("The presidential election was decided in the runoff.", "elections"),
    ("Der Bürgerkrieg im Sudan dauert an.", "conflict_security"),
    ("The ceasefire collapsed after the drone strike near the front line.", "conflict_security"),
    ("The court ruled that the law was unconstitutional, and the judge sentenced him.", "law_crime"),
    ("Das Gericht verurteilte den Angeklagten zu fünf Jahren Haft.", "law_crime"),
    ("Who is the current CEO of Acme, and when was the IPO?", "business_economy"),
    ("Der Außenminister traf seinen Amtskollegen beim G7-Gipfel.", "international_relations"),
    ("Which film won Best Picture at the Oscars this year?", "arts_entertainment"),
    ("The singer's new album debuted at number one.", "arts_entertainment"),
    ("A gas explosion destroyed the building, and the death toll rose overnight.", "accidents"),
    ("Welches ist das neueste Modell von OpenAI?", "ai_ml"),
]


@pytest.mark.parametrize("text,topic", RESEARCH, ids=[t[:40] for t, _ in RESEARCH])
def test_research_questions_get_a_clear_topic(text, topic):
    (best, p), (second, q) = top2(text)
    assert best == topic, f"{text!r}: {best} {p:.2f} (then {second} {q:.2f})"
    assert p >= MIN_CLEAR, f"{text!r}: {best} only {p:.2f}"
    assert argmax_topic(posterior(text)) == topic


@pytest.mark.parametrize("text,topic", NEWS, ids=[t[:40] for t, _ in NEWS])
def test_news_questions_keep_their_topic(text, topic):
    (best, p), _ = top2(text)
    assert best == topic and p >= 0.25, f"{text!r}: {best} {p:.2f}"
    assert argmax_topic(posterior(text)) == topic


def test_battery_question_is_clearly_science_not_health_ai_or_computing():
    for text, _ in RESEARCH[:2]:
        post = posterior(text)
        assert post["science_research"] > 0.6
        assert post["business_economy"] < 0.15  # "Elektroauto" / "electric car" lean to business a little
        for other in ("health_medicine", "ai_ml", "computing_software", "conflict_security"):
            assert post[other] < 0.05, (text, other, post[other])


def test_gene_therapy_question_splits_between_science_and_health():
    # "CRISPR" is science, "Gentherapie" is medicine: a two-way split, not uniform and not a single fake winner
    post = posterior("Welche Rolle spielt CRISPR in der Gentherapie 2026?")
    (a, p), (b, q) = sorted(post.items(), key=lambda kv: -kv[1])[:2]
    assert {a, b} == {"science_research", "health_medicine"} and min(p, q) > 0.25
    assert argmax_topic(post) in (a, b)


def test_electric_vehicle_leans_to_science_and_business():
    post = posterior("Which EV has the longest range this year, and how fast does it charge at a charging station?")
    ranked = sorted(post, key=lambda t: -post[t])
    assert set(ranked[:2]) == {"science_research", "business_economy"}
    assert post["business_economy"] > post["science_research"]  # the charging station is industry vocabulary
    assert argmax_topic(post) == "business_economy"


def test_german_compounds_and_inflections():
    ev = KW.evidence
    sci = KW.topics.index("science_research")
    assert ev("Der Lithiumakku im E-Auto")[sci] >= 1.8  # "Lithiumakku" via the compound head "akku", "E-Auto"
    assert ev("Hochtemperatursupraleiter aus Kupferoxid")[sci] > 0.7  # compound head "supraleiter"
    assert ev("Proteine, Enzyme und Moleküle")[sci] >= 3.0  # German plurals stem to the keyword
    assert ev("proteins, enzymes and molecules")[sci] >= 3.0  # English plurals too


def test_ambiguous_words_stay_weak():
    """Words shared with everyday or reasoning language only nudge the posterior; a math text stays near flat."""
    for text in ("The cell in the spreadsheet holds the reaction of the market.",
                 "Let n be an integer; the algorithm returns the kernel of the matrix.",
                 "In theory the paper is due tomorrow."):
        assert max(KW.posterior(text)) < 0.25, text
    assert KW.evidence("Gen Z likes the new Apple Watch.")[KW.topics.index("science_research")] == 0  # EN "gen"
    assert KW.evidence("Eine Dose Bier, bitte.")[KW.topics.index("health_medicine")] == 0  # DE "Dose" = tin
    assert KW.evidence("Das Gen trägt eine Mutation.")[KW.topics.index("science_research")] >= 2.0  # DE "Gen"


# --------------------------------------------------------------------------------------
# A flat posterior names no topic
# --------------------------------------------------------------------------------------

UNIFORM = {t: 1.0 / len(TOPIC_IDS) for t in TOPIC_IDS}


def test_argmax_topic_on_flat_posteriors():
    assert argmax_topic({}) is None
    assert argmax_topic(UNIFORM) is None
    rounded = {t: round(p, 4) for t, p in UNIFORM.items()}  # the relay stores posteriors rounded to 4 decimals
    assert argmax_topic(rounded) is None
    near_flat = {t: (0.09 if t == "sports" else (1 - 0.09) / 14) for t in TOPIC_IDS}  # below 1.5x the smallest
    assert argmax_topic(near_flat) is None
    assert argmax_topic(posterior("the and of 42")) is None  # the classifier found nothing
    assert argmax_topic(posterior("Prove that the sum of two odd integers is even.")) is None


def test_argmax_topic_on_clear_posteriors():
    assert argmax_topic({"government": 0.7, "sports": 0.3}) == "government"
    assert argmax_topic({"science_research": 0.7}) == "science_research"  # one topic: that topic
    assert argmax_topic({"government": 1.0}) == "government"
    one_generic_hit = posterior("The kernel panicked.")  # 0.2-weight word: 0.13 vs 0.06, still a (weak) topic
    assert max(one_generic_hit.values()) == pytest.approx(0.133, abs=0.01)
    assert argmax_topic(one_generic_hit) == "computing_software"
    assert argmax_topic(posterior(RESEARCH[0][0])) == "science_research"
    assert _argmax(UNIFORM) is None and _argmax(posterior(RESEARCH[0][0])) == "science_research"


def test_lenses_with_no_current_topic():
    assert tier_of("science_research", None) == "mid"
    assert fields_by_tier(None)["mid"] == list(FIELDS) and not fields_by_tier(None)["near"]
    drawn = draw_lens(random.Random(0), None, (), TIERS)
    assert drawn is not None and drawn[1] == "mid"
    # with a science topic the tiers are relative to the science group
    assert tier_of(field_by_name("materials_science").home_topic, "science_research") == "near"
    assert tier_of(field_by_name("machine_learning").home_topic, "science_research") == "mid"
    assert tier_of(field_by_name("electoral_systems").home_topic, "science_research") == "far"


def test_cli_header_shows_topic_or_question_mark():
    printer = TerminalPrinter([ModelSpec(key="a", provider_id="x/a", display="Model A", color="#d97757")],
                              out=io.StringIO())

    def header(post):
        ev = RelayEvent(type="segment_start", segment=0, model="a", phase="think", decision="sample",
                        topic_posterior=post, data={})
        return printer._header(ev)[1]

    assert "topic ?" in header(UNIFORM)
    assert "science_research 71%" in header(posterior(RESEARCH[0][0]))
    assert "government 70%" in header({"government": 0.7, "sports": 0.3})
    assert "topic" not in header({})  # no posterior at all: nothing to say


async def test_mock_explore_run_labels_the_battery_question_science():
    """``cutoffmix run --mock --preset explore --seed 3 <battery question>`` without the mock's streaming delay:
    every think segment is science_research and the lens tiers are relative to the science group."""
    from cutoffmix.config import load_config
    from cutoffmix.relay import RelayRuntime

    runtime = RelayRuntime(load_config(preset="explore"), mock=True, seed=3, mock_latency_s=0.0)
    try:
        events = [ev async for ev in runtime.relay().run(RESEARCH[0][0])]
    finally:
        await runtime.aclose()
    done = events[-1].data
    assert events[-1].type == "done" and done["status"] == "ok"
    ex = done["explore"]
    assert ex["topics"] and set(ex["topics"]) == {"science_research"}
    assert ex["topics_visited"] == {"all": 1, "main_line": 1}
    for seg in done["segments"]:
        if seg["phase"] == "think":
            assert argmax_topic(seg["topic_posterior"]) == "science_research"
            assert seg["topic_posterior"]["science_research"] > 0.6
    assert ex["lens"]["offered"] >= 1
    for field, tier in zip(ex["lens"]["fields"], ex["lens"]["tiers"]):
        assert tier == tier_of(field_by_name(field).home_topic, "science_research"), (field, tier)


def test_web_ui_uses_the_same_flatness_rule():
    html = Path(__file__).resolve().parents[1].joinpath("cutoffmix", "web", "index.html").read_text("utf-8")
    assert "function clearTopic(post)" in html and "hi < 1.5 * lo" in html
    assert "const top = clearTopic(rec.posterior)" in html and '" · —"' in html  # segment label
    assert "S.windowTopic = clearTopic(rec.posterior)" in html and "No clear topic" in html  # window panel
