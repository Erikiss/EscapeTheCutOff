import numpy as np
import pytest

from cutoffmix.router.gate import TimeSensitivityGate
from cutoffmix.router.train import build_gate_corpus

GATE = TimeSensitivityGate()

MATH_PUZZLES = [
    "If 3x + 5 = 20, what is x? Let us solve the equation step by step.",
    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
    "Knights always tell the truth and knaves always lie. A says: 'B is a knave.' B says: 'A and I are alike.'",
    "Prove that the sum of two odd integers is even: (2a+1) + (2b+1) = 2(a+b+1).",
    "Wie viele Primzahlen gibt es zwischen 10 und 30? Wir prüfen jede Zahl auf Teilbarkeit.",
    "Ein Zug fährt mit 80 km/h von A nach B, die 240 km entfernt sind. Wie lange dauert die Fahrt?",
    "What is the remainder when 2^100 is divided by 7?",
]
TIME_SENSITIVE = [
    "Who is the current CEO of Acme?",
    "Who is the current CEO of OpenAI?",
    "Wer ist aktuell Bundeskanzler?",
    "Wer ist der Bundeskanzler von Deutschland?",
    "Is Pope Leo XIV still alive?",
    "What is the latest version of Python?",
    "Welches ist das neueste Modell von OpenAI?",
    "Let me recall who the current chancellor of Germany is: after the 2025 election, Friedrich Merz took office.",
]


@pytest.mark.parametrize("text", MATH_PUZZLES)
def test_math_puzzles_score_low(text):
    assert GATE.score(text) < 0.2


@pytest.mark.parametrize("text", TIME_SENSITIVE)
def test_present_state_questions_score_high(text):
    assert GATE.score(text) > 0.6


def test_scores_are_probabilities_and_ordered():
    for text in ("", "x", "Who is the current CEO? " * 200, "1 + 1 = 2 " * 300):
        assert 0.0 <= GATE.score(text) <= 1.0
    news = GATE.score("The war in Ukraine escalated after the offensive near Kharkiv.")
    assert GATE.score("What is the boiling point of water at sea level?") < news < GATE.score(TIME_SENSITIVE[0])


def test_cues_report_the_lexical_evidence():
    c = GATE.cues("Who is the current CEO of Acme in 2026?")
    assert c["strong_temporal"] >= 1 and c["roles"] >= 1 and c["years"] == 1 and c["present_question"] == 1
    assert c["acronyms"] == 1 and c["news"] > 0 and c["math"] == 0
    assert GATE.cues("Wer ist der Bundeskanzler?")["entity_density"] == 0.0  # German nouns are capitalised anyway
    assert GATE.cues("Prove that 2 + 2 = 4.")["math"] > 1


def test_lexical_gate_cannot_be_saved(tmp_path):
    with pytest.raises(ValueError):
        GATE.save(tmp_path / "gate.npz")
    with pytest.raises(ValueError):
        TimeSensitivityGate(learned_weight=1.5)


def test_trained_gate_blends_and_round_trips(tmp_path):
    texts, labels = build_gate_corpus([], n_synthetic=150, seed=0)
    assert set(labels) == {0, 1}
    gate = TimeSensitivityGate.train(texts, labels, dim=2**15, epochs=6, learned_weight=0.7)
    assert gate.learned_weight == 0.7
    accuracy = np.mean([(gate.score(t) >= 0.5) == bool(y) for t, y in zip(texts, labels)])
    assert accuracy > 0.9
    text = "Wer ist derzeit Trainer des Vereins?"
    learned = float(gate.model.predict_proba(__import__("cutoffmix.router.features", fromlist=["x"]).hashed_features(
        text, gate.model.dim))[1])
    assert gate.score(text) == pytest.approx(0.7 * learned + 0.3 * gate.lexical_score(text))

    path = tmp_path / "gate.npz"
    gate.save(path)
    loaded = TimeSensitivityGate.load(path)
    assert loaded.learned_weight == pytest.approx(0.7)
    for t in texts[:5]:
        assert loaded.score(t) == pytest.approx(gate.score(t))


def test_gate_training_validation(tmp_path):
    with pytest.raises(ValueError):
        TimeSensitivityGate.train(["a", "b"], [1, 1])
    with pytest.raises(ValueError):
        TimeSensitivityGate.train(["a"], [1, 0])
    bogus = tmp_path / "bogus.npz"
    np.savez(bogus, kind=np.asarray("topic_linear"))
    with pytest.raises(ValueError):
        TimeSensitivityGate.load(bogus)
