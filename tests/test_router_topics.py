import math
import statistics
import sys
import time

import numpy as np
import pytest

from cutoffmix.router.topics import (
    EmbeddingTopicClassifier,
    KeywordTopicClassifier,
    LinearTopicClassifier,
    detect_language,
    load_classifier,
    stem,
)
from cutoffmix.router.train import synthetic_topic_corpus
from cutoffmix.taxonomy import TOPIC_IDS
from cutoffmix.types import TopicClassifier

KW = KeywordTopicClassifier()

MATH_LOGIC_TEXTS = [
    "Let n be an integer. If n^2 is even then n is even. Suppose for contradiction that n is odd, so n = 2k+1 and "
    "n^2 = 4k^2+4k+1, which is odd. Therefore the claim holds. The final answer is 5050.",
    "Sei n eine ganze Zahl. Wenn n² gerade ist, dann ist auch n gerade. Angenommen n wäre ungerade, dann wäre "
    "n = 2k+1 und n² ungerade. Zum Beispiel stimmt das für n = 3. Also ist die Aussage bewiesen.",
    "Alice, Bob and Carol each own exactly one pet: a cat, a dog or a fish. Alice does not own the dog. Bob owns "
    "the fish. So Carol owns the dog and Alice owns the cat.",
    "We need the sum of the series 1 + 1/2 + 1/4 + ... which converges to 2. The kernel of the matrix has "
    "dimension 1, so by rank-nullity the rank is 2. The final result matches.",
    "Knights always tell the truth and knaves always lie. A says: 'B is a knave.' If A were a knave, B would be a "
    "knight, a contradiction. FINAL ANSWER: A is a knight.",
]


def _top(clf, text):
    p = clf.posterior(text)
    return clf.topics[int(np.argmax(p))], max(p)


def _evidence(text, topic):
    return KW.evidence(text)[KW.topics.index(topic)]


def _norm_entropy(p):
    return -sum(x * math.log(x) for x in p if x > 0) / math.log(len(p))


def test_keyword_classifier_implements_protocol():
    assert isinstance(KW, TopicClassifier)
    assert KW.topics == list(TOPIC_IDS)
    p = KW.posterior("The ceasefire collapsed after the drone strike.")
    assert len(p) == len(TOPIC_IDS) and math.isclose(sum(p), 1.0) and min(p) > 0


def test_uniform_without_signal():
    for text in ("", "   ", "the and of 42"):
        p = KW.posterior(text)
        assert max(p) - min(p) < 1e-12


@pytest.mark.parametrize("text", MATH_LOGIC_TEXTS)
def test_near_uniform_for_math_and_logic(text):
    p = KW.posterior(text)
    assert _norm_entropy(p) > 0.95
    assert max(p) < 0.15


def test_whole_word_matching():
    assert _evidence("The software update fixed a warm reboot bug.", "conflict_security") == 0
    assert _evidence("Ware and awareness, swarm, wartime-free.", "conflict_security") == 0
    assert _evidence("The war escalated.", "conflict_security") > 0
    assert _evidence("Two wars ended.", "conflict_security") > 0
    assert _top(KW, "GPT-6.2 Sol is the newest model.")[0] == "ai_ml"
    assert _evidence("Gpt-6.2 and GPT6 notes", "ai_ml") > 0  # case-insensitive, hyphen is a boundary


def test_case_sensitive_acronyms_and_false_friends():
    assert _evidence("Who is in charge here?", "health_medicine") == 0
    assert _top(KW, "The WHO declared an mpox outbreak.")[0] == "health_medicine"
    # German "war" means "was" and must not count as armed conflict
    assert _evidence("Er war Präsident von Frankreich und sie waren dabei.", "conflict_security") == 0
    assert _top(KW, "Er war Präsident von Frankreich.")[0] == "government"


def test_german_and_english_topics():
    cases = {
        "Wer ist aktuell Bundeskanzler?": "government",
        "Bei der Landtagswahl in Bayern gewann die CSU.": "elections",
        "Der Bürgerkrieg im Sudan dauert an.": "conflict_security",
        "Der Außenminister traf seinen Amtskollegen beim G7-Gipfel.": "international_relations",
        "Who is the current CEO of Acme?": "business_economy",
        "SpaceX launched Starship; NASA's Artemis program depends on it.": "space_astronomy",
        "The model was fine-tuned with a larger context window.": "ai_ml",
        "Who won the 2026 World Cup final?": "sports",
        "Die Bundeskanzlerin wurde vereidigt.": "government",
    }
    for text, topic in cases.items():
        assert _top(KW, text)[0] == topic, text


def test_prefix_keywords():
    assert _evidence("They are fine-tuning the model.", "ai_ml") > 0
    assert _evidence("COP30 ended in Belem.", "natural_disasters_climate") > 0


def test_more_evidence_sharpens_posterior():
    one = _top(KW, "The court ruled.")[1]
    many = _top(KW, "The court ruled; the judge sentenced him after the trial; prosecutors will appeal the verdict.")[1]
    assert many > one


def test_subset_of_topics_and_validation():
    clf = KeywordTopicClassifier(["government", "sports"])
    assert clf.topics == ["government", "sports"]
    assert _top(clf, "The coach won the championship.")[0] == "sports"
    with pytest.raises(KeyError):
        KeywordTopicClassifier(["no_such_topic"])
    with pytest.raises(ValueError):
        KeywordTopicClassifier(prior=0)
    with pytest.raises(ValueError):
        KeywordTopicClassifier(sharpness=-1)


def test_stemmer_and_language_detection():
    assert stem("wars") == "war"
    assert stem("waren") == "waren" and stem("ware") == "ware" and stem("warm") == "warm"
    assert stem("vaccines") == stem("vaccine")
    assert stem("kanzlerin") == stem("kanzler")
    assert stem("virus") == "virus" and stem("viruses") == "virus"
    assert detect_language(["the", "war", "in", "gaza", "is", "over"]) == "en"
    assert detect_language(["the", "war"]) == ""  # undecided: no false-friend filtering, "war" still counts
    assert _evidence("The war", "conflict_security") > 0
    assert detect_language(["er", "war", "der", "präsident"]) == "de"
    assert detect_language(["gpt"]) == ""


def test_keyword_posterior_is_fast():
    window = (
        "The front line near the city shifted after the offensive. A drone strike hit the air base overnight. "
        "Der Waffenstillstand wurde nach dem Angriff gebrochen. GPT-6.2 Sol released a new version; the "
        "Bundesregierung reagierte. Researchers published in Nature that the species is new. "
    ) * 10
    window = window[:2000]
    assert len(window) == 2000
    KW.posterior(window)  # warm caches
    timings = []
    for _ in range(30):
        t0 = time.perf_counter()
        KW.posterior(window)
        timings.append(time.perf_counter() - t0)
    assert statistics.median(timings) < 0.005


def test_linear_classifier_trains_on_synthetic_corpus(tmp_path):
    texts, labels = synthetic_topic_corpus(per_topic=30, seed=0)
    clf = LinearTopicClassifier.train(texts, labels, dim=2**16, epochs=8, seed=0)
    assert isinstance(clf, TopicClassifier)
    assert clf.topics == list(TOPIC_IDS)
    accuracy = sum(clf.predict(t) == y for t, y in zip(texts, labels)) / len(texts)
    assert accuracy > 0.8
    p = clf.posterior("Der Waffenstillstand wurde nach dem Angriff gebrochen.")
    assert math.isclose(sum(p), 1.0)

    path = tmp_path / "sub" / "topic.npz"
    clf.save(path)
    assert path.exists()
    loaded = LinearTopicClassifier.load(path)
    assert loaded.topics == clf.topics
    for text in texts[:5]:
        assert np.allclose(loaded.posterior(text), clf.posterior(text))


def test_linear_classifier_validation(tmp_path):
    with pytest.raises(ValueError):
        LinearTopicClassifier.train(["a", "b"], ["sports"])
    with pytest.raises(ValueError):
        LinearTopicClassifier.train(["a", "b"], ["sports", "sports"])
    bogus = tmp_path / "bogus.npz"
    np.savez(bogus, kind=np.asarray("something_else"))
    with pytest.raises(ValueError):
        LinearTopicClassifier.load(bogus)


def test_load_classifier(tmp_path):
    assert isinstance(load_classifier("auto"), KeywordTopicClassifier)
    assert isinstance(load_classifier("auto", str(tmp_path / "missing.npz")), KeywordTopicClassifier)
    assert isinstance(load_classifier("keyword"), KeywordTopicClassifier)
    texts, labels = synthetic_topic_corpus(per_topic=5)
    path = tmp_path / "topic.npz"
    LinearTopicClassifier.train(texts, labels, dim=4096, epochs=2).save(path)
    assert isinstance(load_classifier("auto", str(path)), LinearTopicClassifier)
    assert isinstance(load_classifier("linear", str(path)), LinearTopicClassifier)
    with pytest.raises(ValueError):
        load_classifier("linear")
    with pytest.raises(FileNotFoundError):
        load_classifier("linear", str(tmp_path / "missing.npz"))
    with pytest.raises(ValueError):
        load_classifier("nope")


def test_embedding_classifier_needs_optional_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ImportError, match="sentence-transformers"):
        EmbeddingTopicClassifier()
    with pytest.raises(ImportError):
        load_classifier("embedding")


class _BagOfWordsEncoder:
    """Deterministic stand-in for a SentenceTransformer: normalised hashed bag of words."""

    def encode(self, texts, normalize_embeddings=True):
        from cutoffmix.router.features import hashed_features

        out = np.zeros((len(texts), 256))
        for i, t in enumerate(texts):
            for k, v in hashed_features(t, 256).items():
                out[i, k] += v
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


def test_embedding_classifier_with_injected_encoder():
    clf = EmbeddingTopicClassifier(encoder=_BagOfWordsEncoder(), temperature=0.05)
    assert isinstance(clf, TopicClassifier)
    p = clf.posterior("The rocket launched the crew to the space station.")
    assert math.isclose(sum(p), 1.0)
    assert clf.topics[int(np.argmax(p))] == "space_astronomy"
    with pytest.raises(ValueError):
        EmbeddingTopicClassifier(encoder=_BagOfWordsEncoder(), temperature=0)
