"""Topic classifiers: text window -> posterior over the taxonomy topics.

* :class:`KeywordTopicClassifier` -- zero-training default built from the bilingual (EN/DE) taxonomy keywords and
  seed sentences. Matching is token based (whole words and phrases, never raw substrings), with light stemming
  (plurals, German case endings, feminine ``-in``), German compound heads (``Landtagswahl`` -> ``wahl``),
  case-sensitive acronyms (``WHO`` the organisation vs. the pronoun ``who``) and per-language false friends
  (German ``war`` = "was"). Generic words that also occur in math/logic text carry a low weight, so pure reasoning
  text gets a near-uniform posterior.
* :class:`LinearTopicClassifier` -- multinomial logistic regression on hashed features (numpy, CPU), ``.npz`` I/O.
* :class:`EmbeddingTopicClassifier` -- optional sentence-transformers prototype classifier.
* :func:`load_classifier` -- pick one from a config ``kind``/``path``.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cutoffmix.router.features import (
    DEFAULT_DIM,
    WORD_RE,
    SparseSoftmaxRegression,
    hashed_features,
    softmax,
    tokenize,
)
from cutoffmix.taxonomy import TOPIC_IDS, TOPICS, Topic, get_topic
from cutoffmix.types import TopicClassifier

__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "EmbeddingTopicClassifier",
    "KeywordTopicClassifier",
    "LinearTopicClassifier",
    "detect_language",
    "load_classifier",
    "stem",
]

# --------------------------------------------------------------------------------------------------------------------
# Lexical normalisation
# --------------------------------------------------------------------------------------------------------------------

# (suffix, minimum length of what remains). First matching rule wins; applied identically to keywords and text.
_SUFFIX_RULES: tuple[tuple[str, int], ...] = (
    ("ings", 4), ("ing", 4), ("ed", 4), ("es", 4), ("en", 4), ("in", 5), ("e", 4), ("s", 3),
)


@functools.lru_cache(maxsize=65536)
def stem(token: str) -> str:
    """Strip one inflectional suffix from a lowercase token (EN/DE light stemmer).

    Keeps short roots intact so that ``wars -> war`` but ``waren``, ``ware`` and ``warm`` stay distinct from ``war``.
    A final ``s`` is kept after ``s``/``u``/``i`` (``virus``, ``press``, ``analysis``).
    """
    for suffix, min_rest in _SUFFIX_RULES:
        if token.endswith(suffix) and len(token) - len(suffix) >= min_rest:
            if suffix == "s" and token[-2] in "sui":
                continue
            return token[: -len(suffix)]
    return token


_EN_FUNCTION_WORDS = frozenset(
    "the and of to is are was were that this with for on as by it be has have from which who what at not but or its "
    "their they he she we would can been than then there these those also after before when how".split()
)
_DE_FUNCTION_WORDS = frozenset(
    "der die das und ist sind war waren wurde wurden nicht mit ein eine einen einem einer den dem des von zu auf für "
    "im auch sich wer wie nach bei aus noch aber oder wird werden hat haben dass welche welcher seit vor über unter "
    "zum zur am es sie er wir ich".split()
)


def detect_language(tokens: Sequence[str]) -> str:
    """Return ``"en"``, ``"de"`` or ``""`` (undecided) from function-word counts of lowercase ``tokens``."""
    en = sum(1 for t in tokens if t in _EN_FUNCTION_WORDS)
    de = sum(1 for t in tokens if t in _DE_FUNCTION_WORDS)
    if en > de:
        return "en"
    if de > en:
        return "de"
    return ""


# --------------------------------------------------------------------------------------------------------------------
# Keyword lexicon
# --------------------------------------------------------------------------------------------------------------------

# Acronyms that are ordinary words in lowercase: they only match in upper case in the original text.
_CASED_ACRONYMS: dict[str, str] = {"who": "WHO", "iss": "ISS", "esa": "ESA", "ema": "EMA", "ki": "KI", "ai": "AI"}
# Taxonomy keywords that are word prefixes ("fine-tun" -> fine-tuning/fine-tuned, "cop3" -> COP30/COP31).
_PREFIX_KEYWORDS = frozenset({"fine-tun", "cop3"})
# Additional terms the taxonomy lacks (EN "president", German compounds, sport names, ...).
_EXTRA_KEYWORDS: dict[str, tuple[str, ...]] = {
    "conflict_security": ("war crimes", "kriegsverbrechen", "warfare", "drohnenangriff", "raketenangriff"),
    "government": ("president", "presidency", "premier", "parliament", "congress", "senate", "senator", "bundestag",
                   "bundesrat", "mayor", "bürgermeister", "regierungschef", "bundesregierung", "kanzleramt"),
    "elections": ("wahlkampf", "wahlergebnis", "wahlsieger", "exit poll", "campaign trail"),
    "natural_disasters_climate": ("climate change", "klimawandel"),
    "business_economy": ("chief executive officer", "vorstandsvorsitzender", "vorstandsvorsitzende", "konzernchef",
                         "firmenchef", "geschäftsführer", "stock market", "wall street", "nasdaq"),
    "ai_ml": ("ai", "ki", "ai model", "ai lab", "ai safety", "chatgpt", "hugging face"),
    "sports": ("football", "soccer", "fußball", "tennis", "basketball", "baseball", "cricket", "rugby", "marathon",
               "head coach", "cheftrainer", "bundestrainer", "nationalmannschaft"),
}
# Words that occur in non-news reasoning text (math, logic, code, everyday language): weak evidence only.
_GENERIC_WORDS = frozenset("""
final match series season title single release version library framework attack fire collapse collapsed accident
study nature species discovery discovered primary vote votes stock shares company revenue profit inference benchmark
launch orbit chip appeal trial judge king queen manager coach director author novel festival museum premiere
streaming treatment patients hospital physics biology chemistry quantum medal league transfer tournament tour band
api kernel windows pixel rust react hack browser chrome start spiel finale serie stimmen spieler trainer gewinn
umsatz negotiations investigation charged appointed succeeded resigned candidate polls cancer virus carbon climate
fusion inflation earnings investor valuation gpu moon satellite rocket mars front agreement meister turnier verein
roman autor
""".split())
# Function-like words of other languages that collide with a keyword: ignored as single tokens in that language.
_FALSE_FRIENDS: dict[str, frozenset[str]] = {
    "de": frozenset({"war"}),
    "en": frozenset({"front", "brand", "start", "roman", "spiel"}),
}
# Phrases that contain keywords but are about the reasoning itself: consumed with zero weight.
_STOP_PHRASES: tuple[str, ...] = (
    "final answer", "final result", "final step", "final value", "finale antwort", "endgültige antwort",
)
# German compounds: a long token whose stem ends with a (non-generic, >= 5 letter) keyword stem counts for that
# keyword's topics ("Bürgerkrieg" -> krieg). Short heads allowed explicitly; known false compounds blocked.
_COMPOUND_MIN_HEAD = 5
_COMPOUND_SHORT_HEADS = frozenset({"wahl"})
_COMPOUND_BLOCK = frozenset({"auswahl", "beispiel", "vorurteil"})
_COMPOUND_WEIGHT = 0.8
_GENERIC_WEIGHT = 0.2
# Seed sentences contribute their long content words (>= 8 letters: specific nouns such as "Landtagswahl",
# "superconductor"), minus long but generic words that happen to occur in them.
_SEED_WEIGHT = 0.25
_SEED_MIN_LEN = 8
_SEED_STOPWORDS = frozenset("""
according announced approved authorities bewertet building condition conference destroyed eingestuft erreichte
erweitert everyone gebrochen language material menschen overnight predecessor previous previously reasoning tausende
unterstützt
""".split())

Hits = tuple[tuple[int, float], ...]


@dataclass(frozen=True)
class _Lexicon:
    phrases: dict[tuple[str, ...], Hits]  # stem tuple -> hits (empty tuple = stop phrase)
    max_len: int
    cased: dict[str, Hits]  # exact original-case token -> hits
    prefixes: tuple[tuple[tuple[str, ...], str, Hits], ...]  # (leading stems, prefix of the last token, hits)
    compounds: dict[str, Hits]  # compound head stem -> hits (weights already scaled)


def _stems(phrase: str) -> tuple[str, ...]:
    return tuple(stem(t) for t in tokenize(phrase))


def _content_words(sentence: str) -> set[str]:
    return {stem(t) for t in tokenize(sentence) if len(t) >= _SEED_MIN_LEN and t.isalpha() and t not in _SEED_STOPWORDS}


@functools.lru_cache(maxsize=8)
def _build_lexicon(topics: tuple[Topic, ...]) -> _Lexicon:
    phrase_hits: dict[tuple[str, ...], dict[int, float]] = {}
    cased_hits: dict[str, dict[int, float]] = {}
    prefix_hits: dict[tuple[tuple[str, ...], str], dict[int, float]] = {}
    generic = {stem(w) for w in _GENERIC_WORDS}

    def put(table: dict[Any, dict[int, float]], key: Any, ti: int, w: float) -> None:
        slot = table.setdefault(key, {})
        slot[ti] = max(slot.get(ti, 0.0), w)

    for ti, topic in enumerate(topics):
        for raw in topic.keywords + _EXTRA_KEYWORDS.get(topic.id, ()):
            kw = raw.strip().lower()
            if kw in _CASED_ACRONYMS:
                put(cased_hits, _CASED_ACRONYMS[kw], ti, 1.0)
            elif kw in _PREFIX_KEYWORDS:
                toks = tokenize(kw)
                put(prefix_hits, (tuple(stem(t) for t in toks[:-1]), toks[-1]), ti, 1.0)
            else:
                key = _stems(kw)
                if key:
                    put(phrase_hits, key, ti, _GENERIC_WEIGHT if len(key) == 1 and key[0] in generic else 1.0)
    compounds: dict[str, Hits] = {}
    for key, hits in phrase_hits.items():
        head = key[0]
        if len(key) > 1 or head in generic or not head.isalpha():
            continue
        if len(head) >= _COMPOUND_MIN_HEAD or head in _COMPOUND_SHORT_HEADS:
            compounds[head] = tuple((ti, w * _COMPOUND_WEIGHT) for ti, w in hits.items())
    # seed sentences: content words unique to one topic's seeds that no keyword or compound rule covers already
    keyword_stems = {k[0] for k in phrase_hits if len(k) == 1}
    seed_words = [
        {
            w for w in _content_words(" ".join(t.seeds)) - keyword_stems - generic
            if _compound_head(w, compounds) is None
        }
        for t in topics
    ]
    for ti, words in enumerate(seed_words):
        others = set().union(*(w for j, w in enumerate(seed_words) if j != ti))
        for word in words - others:
            put(phrase_hits, (word,), ti, _SEED_WEIGHT)
    for phrase in _STOP_PHRASES:
        phrase_hits[_stems(phrase)] = {}

    def freeze(d: dict[int, float]) -> Hits:
        return tuple(sorted(d.items()))

    return _Lexicon(
        phrases={k: freeze(v) for k, v in phrase_hits.items()},
        max_len=max(len(k) for k in phrase_hits),
        cased={k: freeze(v) for k, v in cased_hits.items()},
        prefixes=tuple((lead, last, freeze(v)) for (lead, last), v in prefix_hits.items()),
        compounds=compounds,
    )


def _compound_head(word: str, compounds: dict[str, Hits]) -> str | None:
    """Longest compound head that ``word`` (a stem) ends with, leaving at least 3 letters of modifier."""
    if len(word) < 7 or word in _COMPOUND_BLOCK:
        return None
    for k in range(len(word) - 3, 3, -1):
        if word[-k:] in compounds:
            return word[-k:]
    return None


def _resolve_topics(topics: Iterable[Topic | str]) -> tuple[Topic, ...]:
    resolved = tuple(t if isinstance(t, Topic) else get_topic(t) for t in topics)
    if not resolved:
        raise ValueError("at least one topic is required")
    return resolved


class KeywordTopicClassifier:
    """Zero-training topic classifier from taxonomy keywords and seed sentences (implements ``TopicClassifier``).

    For each topic ``z`` the evidence is ``s_z = Σ_k w_k (1 + ln c_k)`` over matched keywords ``k`` with weight
    ``w_k`` (1 for specific keywords and phrases, 0.2 for generic words, 0.25 for seed-sentence words, 0.8 for German
    compound heads) and ``c_k`` occurrences. The posterior is ``p_z ∝ (prior + s_z) ** sharpness``: uniform when no
    keyword matches, sharper with more evidence. Matching is longest-phrase-first over word tokens, so ``war`` never
    matches inside ``software`` or ``warm`` while ``gpt`` matches ``GPT-6.2``.
    """

    def __init__(self, topics: Iterable[Topic | str] = TOPICS, *, prior: float = 0.3, sharpness: float = 1.5):
        if prior <= 0:
            raise ValueError(f"prior must be > 0, got {prior}")
        if sharpness <= 0:
            raise ValueError(f"sharpness must be > 0, got {sharpness}")
        resolved = _resolve_topics(topics)
        self.topics: Sequence[str] = [t.id for t in resolved]
        self.prior = prior
        self.sharpness = sharpness
        self._lex = _build_lexicon(resolved)

    def evidence(self, text: str) -> list[float]:
        """Raw keyword evidence ``s_z`` per topic (same order as :attr:`topics`); all zero if nothing matches."""
        lex = self._lex
        original = WORD_RE.findall(text)
        n = len(original)
        scores = [0.0] * len(self.topics)
        if not n:
            return scores
        lower = [t.lower() for t in original]
        stems = [stem(t) for t in lower]
        blocked = _FALSE_FRIENDS.get(detect_language(lower), frozenset())
        counts: dict[Any, int] = {}
        hits_of: dict[Any, Hits] = {}

        def record(key: Any, hits: Hits) -> None:
            counts[key] = counts.get(key, 0) + 1
            hits_of[key] = hits

        i = 0
        while i < n:
            step = 0
            for length in range(min(lex.max_len, n - i), 1, -1):
                key = tuple(stems[i : i + length])
                hits = lex.phrases.get(key)
                if hits is not None:
                    record(key, hits)
                    step = length
                    break
            if not step:
                step = 1
                cased = lex.cased.get(original[i])
                single = lex.phrases.get((stems[i],))
                if cased is not None:
                    record(original[i], cased)
                elif single is not None:
                    if stems[i] not in blocked:
                        record((stems[i],), single)
                else:
                    self._match_prefix_or_compound(lower, stems, i, record)
            i += step
        for key, c in counts.items():
            damp = 1.0 + math.log(c)
            for ti, w in hits_of[key]:
                scores[ti] += w * damp
        return scores

    def _match_prefix_or_compound(
        self, lower: list[str], stems: list[str], i: int, record: Callable[[Any, Hits], None]
    ) -> None:
        lex = self._lex
        for lead, last, hits in lex.prefixes:
            j = i + len(lead)
            if j < len(lower) and tuple(stems[i:j]) == lead and lower[j].startswith(last):
                record(("prefix", lead, last), hits)
                return
        if lower[i] in _COMPOUND_BLOCK:
            return
        head = _compound_head(stems[i], lex.compounds)
        if head is not None:
            record(("compound", head), lex.compounds[head])

    def posterior(self, text: str) -> list[float]:
        """``p(topic | text)`` over :attr:`topics` (non-negative, sums to 1)."""
        s = np.asarray(self.evidence(text), dtype=np.float64)
        w = np.power(self.prior + s, self.sharpness)
        return (w / w.sum()).tolist()


# --------------------------------------------------------------------------------------------------------------------
# Learned classifiers
# --------------------------------------------------------------------------------------------------------------------


class LinearTopicClassifier:
    """Multinomial logistic regression on :func:`~cutoffmix.router.features.hashed_features` (numpy, CPU)."""

    _KIND = "topic_linear"

    def __init__(self, topics: Sequence[str], model: SparseSoftmaxRegression):
        if len(topics) != model.n_classes:
            raise ValueError(f"{len(topics)} topics but the model has {model.n_classes} classes")
        self.topics: Sequence[str] = list(topics)
        self.model = model

    @classmethod
    def train(
        cls,
        texts: Sequence[str],
        labels: Sequence[str],
        *,
        dim: int = DEFAULT_DIM,
        epochs: int = 8,
        lr: float = 0.5,
        l2: float = 1e-6,
        seed: int = 0,
        batch_size: int = 32,
    ) -> LinearTopicClassifier:
        """Fit on ``texts`` labelled with topic ids.

        The topic order follows the taxonomy for known ids, then any other label in sorted order. At least two
        distinct labels are needed.
        """
        if len(texts) != len(labels):
            raise ValueError(f"texts and labels differ in length: {len(texts)} != {len(labels)}")
        present = set(labels)
        topics = [t for t in TOPIC_IDS if t in present] + sorted(present - set(TOPIC_IDS))
        if len(topics) < 2:
            raise ValueError(f"need at least two distinct topic labels, got {topics}")
        index = {t: i for i, t in enumerate(topics)}
        X = [hashed_features(t, dim) for t in texts]
        y = [index[label] for label in labels]
        model = SparseSoftmaxRegression(dim, len(topics)).fit(
            X, y, epochs=epochs, lr=lr, l2=l2, batch_size=batch_size, seed=seed
        )
        return cls(topics, model)

    def posterior(self, text: str) -> list[float]:
        """``p(topic | text)`` over :attr:`topics`."""
        return self.model.predict_proba(hashed_features(text, self.model.dim)).tolist()

    def predict(self, text: str) -> str:
        """Most probable topic id."""
        return self.topics[int(np.argmax(self.model.logits(hashed_features(text, self.model.dim))))]

    def save(self, path: str | Path) -> None:
        """Write a compressed ``.npz`` (no pickles) to exactly ``path``."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            np.savez_compressed(
                f, kind=np.asarray(self._KIND), topics=np.asarray(self.topics), **self.model.to_arrays()
            )

    @classmethod
    def load(cls, path: str | Path) -> LinearTopicClassifier:
        """Load a classifier written by :meth:`save`."""
        with np.load(Path(path), allow_pickle=False) as z:
            kind = str(z["kind"]) if "kind" in z.files else ""
            if kind != cls._KIND:
                raise ValueError(f"{path} is not a LinearTopicClassifier file (kind={kind!r})")
            return cls([str(t) for t in z["topics"]], SparseSoftmaxRegression.from_arrays(z))


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


class EmbeddingTopicClassifier:
    """Prototype classifier on sentence embeddings (optional dependency ``sentence-transformers``).

    Each topic's prototype is the normalised mean embedding of its name + description, its seed sentences and its
    keyword list. ``posterior`` is a softmax of cosine similarities divided by ``temperature``. ``encoder`` may be any
    object with ``encode(list[str], normalize_embeddings=True) -> array``; by default a ``SentenceTransformer`` is
    loaded from ``model_name`` (downloads the model on first use).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        topics: Iterable[Topic | str] = TOPICS,
        temperature: float = 0.05,
        encoder: Any | None = None,
    ):
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "EmbeddingTopicClassifier needs the optional dependency 'sentence-transformers' "
                    "(pip install 'cutoffmix[embed]'); use classifier 'keyword' or 'linear' otherwise"
                ) from e
            encoder = SentenceTransformer(model_name)
        resolved = _resolve_topics(topics)
        self.topics: Sequence[str] = [t.id for t in resolved]
        self.temperature = temperature
        self._encoder = encoder
        protos = []
        for t in resolved:
            texts = [f"{t.name}: {t.description}", *t.seeds, ", ".join(t.keywords)]
            emb = np.asarray(encoder.encode(texts, normalize_embeddings=True), dtype=np.float64)
            mean = emb.mean(axis=0)
            protos.append(mean / (np.linalg.norm(mean) or 1.0))
        self._prototypes = np.stack(protos)

    def posterior(self, text: str) -> list[float]:
        """``p(topic | text)`` from cosine similarity to the topic prototypes."""
        e = np.asarray(self._encoder.encode([text], normalize_embeddings=True), dtype=np.float64)[0]
        return softmax(self._prototypes @ e / self.temperature).tolist()


def load_classifier(kind: str = "auto", path: str | None = None) -> TopicClassifier:
    """Build the configured topic classifier.

    ``kind``: ``"auto"`` (linear if ``path`` is an existing file, else keyword), ``"keyword"``, ``"linear"`` (needs
    ``path`` to a ``.npz`` from :meth:`LinearTopicClassifier.save`) or ``"embedding"`` (``path`` = model name or
    local directory, default :data:`DEFAULT_EMBEDDING_MODEL`).
    """
    if kind == "auto":
        if path and Path(path).is_file():
            return LinearTopicClassifier.load(path)
        return KeywordTopicClassifier()
    if kind == "keyword":
        return KeywordTopicClassifier()
    if kind == "linear":
        if not path:
            raise ValueError("classifier 'linear' needs classifier_path pointing to a trained .npz file")
        if not Path(path).is_file():
            raise FileNotFoundError(f"linear topic classifier not found: {path}")
        return LinearTopicClassifier.load(path)
    if kind == "embedding":
        return EmbeddingTopicClassifier(path or DEFAULT_EMBEDDING_MODEL)
    raise ValueError(f"unknown classifier kind {kind!r}; expected auto, keyword, linear or embedding")
