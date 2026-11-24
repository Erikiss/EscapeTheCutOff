"""Time-sensitivity gate r(W): how much the current window depends on recent world knowledge.

r(W) in [0, 1] sets the routing temperature (``T = T0 / max(r, r_min)``): time-insensitive windows (math, logic,
timeless facts) mix the models almost uniformly, time-sensitive windows ("Who is the current CEO of X?", "Wer ist
aktuell Bundeskanzler?") route sharply to the model with the freshest topic knowledge.

The default is a transparent lexical logistic score over bilingual (EN/DE) cues: temporal words, the years
2024-2027, office/role nouns, present-state questions, capitalised-entity density, acronyms and news-topic keyword
mass (from :class:`~cutoffmix.router.topics.KeywordTopicClassifier`), minus math/logic cues. Optionally a logistic
regression on hashed features is trained (:meth:`TimeSensitivityGate.train`) and blended with the lexical score.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from cutoffmix.router.features import DEFAULT_DIM, WORD_RE, SparseSoftmaxRegression, hashed_features
from cutoffmix.router.topics import KeywordTopicClassifier, detect_language, stem

__all__ = ["TimeSensitivityGate"]


def _stem_set(words: str) -> frozenset[str]:
    return frozenset(stem(w) for w in words.split())


_STRONG_TEMPORAL = _stem_set(
    "current currently latest nowadays today presently newest incumbent ongoing reigning "
    "aktuell aktuelle aktuellen aktueller aktuelles derzeit derzeitig derzeitige derzeitigen derzeitiger zurzeit "
    "momentan gegenwärtig gegenwärtige gegenwärtigen heute heutzutage neueste neuesten neuester neuestes jüngste "
    "jüngsten amtierend amtierende amtierenden amtierender mittlerweile inzwischen bislang"
)
_STRONG_TEMPORAL_PHRASES = (
    "as of", "right now", "at the moment", "at present", "these days", "this year", "this month", "this week",
    "up to date", "most recent", "still alive", "still in office", "zur zeit", "im moment", "dieses jahr",
    "in diesem jahr", "noch am leben", "lebt noch", "noch im amt",
)
_WEAK_TEMPORAL = _stem_set(
    "now recent recently new still yet upcoming soon last kürzlich jetzt neu neue neuen neuer noch bisher zuletzt "
    "letzte letzten"
)
_ROLES = _stem_set(
    "ceo president chancellor minister secretary governor mayor pope king queen monarch leader chairman chairwoman "
    "chair coach manager champion titleholder director ambassador speaker commissioner owner holder premier "
    "bundeskanzler kanzler kanzlerin präsident präsidentin ministerpräsident ministerin bürgermeister vorstandschef "
    "vorsitzende vorsitzender chef chefin trainer trainerin papst könig königin meister weltmeister botschafter "
    "regierungschef staatschef amtsinhaber"
)
_ROLE_PHRASES = ("prime minister", "head coach", "head of state", "head of government", "chief executive")
# distinctly mathematical / puzzle vocabulary (generic reasoning words such as "let", "therefore" or "sum" also
# occur in time-sensitive notes and are deliberately absent)
_MATH = _stem_set(
    "prove proof lemma theorem integer integers equation equations divisible modulo derivative integral polynomial "
    "probability matrix vector triangle digit digits remainder multiply divide fraction puzzle riddle contradiction "
    "knight knave liar beweis gleichung primzahl teilbar wahrscheinlichkeit rätsel dreieck polynom ableitung "
    "widerspruch ungerade"
)
_PRESENT_QUESTION = re.compile(
    r"\b(?:who|wer)\s+(?:is|are|was|'s|ist|sind|leads|heads|runs|holds|owns|coaches|führt|leitet|regiert|trainiert)\b"
    r"|\b(?:what|which|was|welche[srnm]?)\s+(?:is|are|ist|sind)\s+(?:the\s+|der\s+|die\s+|das\s+)?"
    r"(?:latest|newest|current|most\s+recent|neueste\w*|aktuelle\w*)\b"
    r"|\b(?:is|are|ist|sind|lebt)\s+\w+(?:\s+\w+)?\s+(?:still|noch)\b"
)
_YEAR = re.compile(r"\b202[4-7]\b")
_OPERATOR = re.compile(r"[=+*/^<>≤≥≠∑∫√±×÷]")
_SENTENCE_BREAK = ".!?:;\n\"'(["

# logistic weights of the lexical score; each count enters through _sat (1 cue -> 0.75, 2 -> 0.94)
_BIAS = -2.2
_W_STRONG, _W_WEAK, _W_YEAR, _W_ROLE, _W_QUESTION = 2.0, 0.6, 1.2, 1.0, 1.0
_W_ENTITY, _W_ACRONYM, _W_NEWS, _W_MATH = 0.8, 0.4, 1.0, -2.5
_W_QUESTION_ROLE = 0.8  # "who is the <role> ..." asks for a present office holder even without a temporal word
_ENTITY_FULL_DENSITY = 0.15  # share of capitalised mid-sentence tokens that counts as entity-dense


def _sat(x: float) -> float:
    return 1.0 - math.exp(-1.4 * x)


def _padded_count(padded: str, phrases: Sequence[str]) -> int:
    return sum(padded.count(f" {p} ") for p in phrases)


class TimeSensitivityGate:
    """Score r(W) in [0, 1]: lexical baseline, optionally blended with a trained logistic regression.

    ``score = learned_weight * p_learned + (1 - learned_weight) * p_lexical`` when a model is present, else
    ``p_lexical``.
    """

    _KIND = "gate_linear"

    def __init__(
        self,
        model: SparseSoftmaxRegression | None = None,
        *,
        learned_weight: float = 0.5,
        keywords: KeywordTopicClassifier | None = None,
    ):
        if not 0.0 <= learned_weight <= 1.0:
            raise ValueError(f"learned_weight must lie in [0, 1], got {learned_weight}")
        if model is not None and model.n_classes != 2:
            raise ValueError(f"the gate model must be binary, got {model.n_classes} classes")
        self.model = model
        self.learned_weight = learned_weight
        self._keywords = keywords or KeywordTopicClassifier()

    def cues(self, text: str) -> dict[str, float]:
        """Raw lexical cue counts/densities of ``text`` (inputs of :meth:`lexical_score`)."""
        matches = list(WORD_RE.finditer(text))
        lower = [m.group().lower() for m in matches]
        stems = [stem(t) for t in lower]
        padded = " " + " ".join(lower) + " "
        padded_stems = " " + " ".join(stems) + " "
        lang = detect_language(lower)
        entity = 0.0
        if lang != "de" and matches:  # German capitalises every noun: entity density is uninformative there
            capitalised = 0
            for prev, m in zip(matches, matches[1:]):  # the first token always starts a sentence
                gap = text[prev.end() : m.start()]
                if m.group()[0].isupper() and not any(ch in _SENTENCE_BREAK for ch in gap):
                    capitalised += 1
            entity = capitalised / len(matches)
        return {
            "strong_temporal": sum(1 for s in stems if s in _STRONG_TEMPORAL)
            + _padded_count(padded, _STRONG_TEMPORAL_PHRASES),
            "weak_temporal": sum(1 for s in stems if s in _WEAK_TEMPORAL),
            "years": len(_YEAR.findall(text)),
            "roles": sum(1 for s in stems if s in _ROLES)
            + _padded_count(padded_stems, [" ".join(stem(w) for w in p.split()) for p in _ROLE_PHRASES]),
            "present_question": len(_PRESENT_QUESTION.findall(text.lower())),
            "entity_density": entity,
            "acronyms": sum(1 for m in matches if len(m.group()) >= 2 and m.group().isupper()),
            "news": float(sum(self._keywords.evidence(text))),
            "math": sum(1 for s in stems if s in _MATH) + len(_OPERATOR.findall(text)) / 2,
        }

    def lexical_score(self, text: str) -> float:
        """Lexical baseline probability that ``text`` needs recent world knowledge."""
        c = self.cues(text)
        logit = (
            _BIAS
            + _W_STRONG * _sat(c["strong_temporal"])
            + _W_WEAK * _sat(c["weak_temporal"])
            + _W_YEAR * _sat(c["years"])
            + _W_ROLE * _sat(c["roles"])
            + _W_QUESTION * _sat(c["present_question"])
            + _W_QUESTION_ROLE * min(c["present_question"], 1.0) * min(c["roles"], 1.0)
            + _W_ENTITY * min(c["entity_density"] / _ENTITY_FULL_DENSITY, 1.0)
            + _W_ACRONYM * _sat(c["acronyms"])
            + _W_NEWS * _sat(c["news"] / 2)
            + _W_MATH * _sat(c["math"] / 2)
        )
        return 1.0 / (1.0 + math.exp(-logit))

    def score(self, text: str) -> float:
        """r(W) in [0, 1]."""
        lexical = self.lexical_score(text)
        if self.model is None:
            return lexical
        learned = float(self.model.predict_proba(hashed_features(text, self.model.dim))[1])
        return self.learned_weight * learned + (1.0 - self.learned_weight) * lexical

    @classmethod
    def train(
        cls,
        texts: Sequence[str],
        labels: Sequence[float | int | bool],
        *,
        dim: int = DEFAULT_DIM,
        epochs: int = 8,
        lr: float = 0.5,
        l2: float = 1e-6,
        seed: int = 0,
        batch_size: int = 32,
        learned_weight: float = 0.5,
    ) -> "TimeSensitivityGate":
        """Train the learned part on ``texts`` with binary ``labels`` (time-sensitive = truthy / >= 0.5)."""
        if len(texts) != len(labels):
            raise ValueError(f"texts and labels differ in length: {len(texts)} != {len(labels)}")
        y = [1 if float(v) >= 0.5 else 0 for v in labels]
        if len(set(y)) < 2:
            raise ValueError("gate training needs both time-sensitive and time-insensitive examples")
        model = SparseSoftmaxRegression(dim, 2).fit(
            [hashed_features(t, dim) for t in texts], y, epochs=epochs, lr=lr, l2=l2, batch_size=batch_size, seed=seed
        )
        return cls(model, learned_weight=learned_weight)

    def save(self, path: str | Path) -> None:
        """Write the learned part (``.npz``, no pickles) to exactly ``path``; a lexical-only gate cannot be saved."""
        if self.model is None:
            raise ValueError("this gate has no learned model to save (the lexical baseline needs no file)")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            np.savez_compressed(
                f, kind=np.asarray(self._KIND), learned_weight=np.asarray(self.learned_weight), **self.model.to_arrays()
            )

    @classmethod
    def load(cls, path: str | Path) -> "TimeSensitivityGate":
        """Load a gate written by :meth:`save`."""
        with np.load(Path(path), allow_pickle=False) as z:
            kind = str(z["kind"]) if "kind" in z.files else ""
            if kind != cls._KIND:
                raise ValueError(f"{path} is not a TimeSensitivityGate file (kind={kind!r})")
            return cls(SparseSoftmaxRegression.from_arrays(z), learned_weight=float(z["learned_weight"]))
