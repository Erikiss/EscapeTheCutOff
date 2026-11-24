"""Routing: which model writes the next segment of the shared notes.

``Router`` combines a topic classifier (``topics``), a time-sensitivity gate (``gate``), the external cutoff index
(``index``) and a routing policy (``policy``). Training helpers live in ``train`` (CPU) and ``train_gpu`` (optional,
needs torch/transformers; not imported here).
"""

from cutoffmix.router.features import hashed_features, tokenize
from cutoffmix.router.gate import TimeSensitivityGate
from cutoffmix.router.index import CutoffIndex, ProbeIndex, load_index
from cutoffmix.router.policy import (
    ArgmaxHysteresisPolicy,
    FixedPolicy,
    PolicyDecision,
    RandomPolicy,
    RoundRobinPolicy,
    RoutingPolicy,
    SoftmaxPolicy,
    ThompsonPolicy,
    make_policy,
)
from cutoffmix.router.router import RouteDecision, Router
from cutoffmix.router.topics import (
    EmbeddingTopicClassifier,
    KeywordTopicClassifier,
    LinearTopicClassifier,
    load_classifier,
)

__all__ = [
    "ArgmaxHysteresisPolicy",
    "CutoffIndex",
    "EmbeddingTopicClassifier",
    "FixedPolicy",
    "KeywordTopicClassifier",
    "LinearTopicClassifier",
    "PolicyDecision",
    "ProbeIndex",
    "RandomPolicy",
    "RoundRobinPolicy",
    "RouteDecision",
    "Router",
    "RoutingPolicy",
    "SoftmaxPolicy",
    "ThompsonPolicy",
    "TimeSensitivityGate",
    "hashed_features",
    "load_classifier",
    "load_index",
    "make_policy",
    "tokenize",
]
