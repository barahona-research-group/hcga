"""Structural tests for hcga.features: every class is set up correctly and computes something."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures import fixture_graph  # noqa: E402

from hcga.extraction import get_list_feature_classes  # noqa: E402
from hcga.utils import get_trivial_graph  # noqa: E402

# timeout=0 evaluates in-process; the subprocess timeout yields nothing under the spawn
# start method (macOS, Windows), which is what the old tests silently passed with.
test_feature_classes, _ = get_list_feature_classes(
    mode="all", statistics_level="advanced", timeout=0
)

# Classes that currently return no features at all under networkx >= 3.0.
KNOWN_DEAD = {"Hits", "ScaleFree"}

# Classes that do not terminate in-process on a 34-node graph.
UNBOUNDED = {"SimpleCycles"}


def test_shortname_definition():
    for feature_class in test_feature_classes:
        assert feature_class.shortname != "TP"


def test_name_definition():
    for feature_class in test_feature_classes:
        assert feature_class.name != "template"


def test_shortnames_unique():
    shortnames = [feature_class.shortname for feature_class in test_feature_classes]
    assert len(set(shortnames)) == len(shortnames)


def test_trivial_graph():
    graph = get_trivial_graph()
    for feature_class in test_feature_classes:
        feature_class(graph).get_features()


@pytest.mark.parametrize(
    "feature_class",
    [cls for cls in test_feature_classes if cls.__name__ not in UNBOUNDED],
    ids=lambda cls: cls.__name__,
)
def test_class_computes_features(feature_class):
    """Each class returns at least one non-None feature on the karate club graph."""
    if feature_class.__name__ in KNOWN_DEAD:
        pytest.xfail(f"{feature_class.__name__} is broken under current networkx (see audit)")
    features = feature_class(fixture_graph("karate")).get_features()
    computed = {k: v for k, v in features.items() if v is not None}
    assert computed, f"{feature_class.__name__} computed no features"
