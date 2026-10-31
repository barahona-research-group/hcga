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

# Classes whose features are only defined for directed graphs (they return nothing on an
# undirected graph by design), mapped to a fixture on which they are defined. Trophic
# levels (Looplessness) need basal nodes from which every node is reachable, i.e. a DAG-like
# graph rather than a random digraph.
DIRECTED_ONLY = {
    "InOutDegrees": "dir_er30",
    "Reciprocity": "dir_er30",
    "FlowHierarchy": "dir_er30",
    "BasalNodes": "dir_er30",
    "Looplessness": "dir_gn20",
}

# Classes whose features need node features.
NEEDS_NODE_FEATURES = {"NodeFeatures"}


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


@pytest.mark.parametrize("feature_class", test_feature_classes, ids=lambda cls: cls.__name__)
def test_class_computes_features(feature_class):
    """Each class returns at least one non-None feature on a suitable fixture graph."""
    name = feature_class.__name__
    if name in NEEDS_NODE_FEATURES:
        pytest.skip("fixture graphs have no node features")
    graph = fixture_graph(DIRECTED_ONLY.get(name, "karate"))
    features = feature_class(graph).get_features()
    computed = {k: v for k, v in features.items() if v is not None}
    assert computed, f"{name} computed no features"
