"""Simple cycles class."""
from functools import lru_cache

import networkx as nx

from hcga.feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "SimpleCycles"


@lru_cache(maxsize=None)
def simple_cycles_func(graph):
    return list(nx.simple_cycles(graph))


def number_simple_cycles(graph):
    return len(simple_cycles_func(graph))


def simple_cycles_sizes(graph):
    return [len(i) for i in simple_cycles_func(graph)]


class SimpleCycles(FeatureClass):
    """Simple cycles class."""

    modes = ["fast", "medium", "slow"]
    shortname = "SC"
    name = "simple_cycles"
    encoding = "networkx"

    def compute_features(self):

        self.add_feature(
            "number_simple_cycles",
            number_simple_cycles,
            "A simple closed path with no repeated nodes (except the first)",
            InterpretabilityScore(3),
        )

        self.add_feature(
            "simple_cycles_sizes",
            simple_cycles_sizes,
            "the distribution of simple cycle lengths",
            InterpretabilityScore(3),
            statistics="centrality",
        )
