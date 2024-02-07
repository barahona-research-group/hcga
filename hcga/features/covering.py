"""Covering class."""

import networkx as nx

from hcga.feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "Covering"


def min_edge_cover(graph):
    """min_edge_cover"""
    return len(list(nx.min_edge_cover(graph)))


class Covering(FeatureClass):
    """Covering class.

    Features based on the minimum edge cover of a graph.

    Uses networkx, 'https://networkx.org/documentation/stable//reference/algorithms/covering.html

    """

    modes = ["fast", "medium", "slow"]
    shortname = "CV"
    name = "covering"
    encoding = "networkx"

    def compute_features(self):
        self.add_feature(
            "min_edge_cover",
            min_edge_cover,
            "The number of edges which consistutes the minimum edge cover of the graph",
            InterpretabilityScore(3),
        )
