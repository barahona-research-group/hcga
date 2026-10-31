"""Covering class."""

import networkx as nx

from hcga.feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "Covering"


def min_edge_cover(graph):
    """Size of a minimum edge cover (directed graphs treated as undirected).

    Returns None if the graph has an isolated node, since no edge cover exists then.
    """
    if graph.is_directed():
        graph = graph.to_undirected(as_view=True)
    if nx.number_of_isolates(graph) > 0:
        return None
    return len(nx.min_edge_cover(graph))


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
