"""Basic stats class."""

from functools import lru_cache

import networkx as nx

from hcga.feature_class import FeatureClass, InterpretabilityScore
from hcga.features.utils import ensure_connected

featureclass_name = "BasicStats"


def n_nodes(graph):
    """n_nodes"""
    return len(graph.nodes)


def n_edges(graph):
    """n_edges"""
    return len(graph.edges)


def density(graph):
    """density"""
    return nx.density(graph)


def edge_weights(graph):
    """edge_weights"""
    return list(nx.get_edge_attributes(graph, "weight").values())


@lru_cache(maxsize=1)
def _eccentricity(graph):
    """Eccentricities in the largest connected component (of the undirected graph if directed)."""
    if graph.is_directed():
        graph = graph.to_undirected(as_view=True)
    return nx.eccentricity(ensure_connected(graph))


def diameter(graph):
    """Diameter of the largest connected component (undirected view of a directed graph)."""
    return max(_eccentricity(graph).values())


def radius(graph):
    """Radius of the largest connected component (undirected view of a directed graph)."""
    return min(_eccentricity(graph).values())


class BasicStats(FeatureClass):
    """Basic stats class.

    Here we compute basic measures statistics of the graphs, e.g. number of nodes.

    References
    ----------
    .. [1] Mark E. J. Newman.
       *Networks: An Introduction.*
       Oxford University Press, USA, 2010, pp. 169.

    """

    modes = ["fast", "medium", "slow"]
    shortname = "BS"
    name = "basic_stats"
    encoding = "networkx"

    def compute_features(self):
        # Adding basic node and edge numbers
        self.add_feature(
            "num_nodes",
            n_nodes,
            "Number of nodes in the graph",
            InterpretabilityScore("max"),
        )
        self.add_feature(
            "num_edges",
            n_edges,
            "Number of edges in the graph",
            InterpretabilityScore("max"),
        )

        # Adding diameter stats
        self.add_feature(
            "diameter",
            diameter,
            "Diameter of the graph",
            InterpretabilityScore("max"),
        )
        self.add_feature(
            "radius",
            radius,
            "Radius of the graph",
            InterpretabilityScore("max"),
        )

        # Degree stats
        self.add_feature("density", density, "Density of the graph", InterpretabilityScore("max"))
        self.add_feature(
            "edge_weights",
            edge_weights,
            "Weights of the edges",
            InterpretabilityScore("max"),
            statistics="centrality",
        )
