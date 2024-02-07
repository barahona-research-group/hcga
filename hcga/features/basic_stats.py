"""Basic stats class."""

import networkx as nx
import numpy as np

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
    return np.float64(2 * n_edges(graph)) / np.float64(n_nodes(graph) * (n_edges(graph) - 1))


def edge_weights(graph):
    """edge_weights"""
    return list(nx.get_edge_attributes(graph, "weight").values())


def diameter(graph):
    """diameter"""
    return nx.diameter(ensure_connected(graph))


def radius(graph):
    """radius"""
    return nx.radius(ensure_connected(graph))


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
