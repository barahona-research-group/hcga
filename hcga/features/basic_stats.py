"""Basic stats class."""
import networkx as nx
import numpy as np

from ..feature_class import FeatureClass, InterpretabilityScore
from .utils import ensure_connected

featureclass_name = "BasicStats"


class BasicStats(FeatureClass):
    """Basic stats class."""

    modes = ["fast", "medium", "slow"]
    shortname = "BS"
    name = "basic_stats"
    encoding = "networkx"

    def compute_features(self):

        n_nodes = lambda graph: len(graph)
        n_edges = lambda graph: len(graph.edges)

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
            lambda graph: nx.diameter(ensure_connected(graph)),
            "Diameter of the graph",
            InterpretabilityScore("max"),
        )
        self.add_feature(
            "radius",
            lambda graph: nx.radius(ensure_connected(graph)),
            "Radius of the graph",
            InterpretabilityScore("max"),
        )

        # Degree stats
        density = lambda graph: np.float64(2 * n_edges(graph)) / np.float64(
            n_nodes(graph) * (n_edges(graph) - 1)
        )
        self.add_feature(
            "density", density, "Density of the graph", InterpretabilityScore("max")
        )

        self.add_feature(
            "edge_weights",
            lambda graph: list(nx.get_edge_attributes(graph, "weight").values()),
            "Weights of the edges",
            InterpretabilityScore("max"),
            statistics="centrality",
        )
