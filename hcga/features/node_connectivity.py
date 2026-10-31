"""Node connectivity class."""

import itertools
from functools import lru_cache

import networkx as nx
import numpy as np

from hcga.feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "NodeConnectivity"


@lru_cache(maxsize=1)
def eval_node_conn(graph):
    """Local node connectivity of every pair of distinct nodes (ordered pairs for digraphs)."""
    node_connectivity = nx.all_pairs_node_connectivity(graph)
    if graph.is_directed():
        pairs = itertools.permutations(graph, 2)
    else:
        pairs = itertools.combinations(graph, 2)
    return np.array([node_connectivity[u][v] for u, v in pairs], dtype=float)


def node_conn(graph):
    """calculating node connectivity"""
    return list(eval_node_conn(graph))


def node_connectivity(graph):
    """Node connectivity: the minimum local node connectivity over all pairs (Menger)."""
    return int(np.min(eval_node_conn(graph)))


def avg_node_connectivity(graph):
    """Average local node connectivity over all pairs."""
    return float(np.mean(eval_node_conn(graph)))


class NodeConnectivity(FeatureClass):
    """Node connectivity class.

    Implementation of networkx code:
            `Networkx_node_connectivity <https://networkx.github.io/documentation/stable/_modules/\
            networkx/algorithms/connectivity/connectivity.html#node_connectivity>`_
    """

    modes = ["slow"]
    shortname = "NC"
    name = "node_connectivity"
    encoding = "networkx"

    def compute_features(self):
        self.add_feature(
            "node_conn",
            node_conn,
            "Node connectivity (statistics)",
            InterpretabilityScore("max") - 1,
            statistics="centrality",
        )

        # Calculate connectivity
        self.add_feature(
            "node_connectivity",
            node_connectivity,
            "Node connectivity",
            InterpretabilityScore("max") - 1,
        )
        self.add_feature(
            "avg_node_connectivity",
            avg_node_connectivity,
            "Average node connectivity",
            InterpretabilityScore("max") - 1,
        )
        self.add_feature(
            "edge_connectivity",
            nx.edge_connectivity,
            "Edge connectivity",
            InterpretabilityScore("max") - 1,
        )
