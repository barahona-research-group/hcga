"""Minimum cuts class."""

import networkx as nx

from hcga.feature_class import FeatureClass, InterpretabilityScore
from hcga.features.utils import ensure_connected

featureclass_name = "MinimumCuts"


def min_node_cut_size(graph):
    """min_node_cut_size"""
    return len(nx.minimum_node_cut(ensure_connected(graph)))


def min_edge_cut_size(graph):
    """min_node_cut_size"""
    return len(nx.minimum_edge_cut(ensure_connected(graph)))


class MinimumCuts(FeatureClass):
    """Minimum cuts class.

    Features derived from the minimum cuts of a graph.

    The minimum cuts explores the set of nodes or edges of minimum cardinality that,
    if removed, would destroy all paths among source and target in G.

    Calculations using networkx:
        `Networkx_minimum_cuts <https://networkx.github.io/documentation/stable/\
        reference/algorithms/connectivity.html>`_

    References
    ----------
    .. [1] Abdol-Hossein Esfahanian. Connectivity Algorithms.
        http://www.cse.msu.edu/~cse835/Papers/Graph_connectivity_revised.pdf

    """

    modes = ["medium", "slow"]
    shortname = "MiC"
    name = "minimum_cuts"
    encoding = "networkx"

    def compute_features(self):
        self.add_feature(
            "min_node_cut_size",
            min_node_cut_size,
            "Minimum node cut size",
            InterpretabilityScore("max"),
        )

        self.add_feature(
            "min_edge_cut_size",
            min_edge_cut_size,
            "Minimum edge cut size",
            InterpretabilityScore("max"),
        )
