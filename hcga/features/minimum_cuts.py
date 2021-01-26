"""Minimum cuts class."""
import networkx as nx

from ..feature_class import FeatureClass, InterpretabilityScore
from .utils import ensure_connected

featureclass_name = "MinimumCuts"


@ensure_connected
def min_node_cut_size(graph):
    """min_node_cut_size"""
    return len(nx.minimum_node_cut(graph))


@ensure_connected
def min_edge_cut_size(graph):
    """min_node_cut_size"""
    return len(nx.minimum_edge_cut(graph))


class MinimumCuts(FeatureClass):
    """Minimum cuts class.

    Calculations using networkx:
        `Networkx_minimum_cuts <https://networkx.github.io/documentation/stable/\
        reference/algorithms/connectivity.html>`_
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
