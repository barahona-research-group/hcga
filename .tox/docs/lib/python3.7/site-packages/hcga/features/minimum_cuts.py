"""Minimum cuts class."""
import networkx as nx

from ..feature_class import FeatureClass, InterpretabilityScore
from .utils import ensure_connected

featureclass_name = "MinimumCuts"


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
            "min_node_cut_size'",
            lambda graph: len(nx.minimum_node_cut(ensure_connected(graph))),
            "Minimum node cut size",
            InterpretabilityScore("max"),
        )

        self.add_feature(
            "min_edge_cut_size'",
            lambda graph: len(nx.minimum_edge_cut(ensure_connected(graph))),
            "Minimum edge cut size",
            InterpretabilityScore("max"),
        )
