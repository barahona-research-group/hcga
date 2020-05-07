"""Maximal matching class."""
import networkx as nx

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "MaximalMatching"


class MaximalMatching(FeatureClass):
    """Maximal matching class."""

    modes = ["fast", "medium", "slow"]
    shortname = "MM"
    name = "MaximalMatching"
    encoding = "networkx"

    def compute_features(self):
        """Compute the maximal matching of the network.

        A matching is a subset of edges such that no edges are connected to
        the same node. It is maximal if the addition of another edge to this
        subset no longer makes it a matching.


        Notes
        -----
        Maximal matching calculations using networkx:
            `Networkx_maximal_matching <https://networkx.github.io/documentation/stable/\
            reference/algorithms/matching.html>`_
        """

        self.add_feature(
            "maximal_matching",
            lambda graph: len(nx.maximal_matching(graph)),
            "Maximal matching",
            InterpretabilityScore(4),
        )
