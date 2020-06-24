"""Maximal matching class."""
import networkx as nx

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "MaximalMatching"


class MaximalMatching(FeatureClass):
    """Maximal matching class.

    Maximal matching calculations using networkx:
        `Networkx_maximal_matching <https://networkx.github.io/documentation/stable/\
        reference/algorithms/matching.html>`_
    """

    modes = ["fast", "medium", "slow"]
    shortname = "MM"
    name = "MaximalMatching"
    encoding = "networkx"

    def compute_features(self):

        self.add_feature(
            "maximal_matching",
            lambda graph: len(nx.maximal_matching(graph)),
            "Maximal matching",
            InterpretabilityScore(4),
        )
