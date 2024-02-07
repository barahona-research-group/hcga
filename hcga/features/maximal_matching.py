"""Maximal matching class."""

import networkx as nx

from hcga.feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "MaximalMatching"


def maximal_matching(graph):
    """maximal_matching"""
    return len(nx.maximal_matching(graph))


class MaximalMatching(FeatureClass):
    """Maximal matching class.

    A matching is a subset of edges in which no node occurs more than once.
    A maximal matching cannot add more edges and still be a matching.

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
            maximal_matching,
            "Maximal matching",
            InterpretabilityScore(4),
        )
