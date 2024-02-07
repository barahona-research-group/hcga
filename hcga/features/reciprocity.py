"""Reciprocity class."""

import networkx as nx

from hcga.feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "Reciprocity"


class Reciprocity(FeatureClass):
    """Reciprocity class.

    Features based on the reciprocity in a directed graph.

    The reciprocity of a directed graph is defined as the ratio
    of the number of edges pointing in both directions to the total
    number of edges in the graph.
    Formally, $r = |{(u,v) \in G|(v,u) \in G}| / |{(u,v) \in G}|$.

    The reciprocity of a single node u is defined similarly,
    it is the ratio of the number of edges in both directions to
    the total number of edges attached to node u.

    Reciprocity calculations using networkx:
            `Reciprocity <https://networkx.org/documentation/stable/reference/algorithms/\
                reciprocity.html>`_

    """

    modes = ["fast", "medium", "slow"]
    shortname = "Rec"
    name = "reciprocity"
    encoding = "networkx"

    def compute_features(self):
        # graph clique number
        self.add_feature(
            "reciprocity",
            nx.overall_reciprocity,
            "fraction of edges pointing in both directions to total number of edges",
            InterpretabilityScore(3),
        )
