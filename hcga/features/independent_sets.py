"""Independent sets class."""
import networkx as nx

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "IndependentSets"


def size_max_indep_set(graph):
    """size_max_indep_set"""
    return len(nx.maximal_independent_set(graph))


class IndependentSets(FeatureClass):
    """Independent sets class."""

    modes = ["fast", "medium", "slow"]
    shortname = "IS"
    name = "independent_sets"
    encoding = "networkx"

    def compute_features(self):

        self.add_feature(
            "size_max_indep_set",
            size_max_indep_set,
            "The number of nodes in the maximal independent set",
            InterpretabilityScore(3),
        )
