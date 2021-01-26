"""Reciprocity class."""
import networkx as nx

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "Reciprocity"


class Reciprocity(FeatureClass):
    """Reciprocity class."""

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
