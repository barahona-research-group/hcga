"""Vitality class."""
import networkx as nx

from hcga.feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "Vitality"


def vitality(graph):
    return list(nx.closeness_vitality(graph).values())


class Vitality(FeatureClass):
    """Vitality class."""

    modes = ["slow"]
    shortname = "V"
    name = "vitality"
    encoding = "networkx"

    def compute_features(self):

        # distribution of vitality
        self.add_feature(
            "vitality",
            vitality,
            "The closeness vitality of a node is the change in the sum of distances between \
            all node pairs when excluding that node",
            InterpretabilityScore(3),
            statistics="centrality",
        )
