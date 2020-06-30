"""Connectance class."""
import networkx as nx

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "connectance"

"Directed networks only"

class Connectance(FeatureClass):
    """Connectance class.
    """

    modes = ["fast", "medium", "slow"]
    shortname = "Cns"
    name = "connectance"
    encoding = "networkx"

    def compute_features(self):

        self.add_feature(
            "connectance",
            lambda graph: nx.number_of_edges(graph)/(nx.number_of_nodes(graph)**2),
            "ratio of number of edges to maximum possible number of edges",
            InterpretabilityScore(3)
        )