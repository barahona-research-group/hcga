"""Vitality class."""
import networkx as nx

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "Vitality"


class Vitality(FeatureClass):
    """Vitality class."""

    modes = ["slow"]
    shortname = "V"
    name = "vitality"
    encoding = "networkx"

    def compute_features(self):
        """Compute vitality measures.

        Notes
        -----


        References
        ----------
        """

        # distribution of vitality
        self.add_feature(
            "vitality",
            lambda graph: list(nx.closeness_vitality(graph).values()),
            "The closeness vitality of a node is the change in the sum of distances between \
            all node pairs when excluding that node",
            InterpretabilityScore(3),
            statistics="centrality",
        )
