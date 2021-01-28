"Connectance class"
import networkx as nx

from hcga.feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "Connectance"


class Connectance(FeatureClass):
    """Connectance class."""

    modes = ["fast", "medium", "slow"]
    shortname = "Cns"
    name = "connectance"
    encoding = "networkx"

    def compute_features(self):

        self.add_feature(
            "connectance",
            nx.density,
            "ratio of number of edges to maximum possible number of edges",
            InterpretabilityScore(3),
        )
