"""Distance Measures class."""
import networkx as nx

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "Efficiency"


class Efficiency(FeatureClass):
    """Distance Measures class."""

    modes = ["fast", "medium", "slow"]
    shortname = "EF"
    name = "efficiency"
    encoding = "networkx"

    def compute_features(self):

        # local effiency
        self.add_feature(
            "local_efficiency",
            nx.local_efficiency,
            "The local efficiency",
            InterpretabilityScore(4),
        )

        # global effiency
        self.add_feature(
            "global_efficiency",
            nx.global_efficiency,
            "The global efficiency",
            InterpretabilityScore(4),
        )
