"""Flow hierarchy class."""
import networkx as nx

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "FlowHierarchy"


class FlowHierarchy(FeatureClass):
    """Flow hierarchy class."""

    modes = ["fast", "medium", "slow"]
    shortname = "FH"
    name = "flow_hierarchy"
    encoding = "networkx"

    def compute_features(self):

        # graph clique number
        self.add_feature(
            "flow_hierarchy",
            lambda graph: nx.flow_hierarchy(graph),
            "fraction of edges not participating in cycles",
            InterpretabilityScore(3),
        )
