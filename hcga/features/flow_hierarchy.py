"""Flow hierarchy class."""
from functools import partial

import networkx as nx

from hcga.feature_class import FeatureClass, InterpretabilityScore

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
            nx.flow_hierarchy,
            "fraction of edges not participating in cycles",
            InterpretabilityScore(3),
        )

        self.add_feature(
            "flow_hierarchy_weighted",
            partial(nx.flow_hierarchy, weight="weight"),
            "fraction of edges not participating in cycles",
            InterpretabilityScore(3),
        )
