"""Covering class."""
import networkx as nx

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "Covering"


class Covering(FeatureClass):
    """Covering class."""

    modes = ["fast", "medium", "slow"]
    shortname = "CV"
    name = "covering"
    encoding = "networkx"

    def compute_features(self):

        self.add_feature(
            "min_edge_cover",
            lambda graph: len(list(nx.min_edge_cover(graph))),
            "The number of edges which consistutes the minimum edge cover of the graph",
            InterpretabilityScore(3),
        )
