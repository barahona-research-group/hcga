"""Eulerian Measures class."""
import networkx as nx

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "Eulerian"


class Eulerian(FeatureClass):
    """Eulerian Measures class."""

    modes = ["fast", "medium", "slow"]
    shortname = "EU"
    name = "eulerian"
    encoding = "networkx"

    def compute_features(self):
        """Compute the Eulerian measures of the network.

        Computed statistics
        -----
        Put here the list of things that are computed, with corresponding names
        """

        # checking if eulerian
        self.add_feature(
            "eulerian",
            lambda graph: nx.is_eulerian(graph) * 1,
            "A graph is eulerian if it has a eulerian circuit: a closed walk that includes \
            each edges of the graph exactly once",
            InterpretabilityScore(3),
        )

        # checking if semi eulerian
        self.add_feature(
            "semi_eulerian",
            lambda graph: nx.is_semieulerian(graph) * 1,
            "A graph is semi eulerian if it has a eulerian path but no eulerian circuit",
            InterpretabilityScore(3),
        )

        # checking if eulerian path exists
        self.add_feature(
            "semi_eulerian",
            lambda graph: nx.has_eulerian_path(graph) * 1,
            "Whether a eulerian path exists in the network",
            InterpretabilityScore(3),
        )
