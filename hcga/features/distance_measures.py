"""Distance Measures class."""
import networkx as nx
import numpy as np

from ..feature_class import FeatureClass, InterpretabilityScore
from .utils import ensure_connected

featureclass_name = "DistanceMeasures"


class DistanceMeasures(FeatureClass):
    """Distance Measures class."""

    modes = ["fast", "medium", "slow"]
    shortname = "DM"
    name = "distance_measures"
    encoding = "networkx"

    def compute_features(self):
        """Compute the distance measures of the network.

        Computed statistics
        -----
        Put here the list of things that are computed, with corresponding names
        """

        # barycenter
        self.add_feature(
            "barycenter_size",
            lambda graph: len(nx.barycenter(ensure_connected(graph))),
            "The barycenter is the subgraph which minimises a distance function",
            InterpretabilityScore(4),
        )

        # center
        self.add_feature(
            "center_size",
            lambda graph: len(nx.center(ensure_connected(graph))),
            "The center is the subgraph of nodes with eccentricity equal to radius",
            InterpretabilityScore(3),
        )

        # extrema bounding
        self.add_feature(
            "center_size",
            lambda graph: nx.extrema_bounding(ensure_connected(graph)),
            "The largest distance in the graph",
            InterpretabilityScore(4),
        )

        # periphery
        self.add_feature(
            "periphery",
            lambda graph: len(nx.periphery(ensure_connected(graph))),
            "The number of peripheral nodes in the graph",
            InterpretabilityScore(4),
        )

        # eccentricity
        self.add_feature(
            "eccentricity",
            lambda graph: list(nx.eccentricity(ensure_connected(graph)).values()),
            "The distribution of node eccentricity across the network",
            InterpretabilityScore(3),
            statistics="centrality",
        )
