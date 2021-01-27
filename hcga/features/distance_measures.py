"""Distance Measures class."""
import networkx as nx

from hcga.feature_class import FeatureClass, InterpretabilityScore
from hcga.features.utils import ensure_connected

featureclass_name = "DistanceMeasures"

@ensure_connected
def barycenter_size(graph):
    """barycenter_size"""
    return len(nx.barycenter(graph))

@ensure_connected
def barycenter_size_weighted(graph):
    """barycenter_size_weighted"""
    return len(nx.barycenter(graph, weight="weight"))

@ensure_connected
def center_size(graph):
    """center_size"""
    return len(nx.center(graph))

@ensure_connected
def periphery(graph):
    """periphery"""
    return len(nx.periphery(graph))

@ensure_connected
def eccentricity(graph):
    """eccentricity"""
    return list(nx.eccentricity(graph).values())


class DistanceMeasures(FeatureClass):
    """Distance Measures class."""

    modes = ["fast", "medium", "slow"]
    shortname = "DM"
    name = "distance_measures"
    encoding = "networkx"

    def compute_features(self):

        # barycenter
        self.add_feature(
            "barycenter_size",
            barycenter_size,
            "The barycenter is the subgraph which minimises a distance function",
            InterpretabilityScore(4),
        )
        self.add_feature(
            "barycenter_size_weighted",
            barycenter_size_weighted,
            "The barycenter is the subgraph which minimises a distance function",
            InterpretabilityScore(4),
        )

        # center
        self.add_feature(
            "center_size",
            center_size,
            "The center is the subgraph of nodes with eccentricity equal to radius",
            InterpretabilityScore(3),
        )

        # extrema bounding
        self.add_feature(
            "center_size",
            ensure_connected(nx.extrema_bounding),
            "The largest distance in the graph",
            InterpretabilityScore(4),
        )

        # periphery
        self.add_feature(
            "periphery",
            periphery,
            "The number of peripheral nodes in the graph",
            InterpretabilityScore(4),
        )

        # eccentricity
        self.add_feature(
            "eccentricity",
            eccentricity,
            "The distribution of node eccentricity across the network",
            InterpretabilityScore(3),
            statistics="centrality",
        )
