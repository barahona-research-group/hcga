"""Distance Measures class."""

import networkx as nx

from hcga.feature_class import FeatureClass, InterpretabilityScore
from hcga.features.utils import ensure_connected

featureclass_name = "DistanceMeasures"


def barycenter_size(graph):
    """barycenter_size"""
    return len(nx.barycenter(ensure_connected(graph)))


def barycenter_size_weighted(graph):
    """barycenter_size_weighted"""
    return len(nx.barycenter(ensure_connected(graph), weight="weight"))


def center_size(graph):
    """center_size"""
    return len(nx.center(ensure_connected(graph)))


def periphery(graph):
    """periphery"""
    return len(nx.periphery(ensure_connected(graph)))


def eccentricity(graph):
    """eccentricity"""
    return list(nx.eccentricity(ensure_connected(graph)).values())


def extrema_bounding(graph):
    """extrema_bounding"""
    return nx.diameter(ensure_connected(graph), usebounds=True)


class DistanceMeasures(FeatureClass):
    """Distance Measures class.

    Calculates features based on distance measures.

    Uses networkx, see 'https://networkx.org/documentation/stable/reference/algorithms/\
        distance_measures.html`
    """

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
            "extrema_bounding",
            extrema_bounding,
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
