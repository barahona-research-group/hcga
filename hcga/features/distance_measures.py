"""Distance Measures class."""

from functools import lru_cache

import networkx as nx

from hcga.feature_class import FeatureClass, InterpretabilityScore
from hcga.features.utils import ensure_connected

featureclass_name = "DistanceMeasures"


@lru_cache(maxsize=1)
def _connected_undirected(graph):
    """Largest connected component of the graph (made undirected for digraphs)."""
    if graph.is_directed():
        graph = graph.to_undirected()
    return ensure_connected(graph)


@lru_cache(maxsize=1)
def _shortest_path_lengths(graph):
    """Unweighted all-pairs shortest path lengths on the largest connected component."""
    return dict(nx.shortest_path_length(_connected_undirected(graph)))


@lru_cache(maxsize=1)
def _eccentricity(graph):
    """Eccentricity of each node of the largest connected component."""
    return nx.eccentricity(_connected_undirected(graph), sp=_shortest_path_lengths(graph))


def barycenter_size(graph):
    """barycenter_size"""
    return len(nx.barycenter(_connected_undirected(graph), sp=_shortest_path_lengths(graph)))


def barycenter_size_weighted(graph):
    """barycenter_size_weighted"""
    return len(nx.barycenter(_connected_undirected(graph), weight="weight"))


def center_size(graph):
    """center_size"""
    return len(nx.center(_connected_undirected(graph), e=_eccentricity(graph)))


def periphery(graph):
    """periphery"""
    return len(nx.periphery(_connected_undirected(graph), e=_eccentricity(graph)))


def eccentricity(graph):
    """eccentricity"""
    return list(_eccentricity(graph).values())


def extrema_bounding(graph):
    """extrema_bounding (the diameter, here from the shared eccentricities)"""
    return nx.diameter(_connected_undirected(graph), e=_eccentricity(graph))


class DistanceMeasures(FeatureClass):
    """Distance Measures class.

    Calculates features based on distance measures, on the largest connected component
    (of the undirected graph, for digraphs).

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
