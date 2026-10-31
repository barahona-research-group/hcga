"""Shortest paths class."""

from functools import lru_cache

import networkx as nx
import numpy as np

from hcga.feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "ShortestPaths"

# pylint: disable=no-value-for-parameter


@lru_cache(maxsize=1)
def eval_shortest_paths(graph):
    """All-pairs shortest path lengths, computed once per graph.

    Edge attribute 'weight' is used as the distance (hcga sets weight=1 on unweighted graphs,
    so lengths are then hop counts). Unreachable pairs are absent.
    """
    return dict(nx.all_pairs_dijkstra_path_length(graph, weight="weight"))


def largest_shortest_path(graph):
    """Longest shortest path from each node to the other nodes it can reach."""
    return [
        max(d for v, d in dists.items() if v != u)
        for u, dists in eval_shortest_paths(graph).items()
        if len(dists) > 1
    ]


def mean_shortest_path(graph):
    """Mean shortest path from each node to the other nodes it can reach."""
    return [
        np.mean([d for v, d in dists.items() if v != u])
        for u, dists in eval_shortest_paths(graph).items()
        if len(dists) > 1
    ]


def shortest_paths(graph):
    """Shortest path lengths between all ordered pairs of distinct, connected nodes."""
    return [
        d for u, dists in eval_shortest_paths(graph).items() for v, d in dists.items() if v != u
    ]


def number_shortest_paths_directed(graph):
    """"""
    return len(shortest_paths(graph))


def eccentricity(graph):
    """Eccentricity of each node that reaches at least one other node."""
    return [
        max(dists.values()) for dists in eval_shortest_paths(graph).values() if len(dists) > 1
    ]


def diameter_directed(graph):
    """"""
    return max(eccentricity(graph))


def radius_directed(graph):
    """"""
    return min(eccentricity(graph))


class ShortestPaths(FeatureClass):
    """Shortest paths class.

    Features based on the shortest paths across the network.

    All features use the edge attribute 'weight' as the path length (hop counts on unweighted
    graphs) and ignore unreachable pairs.

    Shortest paths calculations using networkx:
        `Networkx_scale free <https://networkx.org/documentation/stable/reference/algorithms/\
            shortest_paths.html>`_

    """

    modes = ["fast", "medium", "slow"]
    shortname = "SP"
    name = "shortest_paths"
    encoding = "networkx"

    def compute_features(self):
        # the longest path for each node
        self.add_feature(
            "largest_shortest_path",
            largest_shortest_path,
            "For each node we compute the shortest paths to every other node. \
            We then find the longest 'shortest path' for each node.",
            InterpretabilityScore(3),
            statistics="centrality",
        )

        # the mean shortest path for each node
        self.add_feature(
            "mean_shortest_path",
            mean_shortest_path,
            "For each node we compute the shortest paths to every other node. \
            We then find the mean of the 'shortest paths' for each node.",
            InterpretabilityScore(3),
            statistics="centrality",
        )
        self.add_feature(
            "number_shortest_paths_directed",
            number_shortest_paths_directed,
            "the number of shortest paths (not counting paths of infinite length)",
            InterpretabilityScore(3),
        )

        self.add_feature(
            "shortest_paths_length_directed",
            shortest_paths,
            "the number of shortest path lengths (not counting paths of infinite length)",
            InterpretabilityScore(3),
            statistics="centrality",
        )
        self.add_feature(
            "eccentricity_directed",
            eccentricity,
            "the ditribution of eccentricities (not counting paths of infinite length)",
            InterpretabilityScore(3),
            statistics="centrality",
        )

        self.add_feature(
            "diameter_directed",
            diameter_directed,
            "the diameter of the graph (not counting paths of infinite length)",
            InterpretabilityScore(3),
        )

        self.add_feature(
            "radius_directed",
            radius_directed,
            "the radius of the graph (not counting paths of infinite length)",
            InterpretabilityScore(3),
        )
