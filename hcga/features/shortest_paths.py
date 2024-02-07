"""Shortest paths class."""

from functools import lru_cache

import networkx as nx
import numpy as np

from hcga.feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "ShortestPaths"

# pylint: disable=no-value-for-parameter


@lru_cache(maxsize=None)
def eval_shortest_paths(graph):
    """eval_shortest_paths"""
    return nx.shortest_path(graph)


def largest_shortest_path(graph):
    """"""
    return [
        len(list(eval_shortest_paths(graph)[u].values())[-1]) for u in eval_shortest_paths(graph)
    ]


def mean_shortest_path(graph):
    """"""
    return [
        np.mean([len(k) for k in list(eval_shortest_paths(graph)[u].values())])
        for u in eval_shortest_paths(graph)
    ]


@lru_cache(maxsize=None)
def shortest_paths(graph):
    """"""
    ss = list(nx.all_pairs_dijkstra_path_length(graph))
    p = []
    for s in ss:
        p += list(s[1].values())
    return [j for j in p if j > 0]


def number_shortest_paths_directed(graph):
    """"""
    return len(shortest_paths(graph))


@lru_cache(maxsize=None)
def eccentricity(graph):
    """"""
    ss = list(nx.all_pairs_dijkstra_path_length(graph))
    p = []
    for s in ss:
        p.append(max(list(s[1].values())))
    return p


def diameter_directed(graph):
    """"""
    return max(eccentricity(graph))


def radius_directed(graph):
    """"""
    min(eccentricity(graph))


class ShortestPaths(FeatureClass):
    """Shortest paths class.

    Features based on the shortest paths across the network.

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
