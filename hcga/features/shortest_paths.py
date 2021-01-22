"""Shortest paths class."""
from functools import lru_cache

import networkx as nx
import numpy as np

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "ShortestPaths"

# pylint: disable=no-value-for-parameter


class ShortestPaths(FeatureClass):
    """Shortest paths class."""

    modes = ["fast", "medium", "slow"]
    shortname = "SP"
    name = "shortest_paths"
    encoding = "networkx"

    def compute_features(self):
        @lru_cache(maxsize=None)
        def eval_shortest_paths(graph):
            return nx.shortest_path(graph)

        # the longest path for each node
        largest_shortest_path = lambda graph: [
            len(list(eval_shortest_paths(graph)[u].values())[-1])
            for u in eval_shortest_paths(graph)
        ]
        self.add_feature(
            "largest_shortest_path",
            largest_shortest_path,
            "For each node we compute the shortest paths to every other node. \
            We then find the longest 'shortest path' for each node.",
            InterpretabilityScore(3),
            statistics="centrality",
        )

        # the mean shortest path for each node
        mean_shortest_path = lambda graph: [
            np.mean([len(k) for k in list(eval_shortest_paths(graph)[u].values())])
            for u in eval_shortest_paths(graph)
        ]
        self.add_feature(
            "mean_shortest_path",
            mean_shortest_path,
            "For each node we compute the shortest paths to every other node. \
            We then find the mean of the 'shortest paths' for each node.",
            InterpretabilityScore(3),
            statistics="centrality",
        )

        def shortest_paths(graph):
            ss = list(nx.all_pairs_dijkstra_path_length(graph))
            p = []
            for s in ss:
                p += list(s[1].values())
            return [j for j in p if j > 0]

        self.add_feature(
            "number_shortest_paths_directed",
            lambda graph: len(shortest_paths(graph)),
            "the number of shortest paths (not counting paths of infinite length)",
            InterpretabilityScore(3),
        )

        self.add_feature(
            "shortest_paths_length_directed",
            lambda graph: shortest_paths(graph),
            "the number of shortest path lengths (not counting paths of infinite length)",
            InterpretabilityScore(3),
            statistics="centrality",
        )

        def eccentricity(graph):
            ss = list(nx.all_pairs_dijkstra_path_length(graph))
            p = []
            for s in ss:
                p.append(max(list(s[1].values())))
            return p

        self.add_feature(
            "eccentricity_directed",
            lambda graph: eccentricity(graph),
            "the ditribution of eccentricities (not counting paths of infinite length)",
            InterpretabilityScore(3),
            statistics="centrality",
        )

        self.add_feature(
            "diameter_directed",
            lambda graph: max(eccentricity(graph)),
            "the diameter of the graph (not counting paths of infinite length)",
            InterpretabilityScore(3),
        )

        self.add_feature(
            "radius_directed",
            lambda graph: min(eccentricity(graph)),
            "the radius of the graph (not counting paths of infinite length)",
            InterpretabilityScore(3),
        )
