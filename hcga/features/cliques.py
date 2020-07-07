"""Cliques class."""
from functools import lru_cache

import numpy as np
from networkx.algorithms import clique
from networkx import to_undirected

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "Cliques"


class Cliques(FeatureClass):
    """Cliques class."""

    modes = ["fast", "medium", "slow"]
    shortname = "Cli"
    name = "cliques"
    encoding = "networkx"

    def compute_features(self):
        
        # graph clique number
        self.add_feature(
            "graph_clique_number",
            lambda graph: clique.graph_clique_number(to_undirected(graph)),
            "The clique number of a graph is the size of the largest clique in the graph",
            InterpretabilityScore(3),
        )

        # number of maximal cliques
        self.add_feature(
            "num_max_cliques",
            lambda graph: clique.graph_number_of_cliques(to_undirected(graph)),
            "The number of maximal cliques in the graph",
            InterpretabilityScore(3),
        )

        n_cliques = lambda graph: len(
            [u for u in list(clique.enumerate_all_cliques(to_undirected(graph))) if len(u) > 1]
        )
        self.add_feature(
            "num_cliques",
            n_cliques,
            "The number of cliques in the graph",
            InterpretabilityScore(3),
        )

        clique_sizes = lambda graph: [
            len(u) for u in list(clique.enumerate_all_cliques(to_undirected(graph))) if len(u) > 1
        ]
        self.add_feature(
            "clique_sizes",
            clique_sizes,
            "the distribution of clique sizes",
            InterpretabilityScore(3),
            statistics="centrality",
        )

        @lru_cache(maxsize=None)
        def eval_cliques(graph):
            """this evaluates the main function and cach it for speed up."""
            cliques = [len(u) for u in list(clique.find_cliques(to_undirected(graph))) if len(u) > 1]
            return np.bincount(cliques)[np.nonzero(np.bincount(cliques))]

        maximal_clique_sizes = (
            lambda graph: eval_cliques(graph)[0] / eval_cliques(graph)[-1]
        )
        self.add_feature(
            "clique_sizes_maximal",
            maximal_clique_sizes,
            "the ratio of number of max and min size cliques",
            InterpretabilityScore(3),
        )
