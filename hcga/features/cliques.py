"""Cliques class."""

from functools import lru_cache

import numpy as np
from networkx.algorithms import clique

from hcga.feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "Cliques"


def n_cliques(graph):
    """n_cliques"""
    return len([u for u in list(clique.enumerate_all_cliques(graph)) if len(u) > 1])


def clique_sizes(graph):
    """clique_sizes"""
    return [len(u) for u in list(clique.enumerate_all_cliques(graph)) if len(u) > 1]


@lru_cache(maxsize=None)
def eval_cliques(graph):
    """this evaluates the main function and cach it for speed up."""
    cliques = [len(u) for u in list(clique.find_cliques(graph)) if len(u) > 1]
    return np.bincount(cliques)[np.nonzero(np.bincount(cliques))]


def maximal_clique_sizes(graph):
    """maximal_clique_sizes"""
    return eval_cliques(graph)[0] / eval_cliques(graph)[-1]


class Cliques(FeatureClass):
    """Cliques class.

    Here we construct features based on cliques
    (subsets of vertices, all adjacent to each other, also called complete subgraphs).

    References
    ----------
    .. [1] Bron, C. and Kerbosch, J.
       "Algorithm 457: finding all cliques of an undirected graph".
       *Communications of the ACM* 16, 9 (Sep. 1973), 575--577.
       <http://portal.acm.org/citation.cfm?doid=362342.362367>
    .. [2] F. Cazals, C. Karande,
       "A note on the problem of reporting maximal cliques",
       *Theoretical Computer Science*,
       Volume 407, Issues 1--3, 6 November 2008, Pages 564--568,
       <https://doi.org/10.1016/j.tcs.2008.05.010>
    .. [3] Yun Zhang, Abu-Khzam, F.N., Baldwin, N.E., Chesler, E.J.,
           Langston, M.A., Samatova, N.F.,
           "Genome-Scale Computational Approaches to Memory-Intensive
           Applications in Systems Biology".
           *Supercomputing*, 2005. Proceedings of the ACM/IEEE SC 2005
           Conference, pp. 12, 12--18 Nov. 2005.
           <https://doi.org/10.1109/SC.2005.29>.
    """

    modes = ["fast", "medium", "slow"]
    shortname = "Cli"
    name = "cliques"
    encoding = "networkx"

    def compute_features(self):
        # graph clique number
        self.add_feature(
            "graph_clique_number",
            lambda graph: max(len(c) for c in clique.find_cliques(graph)),
            "The clique number of a graph is the size of the largest clique in the graph",
            InterpretabilityScore(3),
        )

        # number of maximal cliques
        self.add_feature(
            "num_max_cliques",
            lambda graph: sum(1 for _ in clique.find_cliques(graph)),
            "The number of maximal cliques in the graph",
            InterpretabilityScore(3),
        )

        self.add_feature(
            "num_cliques",
            n_cliques,
            "The number of cliques in the graph",
            InterpretabilityScore(3),
        )

        self.add_feature(
            "clique_sizes",
            clique_sizes,
            "the distribution of clique sizes",
            InterpretabilityScore(3),
            statistics="centrality",
        )

        self.add_feature(
            "clique_sizes_maximal",
            maximal_clique_sizes,
            "the ratio of number of max and min size cliques",
            InterpretabilityScore(3),
        )
