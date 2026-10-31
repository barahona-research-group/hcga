"""Cliques class."""

from functools import lru_cache

import numpy as np
from networkx.algorithms import clique

from hcga.feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "Cliques"

# Cliques (complete subgraphs, not only maximal ones) are counted up to this many nodes:
# their total number grows exponentially with the clique number.
CLIQUE_SIZE_BOUND = 4


def _undirected(graph):
    """Cliques are defined on undirected graphs."""
    if graph.is_directed():
        return graph.to_undirected(as_view=True)
    return graph


@lru_cache(maxsize=None)
def eval_maximal_cliques(graph):
    """Sizes of all maximal cliques, from a single find_cliques pass."""
    return [len(c) for c in clique.find_cliques(_undirected(graph))]


@lru_cache(maxsize=None)
def eval_clique_counts(graph):
    """Number of cliques of each size k, for 2 <= k <= CLIQUE_SIZE_BOUND.

    Counts the same cliques as networkx's enumerate_all_cliques, but stops at
    CLIQUE_SIZE_BOUND nodes, and counts the largest size without listing those cliques.
    """
    graph = _undirected(graph)
    index = {node: i for i, node in enumerate(graph)}
    later_nbrs = {u: {v for v in graph[u] if index[v] > index[u]} for u in graph}
    counts = dict.fromkeys(range(2, CLIQUE_SIZE_BOUND + 1), 0)

    def _extend(size, candidates):
        # each candidate extends the current clique of size - 1 to a clique of this size
        counts[size] += len(candidates)
        if size == CLIQUE_SIZE_BOUND:
            return
        for v in candidates:
            _extend(size + 1, candidates & later_nbrs[v])

    for u in graph:
        _extend(2, later_nbrs[u])
    return counts


def n_cliques(graph):
    """Number of cliques with 2 to CLIQUE_SIZE_BOUND nodes."""
    return sum(eval_clique_counts(graph).values())


def clique_sizes(graph):
    """Sizes of all cliques with 2 to CLIQUE_SIZE_BOUND nodes."""
    counts = eval_clique_counts(graph)
    return np.repeat(list(counts.keys()), list(counts.values()))


def eval_cliques(graph):
    """Number of maximal cliques of each size present (size > 1), smallest size first."""
    cliques = [size for size in eval_maximal_cliques(graph) if size > 1]
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
            lambda graph: max(eval_maximal_cliques(graph)),
            "The clique number of a graph is the size of the largest clique in the graph",
            InterpretabilityScore(3),
        )

        # number of maximal cliques
        self.add_feature(
            "num_max_cliques",
            lambda graph: len(eval_maximal_cliques(graph)),
            "The number of maximal cliques in the graph",
            InterpretabilityScore(3),
        )

        self.add_feature(
            "num_cliques",
            n_cliques,
            "The number of cliques with 2 to 4 nodes in the graph",
            InterpretabilityScore(3),
        )

        self.add_feature(
            "clique_sizes",
            clique_sizes,
            "the distribution of clique sizes (cliques with 2 to 4 nodes)",
            InterpretabilityScore(3),
            statistics="centrality",
        )

        self.add_feature(
            "clique_sizes_maximal",
            maximal_clique_sizes,
            "the ratio of the numbers of smallest and largest maximal cliques",
            InterpretabilityScore(3),
        )
