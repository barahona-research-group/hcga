"""Simple cycles class."""

from collections import Counter
from functools import lru_cache

import networkx as nx

from hcga.feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "SimpleCycles"

# Only cycles up to this length are enumerated: the number of simple cycles grows
# exponentially with E - N, so unbounded enumeration does not finish on modest graphs.
LENGTH_BOUND = 6


@lru_cache(maxsize=None)
def simple_cycles_func(graph):
    """Counter of the lengths of the simple cycles of length at most LENGTH_BOUND."""
    return Counter(len(c) for c in nx.simple_cycles(graph, length_bound=LENGTH_BOUND))


def number_simple_cycles(graph):
    """number_simple_cycles"""
    return sum(simple_cycles_func(graph).values())


def simple_cycles_sizes(graph):
    """simple_cycles_sizes"""
    return sorted(simple_cycles_func(graph).elements())


class SimpleCycles(FeatureClass):
    """Simple cycles class.

    Computes features based on simple cycles (elementary circuits) of a graph.
    Only cycles of length at most LENGTH_BOUND (= 6) are counted, since the total number of
    simple cycles grows exponentially with the number of edges.

    A `simple cycle`, or `elementary circuit`, is a closed path where
    no node appears twice. Two elementary circuits are distinct if they
    are not cyclic permutations of each other.

    Cycles calculations using networkx:
        `Cycles <https://networkx.org/documentation/stable/_modules/networkx/algorithms/\
            cycles.html>`_


    References
    ----------
    .. [1] Finding all the elementary circuits of a directed graph.
       D. B. Johnson, SIAM Journal on Computing 4, no. 1, 77-84, 1975.
       https://doi.org/10.1137/0204007
    .. [2] Enumerating the cycles of a digraph: a new preprocessing strategy.
       G. Loizou and P. Thanish, Information Sciences, v. 27, 163-182, 1982.
    .. [3] A search strategy for the elementary cycles of a directed graph.
       J.L. Szwarcfiter and P.E. Lauer, BIT NUMERICAL MATHEMATICS,
       v. 16, no. 2, 192-204, 1976.

    """

    modes = ["slow"]
    shortname = "SC"
    name = "simple_cycles"
    encoding = "networkx"

    def compute_features(self):
        self.add_feature(
            "number_simple_cycles",
            number_simple_cycles,
            "The number of simple cycles (closed paths with no repeated nodes) of length at most 6",
            InterpretabilityScore(3),
        )
        self.add_feature(
            "simple_cycles_sizes",
            simple_cycles_sizes,
            "the distribution of simple cycle lengths (cycles of length at most 6)",
            InterpretabilityScore(3),
            statistics="centrality",
        )
