"""Cycle Basis class."""

from functools import lru_cache

import networkx as nx
import numpy as np

from hcga.feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "CycleBasis"


@lru_cache(maxsize=None)
def eval_cycle_basis(graph):
    """this evaluates the main function and cach it for speed up."""
    return nx.cycle_basis(graph)


def num_cycles(graph):
    """num_cycles"""
    return len(eval_cycle_basis(graph))


def average_cycle_length(graph):
    """average_cycle_length"""
    return np.mean([len(cycle) for cycle in eval_cycle_basis(graph)])


def minimum_cycle_length(graph):
    """Minimum cycle length returns 0 if graph is a tree."""
    if not eval_cycle_basis(graph):
        return 0
    return np.min([len(cycle) for cycle in eval_cycle_basis(graph)])


def ratio_nodes_cycle(graph):
    """ratio_nodes_cycle"""
    return len(np.unique([_n for cycle in eval_cycle_basis(graph) for _n in cycle])) / len(graph)


class CycleBasis(FeatureClass):
    """Cycle Basis class.

    Computes features based on the cycles in the graph.

    A basis for cycles of a network is a minimal collection of cycles
    such that any cycle in the network can be written as a sum of cycles in the basis.
    Here summation of cycles is defined as “exclusive or” of the edges.
    Cycle bases are useful, e.g. when deriving equations for electric circuits
    using Kirchhoff’s Laws.

    Uses networkx, see 'https://networkx.org/documentation/stable/reference/algorithms/generated/\
        networkx.algorithms.cycles.cycle_basis.html`

    References
    ----------
    .. [1] Paton, K. An algorithm for finding a fundamental set of
       cycles of a graph. Comm. ACM 12, 9 (Sept 1969), 514-518.

    """

    modes = ["fast", "medium", "slow"]
    shortname = "CYB"
    name = "cycle_basis"
    encoding = "networkx"

    def compute_features(self):
        self.add_feature(
            "num_cycles",
            num_cycles,
            "The total number of cycles in the graph",
            InterpretabilityScore(3),
        )

        self.add_feature(
            "average_cycle_length",
            average_cycle_length,
            "The average length of cycles in the graph",
            InterpretabilityScore(3),
        )

        self.add_feature(
            "minimum_cycle_length",
            minimum_cycle_length,
            "The minimum length of cycles in the graph",
            InterpretabilityScore(3),
        )

        self.add_feature(
            "ratio_nodes_cycle",
            ratio_nodes_cycle,
            "The ratio of nodes that appear in at least one cycle to the total number of nodes",
            InterpretabilityScore(3),
        )
