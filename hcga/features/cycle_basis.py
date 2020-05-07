"""Cycle Basis class."""
from functools import lru_cache
import networkx as nx
import numpy as np

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "CycleBasis"


class CycleBasis(FeatureClass):
    """Cycle Basis class."""

    modes = ["fast", "medium", "slow"]
    shortname = "CYB"
    name = "cycle_basis"
    encoding = "networkx"

    def compute_features(self):
        """Compute the cycle basis of the network.

        Computed statistics
        -----
        Put here the list of things that are computed, with corresponding names
        """

        @lru_cache(maxsize=None)
        def eval_cycle_basis(graph):
            """this evaluates the main function and cach it for speed up."""
            return nx.cycle_basis(graph)

        self.add_feature(
            "num_cycles",
            lambda graph: len(eval_cycle_basis(graph)),
            "The total number of cycles in the graph",
            InterpretabilityScore(3),
        )

        self.add_feature(
            "average_cycle_length",
            lambda graph: np.mean([len(l) for l in eval_cycle_basis(graph)]),
            "The average length of cycles in the graph",
            InterpretabilityScore(3),
        )

        self.add_feature(
            "minimum_cycle_length",
            lambda graph: np.min([len(l) for l in eval_cycle_basis(graph)]),
            "The minimum length of cycles in the graph",
            InterpretabilityScore(3),
        )

        ratio_nodes_cycle = lambda graph: len(
            np.unique([x for l in eval_cycle_basis(graph) for x in l])
        ) / len(graph)
        self.add_feature(
            "ratio_nodes_cycle",
            ratio_nodes_cycle,
            "The ratio of nodes that appear in at least one cycle to the total number of nodes in the graph",
            InterpretabilityScore(3),
        )
