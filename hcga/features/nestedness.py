"Nestness class"
import networkx as nx

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "Nestedness"

"A graph has a nested structure when smaller components contain a subset of larger components"


def nestedness_func(g):
    """Compute nestedness."""
    n = nx.number_of_nodes(g)

    neighbors = [0 for i in range(n)]
    for j in range(n):
        neighbors[j] = set(g.neighbors(j))

    sum_n_ij = 0
    sum_n_m = 0

    for j in range(1, n):
        for i in range(j):
            n_ij = len(neighbors[i].intersection(neighbors[j]))
            n_m = min(len(neighbors[i]), len(neighbors[j]))
            sum_n_ij += n_ij
            sum_n_m += n_m
    if sum_n_m == 0:
        return 0
    return sum_n_ij / sum_n_m


class Nestedness(FeatureClass):
    """Nestedness class."""

    modes = ["fast", "medium", "slow"]
    shortname = "Nes"
    name = "nestedness"
    encoding = "networkx"

    def compute_features(self):

        self.add_feature(
            "nestedness",
            lambda graph: nestedness_func(graph),
            "A measure of the nested structure of the network",
            InterpretabilityScore(3),
        )
