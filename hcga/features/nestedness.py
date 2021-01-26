"Nestness class"
import networkx as nx

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "Nestedness"

"A graph has a nested structure when smaller components contain a subset of larger components"


def nestedness_func(g):
    """Nestedness measures class

    Features based on the nestedness of a graph.
    A graph has a nested structure when smaller components contain a subset of larger components

    References
    ----------
    .. [1] Grimm, Alexander, and Claudio J. Tessone. "Detecting nestedness in graphs."
    International Workshop on Complex Networks and their Applications.
    Springer, Cham, 2016.    

    """
    n = nx.number_of_nodes(g)
    nodes = list(g.nodes())
    neighbors = [0 for i in range(n)]

    for i, j in enumerate(nodes):
        neighbors[i] = set(g.neighbors(j))

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
