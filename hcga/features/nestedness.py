"Nestness class"
import networkx as nx
import numpy as np

from hcga.feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "Nestedness"

"A graph has a nested structure when smaller components contain a subset of larger components"


def nestedness(graph):
    """Nestedness measures class

    Features based on the nestedness of a graph.
    A graph has a nested structure when smaller components contain a subset of larger components

    References
    ----------
    .. [1] Grimm, Alexander, and Claudio J. Tessone. "Detecting nestedness in graphs."
    International Workshop on Complex Networks and their Applications.
    Springer, Cham, 2016.

    """
    # eta = sum_{i<j} |N(i) & N(j)| / sum_{i<j} min(k_i, k_j), with N(i) the neighbours
    # (successors for directed graphs), computed with sparse matrix products.
    if graph.number_of_nodes() == 0:
        return 0
    adjacency = nx.to_scipy_sparse_array(graph, weight=None, format="csr")
    adjacency.data[:] = 1.0
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    n = len(degrees)

    # common neighbours of all pairs i < j: off-diagonal half of A A^T
    common = adjacency @ adjacency.T
    sum_n_ij = (common.sum() - common.diagonal().sum()) / 2

    # sum over pairs of the smaller degree: the i-th smallest degree is the minimum of n-1-i pairs
    sorted_degrees = np.sort(degrees)
    sum_n_m = np.sum(sorted_degrees * (n - 1 - np.arange(n)))
    if sum_n_m == 0:
        return 0
    return float(sum_n_ij / sum_n_m)


class Nestedness(FeatureClass):
    """Nestedness class."""

    modes = ["fast", "medium", "slow"]
    shortname = "Nes"
    name = "nestedness"
    encoding = "networkx"

    def compute_features(self):
        self.add_feature(
            "nestedness",
            nestedness,
            "A measure of the nested structure of the network",
            InterpretabilityScore(3),
        )
