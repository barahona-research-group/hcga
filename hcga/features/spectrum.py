"""Spectrum class."""

from functools import lru_cache, partial

import networkx as nx
import numpy as np
import scipy.linalg

from hcga.feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "Spectrum"


def _sorted_eigenvalues(matrix, directed):
    """Eigenvalues in descending order (of real part, for non-symmetric matrices)."""
    if directed:
        eigenvalues = np.linalg.eigvals(matrix)
        return eigenvalues[np.argsort(-eigenvalues.real, kind="stable")].real
    return scipy.linalg.eigvalsh(matrix)[::-1]


@lru_cache(maxsize=1)
def eval_spectrum_adj(graph):
    """Eigenvalues of the weighted adjacency matrix, in descending order."""
    return _sorted_eigenvalues(nx.to_numpy_array(graph, weight="weight"), graph.is_directed())


def eigenvalue_ratio(graph, i, j):
    """Ratio of the i-th to the j-th largest adjacency eigenvalue."""
    return eval_spectrum_adj(graph)[i] / eval_spectrum_adj(graph)[j]


@lru_cache(maxsize=1)
def eval_spectrum_modularity(graph):
    """"""
    return np.real(nx.linalg.spectrum.modularity_spectrum(graph))


@lru_cache(maxsize=1)
def eval_spectrum_laplacian(graph):
    """Eigenvalues of the weighted Laplacian (out-degree Laplacian for digraphs).

    The directed Laplacian is not symmetric, so eigvalsh (used by nx.laplacian_spectrum)
    does not apply; we take the real parts of its eigenvalues.
    """
    laplacian = nx.laplacian_matrix(graph, weight="weight").toarray()
    return _sorted_eigenvalues(laplacian, graph.is_directed())


class Spectrum(FeatureClass):
    """Spectrum class.

    Features based on the eigenvectors and eigenvalues of the graph.

    The features here are derived from spectral graph theory [1]_.

    References
    ----------
    .. [1] Cvetkovic, Dragos, et al. Eigenspaces of graphs. No. 66. Cambridge University Press,\
        1997.

    """

    modes = ["medium", "slow"]
    shortname = "SPM"
    name = "spectrum"
    encoding = "networkx"

    def compute_features(self):
        # distribution of eigenvalues
        self.add_feature(
            "eigenvalues_adjacency",
            eval_spectrum_adj,
            "The summary statistics of eigenvalues of adjacency matrix",
            InterpretabilityScore(3),
            statistics="centrality",
        )

        # get ratio of eigenvalues
        n_eigs = 10
        for i in range(n_eigs):
            for j in range(i):
                self.add_feature(
                    f"eigenvalue_ratio_{i}_{j}",
                    partial(eigenvalue_ratio, i=i, j=j),
                    f"The ratio of the {i} and {j} eigenvalues",
                    InterpretabilityScore(2),
                )
        # distribution of eigenvalues
        self.add_feature(
            "eigenvalues_modularity",
            eval_spectrum_modularity,
            "The summary statistics of eigenvalues of modularity matrix",
            InterpretabilityScore(3),
            statistics="centrality",
        )

        # distribution of eigenvalues
        self.add_feature(
            "eigenvalues_laplacian",
            eval_spectrum_laplacian,
            "The summary statistics of eigenvalues of laplacian matrix",
            InterpretabilityScore(3),
            statistics="centrality",
        )
