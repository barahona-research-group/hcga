"""Spectrum class."""
from functools import lru_cache

import networkx as nx
import numpy as np

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "Spectrum"


class Spectrum(FeatureClass):
    """Spectrum class."""

    modes = ["medium", "slow"]
    shortname = "SPM"
    name = "spectrum"
    encoding = "networkx"

    def compute_features(self):
        @lru_cache(maxsize=None)
        def eval_spectrum_adj(graph):
            return np.real(nx.linalg.spectrum.adjacency_spectrum(graph))

        # distribution of eigenvalues
        self.add_feature(
            "eigenvalues_adjacency",
            lambda graph: eval_spectrum_adj(graph),
            "The summary statistics of eigenvalues of adjacency matrix",
            InterpretabilityScore(3),
            statistics="centrality",
        )

        # get ratio of eigenvalues
        n_eigs = 10
        for i in range(n_eigs):
            for j in range(i):
                self.add_feature(
                    "eigenvalue_ratio_{}_{}".format(i, j),
                    lambda graph: eval_spectrum_adj(graph)[j] / eval_spectrum_adj(graph)[i],
                    "The ratio of the {} and {} eigenvalues".format(i, j),
                    InterpretabilityScore(2),
                )

        @lru_cache(maxsize=None)
        def eval_spectrum_modularity(graph):
            return np.real(nx.linalg.spectrum.modularity_spectrum(graph))

        # distribution of eigenvalues
        self.add_feature(
            "eigenvalues_modularity",
            lambda graph: eval_spectrum_modularity(graph),
            "The summary statistics of eigenvalues of modularity matrix",
            InterpretabilityScore(3),
            statistics="centrality",
        )

        @lru_cache(maxsize=None)
        def eval_spectrum_laplacian(graph):
            return np.real(nx.linalg.spectrum.laplacian_spectrum(graph))

        # distribution of eigenvalues
        self.add_feature(
            "eigenvalues_laplacian",
            lambda graph: eval_spectrum_laplacian(graph),
            "The summary statistics of eigenvalues of laplacian matrix",
            InterpretabilityScore(3),
            statistics="centrality",
        )
