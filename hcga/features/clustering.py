"""Clustering class."""
import networkx as nx
import numpy as np

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "Clustering"


class Clustering(FeatureClass):
    """Clustering class."""

    modes = ["medium", "slow"]
    shortname = "Clu"
    name = "clustering"
    encoding = "networkx"

    def compute_features(self):
        """Compute the various clustering measures.

        Notes
        -----
        Implementation of networkx code:
            `Networkx_clustering <https://networkx.github.io/documentation/stable\
            /reference/algorithms/clustering.html>`_

        We followed the same structure as networkx for implementing clustering features.
        """

        triang = lambda graph: np.asarray(list(nx.triangles(graph).values())).mean()
        self.add_feature(
            "num_triangles",
            triang,
            "Number of triangles in the graph",
            InterpretabilityScore("max"),
        )

        transi = lambda graph: nx.transitivity(graph)
        self.add_feature(
            "transitivity",
            transi,
            "Transitivity of the graph",
            InterpretabilityScore("max"),
        )

        # Average clustering coefficient
        clustering_dist = lambda graph: list(nx.clustering(graph).values())
        self.add_feature(
            "clustering",
            clustering_dist,
            "the clustering of the graph",
            InterpretabilityScore("max"),
            statistics="centrality",
        )

        # generalised degree
        square_clustering_dist = lambda graph: list(
            nx.square_clustering(graph).values()
        )
        self.add_feature(
            "square_clustering",
            square_clustering_dist,
            "the square clustering of the graph",
            InterpretabilityScore("max"),
            statistics="centrality",
        )
