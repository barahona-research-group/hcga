"""Clustering class."""
import networkx as nx
import numpy as np

from hcga.feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "Clustering"


def triang(graph):
    """triang"""
    return np.asarray(list(nx.triangles(graph).values())).mean()


def clustering_dist(graph):
    """clustering_dist"""
    return list(nx.clustering(graph).values())


def square_clustering_dist(graph):
    """square_clustering_dist"""
    return list(nx.square_clustering(graph).values())


class Clustering(FeatureClass):
    """Clustering class."""

    modes = ["medium", "slow"]
    shortname = "Clu"
    name = "clustering"
    encoding = "networkx"

    def compute_features(self):

        self.add_feature(
            "num_triangles",
            triang,
            "Number of triangles in the graph",
            InterpretabilityScore("max"),
        )

        self.add_feature(
            "transitivity",
            nx.transitivity,
            "Transitivity of the graph",
            InterpretabilityScore("max"),
        )

        # Average clustering coefficient
        self.add_feature(
            "clustering",
            clustering_dist,
            "the clustering of the graph",
            InterpretabilityScore("max"),
            statistics="centrality",
        )

        # generalised degree
        self.add_feature(
            "square_clustering",
            square_clustering_dist,
            "the square clustering of the graph",
            InterpretabilityScore("max"),
            statistics="centrality",
        )
