"""Clustering class."""

import networkx as nx

from hcga.feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "Clustering"


def _undirected(graph):
    """Undirected view of a directed graph (triangles and squares are undirected notions)."""
    if graph.is_directed():
        return graph.to_undirected(as_view=True)
    return graph


def triang(graph):
    """Total number of triangles (each triangle is counted at its three nodes, hence // 3)."""
    return sum(nx.triangles(_undirected(graph)).values()) // 3


def transitivity(graph):
    """transitivity"""
    return nx.transitivity(_undirected(graph))


def clustering_dist(graph):
    """clustering_dist"""
    return list(nx.clustering(graph).values())


def square_clustering_dist(graph):
    """square_clustering_dist"""
    return list(nx.square_clustering(_undirected(graph)).values())


class Clustering(FeatureClass):
    """Clustering class.

    Here we construct features based on the number of triangles in a graph.

    Uses networkx, see 'https://networkx.github.io/documentation/stable/reference/\
        algorithms/clustering.html`

    We compute:
    The number of triangles
    Transitivity [1]_
    Clustering [2]_ [3]_ [4]_

    References
    ----------
    .. [1]  Biggs, Norman (1993).
       Algebraic Graph Theory (2nd ed.).
       Cambridge: Cambridge University Press. p. 118.
    .. [2] Generalizations of the clustering coefficient to weighted
       complex networks by J. Saramäki, M. Kivelä, J.-P. Onnela,
       K. Kaski, and J. Kertész, Physical Review E, 75 027105 (2007).
       http://jponnela.com/web_documents/a9.pdf
    .. [3] Intensity and coherence of motifs in weighted complex
       networks by J. P. Onnela, J. Saramäki, J. Kertész, and K. Kaski,
       Physical Review E, 71(6), 065103 (2005).
    .. [4] Clustering in complex directed networks by G. Fagiolo,
       Physical Review E, 76(2), 026107 (2007).

    """

    modes = ["medium", "slow"]
    shortname = "Clu"
    name = "clustering"
    encoding = "networkx"

    def compute_features(self):
        self.add_feature(
            "num_triangles",
            triang,
            "Total number of triangles in the graph (undirected for directed graphs)",
            InterpretabilityScore("max"),
        )

        self.add_feature(
            "transitivity",
            transitivity,
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
