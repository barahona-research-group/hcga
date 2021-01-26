"""Clustering class."""
import networkx as nx
import numpy as np

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "Clustering"


class Clustering(FeatureClass):
    """Clustering class.

    Here we construct features based on the number of triangles in a graph.

    Uses networkx, see 'https://networkx.github.io/documentation/stable/reference/\
        algorithms/clustering.html`

    We compute:
    The number of triangles
    Transitivity [1]_
    Clustering [2]_[3]_[4]_

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
        square_clustering_dist = lambda graph: list(nx.square_clustering(graph).values())
        self.add_feature(
            "square_clustering",
            square_clustering_dist,
            "the square clustering of the graph",
            InterpretabilityScore("max"),
            statistics="centrality",
        )
