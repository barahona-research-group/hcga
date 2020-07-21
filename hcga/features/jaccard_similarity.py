"""Jaccard Similarity class."""
import networkx as nx
from networkx.algorithms import centrality
import numpy as np

from .utils import ensure_connected, remove_selfloops
from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "JaccardSimilarity"

"""
Create the Jaccard similarity matrix for nodes in the network,
then convert this to a graph and extract some features
This is defined as a/(a+b+c), where
a = number of common neighbours
b = number of neighbours of node 1 that are not neighbours of node 2
c = number of neighbours of node 2 that are not neighbours of node 1
Treating this matrix as an adjacency matrix, we can compute network some features
ref: https://www.biorxiv.org/content/10.1101/112540v4.full

For some features we remove selfloops, since the diagonal of the Jaccard
similarity consists of ones, and therefore all nodes will have a selfloop with weight one
"""


def jaccard_similarity(graph):
    """Construct a graph from Jaccard similarity matrix"""

    n = nx.number_of_nodes(graph)
    jsm = np.eye(n)

    neighbors = [0 for i in range(n)]

    for j in range(n):
        neighbors[j] = set(graph.neighbors(j))

    for i in range(n):
        for j in range(i + 1, n):
            a = len(neighbors[i].intersection(neighbors[j]))
            if a == 0:
                jsm[i, j] = 0
            else:
                b = len(neighbors[i].difference(neighbors[j]))
                c = len(neighbors[j].difference(neighbors[i]))
                jsm[i, j] = a / (a + b + c)

    return nx.Graph(jsm)


class JaccardSimilarity(FeatureClass):
    """Jaccard Similarity class."""

    modes = ["fast", "medium", "slow"]
    shortname = "JS"
    name = "jaccard_similarity"
    encoding = "networkx"

    def compute_features(self):

        g = jaccard_similarity(self.graph)

        # Basic stats
        self.add_feature(
            "number_of_edges",
            lambda graph: graph.number_of_edges(),
            "Number of edges in Jaccard similarity graph",
            InterpretabilityScore(5),
            function_args=g,
        )

        self.add_feature(
            "number_of_edges_no_selfloops",
            lambda graph: remove_selfloops(graph).number_of_edges(),
            "Number of edges, not including selfloops, in Jaccard similarity graph",
            InterpretabilityScore(5),
            function_args=g,
        )

        self.add_feature(
            "connectance",
            lambda graph: nx.density(graph),
            "Connectance of Jaccard similarity graph",
            InterpretabilityScore(5),
            function_args=g,
        )

        self.add_feature(
            "diameter",
            lambda graph: nx.diameter(ensure_connected(graph)),
            "Diameter of Jaccard similarity graph",
            InterpretabilityScore(5),
            function_args=g,
        )

        self.add_feature(
            "radius",
            lambda graph: nx.radius(ensure_connected(graph)),
            "Radius of Jaccard similarity graph",
            InterpretabilityScore(5),
            function_args=g,
        )

        self.add_feature(
            "edge_weights",
            lambda graph: list(
                nx.get_edge_attributes(remove_selfloops(graph), "weight").values()
            ),
            "Weights of the edges in Jaccard similarity graph",
            InterpretabilityScore(5),
            function_args=g,
            statistics="centrality",
        )

        # Assortativity
        self.add_feature(
            "degree_assortativity_coeff",
            lambda graph: nx.degree_assortativity_coefficient(graph),
            "Similarity of connections in Jaccard similarity graph with respect to the node degree",
            InterpretabilityScore(4),
            function_args=g,
        )

        # Wiener index
        self.add_feature(
            "wiener index",
            lambda graph: nx.wiener_index(graph),
            "The wiener index is defined as the sum of the lengths of the shortest paths \
            between all pairs of vertices",
            InterpretabilityScore(4),
            function_args=g,
        )

        # Centralities
        self.add_feature(
            "degree centrality",
            lambda graph: list(nx.degree_centrality(graph).values()),
            "The degree centrality distribution",
            InterpretabilityScore(5),
            function_args=g,
            statistics="centrality",
        )

        self.add_feature(
            "eigenvector centrality",
            lambda graph: list(
                centrality.eigenvector_centrality_numpy(
                    ensure_connected(graph)
                ).values()
            ),
            "Eigenvector centrality computes the centrality for a node based \
            on the centrality of its neighbors",
            InterpretabilityScore(4),
            function_args=g,
            statistics="centrality",
        )
