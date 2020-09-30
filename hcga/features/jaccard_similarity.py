"""Jaccard Similarity class."""
import networkx as nx
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

    for i, j in enumerate(graph.nodes()):
        neighbors[i] = set(graph.neighbors(j))

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

        # Assortativity
        self.add_feature(
            "degree_assortativity_coeff",
            lambda graph: nx.degree_assortativity_coefficient(graph),
            "Similarity of connections in Jaccard similarity graph with respect to the node degree",
            InterpretabilityScore(4),
            function_args=g,
        )

        # Cliques
        self.add_feature(
            "graph_clique_number",
            lambda graph: nx.graph_clique_number(graph),
            "The size of the largest clique in the Jaccard similarity graph",
            InterpretabilityScore(3),
            function_args=g,
        )

        self.add_feature(
            "num_max_cliques",
            lambda graph: nx.graph_number_of_cliques(graph),
            "The number of maximal cliques in the Jaccard similarity graph",
            InterpretabilityScore(3),
            function_args=g,
        )

        # Clustering
        self.add_feature(
            "transitivity",
            lambda graph: nx.transitivity(graph),
            "Transitivity of the graph",
            InterpretabilityScore(4),
            function_args=g,
        )

        # Components
        self.add_feature(
            "is_connected",
            lambda graph: nx.is_connected(graph) * 1,
            "Whether the Jaccard similarity graph is connected or not",
            InterpretabilityScore(5),
            function_args=g,
        )

        self.add_feature(
            "num_connected_components",
            lambda graph: nx.number_connected_components(graph),
            "The number of connected components",
            InterpretabilityScore(5),
            function_args=g,
        )

        self.add_feature(
            "largest_connected_component",
            lambda graph: ensure_connected(graph).number_of_nodes(),
            "The size of the largest connected component",
            InterpretabilityScore(4),
            function_args=g,
        )

        # Efficiency
        self.add_feature(
            "global_efficiency",
            lambda graph: nx.global_efficiency(graph),
            "The global efficiency",
            InterpretabilityScore(4),
            function_args=g,
        )

        # Node connectivity
        self.add_feature(
            "node_connectivity",
            lambda graph: nx.node_connectivity(graph),
            "Node connectivity",
            InterpretabilityScore(4),
            function_args=g,
        )

        self.add_feature(
            "edge_connectivity",
            lambda graph: nx.edge_connectivity(graph),
            "Edge connectivity",
            InterpretabilityScore(4),
            function_args=g,
        )
