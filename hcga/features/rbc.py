"""Role-similarity Based Comparison class."""
import networkx as nx
import numpy as np
from sklearn.preprocessing import normalize

from ..feature_class import FeatureClass, InterpretabilityScore
from .utils import ensure_connected, remove_selfloops

featureclass_name = "RolesimilarityBasedComparison"

"""
Create the role-similarity based comparison (rbc) matrix for nodes in the network,
then convert this to a graph and extract some features
ref: https://arxiv.org/abs/1103.5582
For some features we remove selfloops, since the diagonal of the rbc matrix
consists of ones, and therefore all nodes will have a selfloop with weight one
"""


def rbc(graph):
    """Rbc computation."""
    a = np.where(nx.adj_matrix(graph).toarray() > 0, 1, 0)
    g = nx.DiGraph(a)

    if nx.is_directed_acyclic_graph(g):
        k = nx.dag_longest_path_length(g)
        beta = 0.95

    else:
        lamb = max(np.linalg.eig(a)[0])
        if lamb != 0:
            beta = 0.95 / lamb
        else:
            beta = 0.95
        k = 10

    n = g.number_of_nodes()
    ones = np.ones(n)
    ba = beta * a
    ba_t = np.transpose(ba)

    x = np.zeros([n, k * 2])
    for i in range(1, k + 1):
        x[:, i - 1] = np.dot(np.linalg.matrix_power(ba, i), ones)
        x[:, i + k - 1] = np.dot(np.linalg.matrix_power(ba_t, i), ones)
    x_norm = normalize(x, axis=1)
    y = np.matmul(x_norm, np.transpose(x_norm))

    return nx.Graph(y)


class RolesimilarityBasedComparison(FeatureClass):
    """Role-similarity Based Comparison class."""

    modes = ["fast", "medium", "slow"]
    shortname = "RBC"
    name = "rbc"
    encoding = "networkx"

    def compute_features(self):

        g = rbc(self.graph)

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
