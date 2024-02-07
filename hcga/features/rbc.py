"""Role-similarity Based Comparison class."""

from functools import lru_cache

import networkx as nx
import numpy as np
from sklearn.preprocessing import normalize

from hcga.feature_class import FeatureClass, InterpretabilityScore
from hcga.features.utils import ensure_connected, remove_selfloops

featureclass_name = "RolesimilarityBasedComparison"


@lru_cache(maxsize=None)
def rbc(graph):
    """Rbc computation.

    Features based on the role of a node in a directed network.

    Create the role-similarity based comparison (rbc) matrix for nodes in the network,
    then convert this to a graph and extract some features
    ref: https://arxiv.org/abs/1103.5582
    For some features we remove selfloops, since the diagonal of the rbc matrix
    consists of ones, and therefore all nodes will have a selfloop with weight one

    References
    ----------
    .. [1] Cooper, Kathryn, and Mauricio Barahona.
        "Role-based similarity in directed networks."
        arXiv preprint arXiv:1012.2726 (2010).


    """
    a = np.where(nx.adjacency_matrix(graph).toarray() > 0, 1, 0)
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


def number_of_edges(graph):
    """"""
    return rbc(graph).number_of_edges()


def number_of_edges_no_selfloops(graph):
    """"""
    return remove_selfloops(rbc(graph)).number_of_edges()


def connectance(graph):
    """"""
    return nx.density(rbc(graph))


def diameter(graph):
    """"""
    return nx.diameter(rbc(ensure_connected(graph)))


def radius(graph):
    """"""
    return nx.radius(rbc(ensure_connected(graph)))


def degree_assortativity_coeff(graph):
    """"""
    return nx.degree_assortativity_coefficient(rbc(graph))


def transitivity(graph):
    """"""
    return nx.transitivity(rbc(graph))


def is_connected(graph):
    """"""
    return nx.is_connected(rbc(graph)) * 1


def num_connected_components(graph):
    """"""
    return nx.number_connected_components(rbc(graph))


def largest_connected_component(graph):
    """"""
    return rbc(ensure_connected(graph)).number_of_nodes()


def global_efficiency(graph):
    """"""
    return nx.global_efficiency(rbc(graph))


def node_connectivity(graph):
    """"""
    return nx.node_connectivity(rbc(graph))


def edge_connectivity(graph):
    """"""
    return nx.edge_connectivity(rbc(graph))


class RolesimilarityBasedComparison(FeatureClass):
    """Role-similarity Based Comparison class."""

    modes = ["fast", "medium", "slow"]
    shortname = "RBC"
    name = "rbc"
    encoding = "networkx"

    def compute_features(self):
        # Basic stats
        self.add_feature(
            "number_of_edges",
            number_of_edges,
            "Number of edges in Jaccard similarity graph",
            InterpretabilityScore(5),
        )

        self.add_feature(
            "number_of_edges_no_selfloops",
            number_of_edges_no_selfloops,
            "Number of edges, not including selfloops, in Jaccard similarity graph",
            InterpretabilityScore(5),
        )

        self.add_feature(
            "connectance",
            connectance,
            "Connectance of Jaccard similarity graph",
            InterpretabilityScore(5),
        )

        self.add_feature(
            "diameter",
            diameter,
            "Diameter of Jaccard similarity graph",
            InterpretabilityScore(5),
        )

        self.add_feature(
            "radius",
            radius,
            "Radius of Jaccard similarity graph",
            InterpretabilityScore(5),
        )

        # Assortativity
        self.add_feature(
            "degree_assortativity_coeff",
            degree_assortativity_coeff,
            "Similarity of connections in Jaccard similarity graph with respect to the node degree",
            InterpretabilityScore(4),
        )

        # Clustering
        self.add_feature(
            "transitivity",
            transitivity,
            "Transitivity of the graph",
            InterpretabilityScore(4),
        )

        # Components
        self.add_feature(
            "is_connected",
            is_connected,
            "Whether the Jaccard similarity graph is connected or not",
            InterpretabilityScore(5),
        )

        self.add_feature(
            "num_connected_components",
            num_connected_components,
            "The number of connected components",
            InterpretabilityScore(5),
        )

        self.add_feature(
            "largest_connected_component",
            largest_connected_component,
            "The size of the largest connected component",
            InterpretabilityScore(4),
        )

        # Efficiency
        self.add_feature(
            "global_efficiency",
            global_efficiency,
            "The global efficiency",
            InterpretabilityScore(4),
        )

        # Node connectivity
        self.add_feature(
            "node_connectivity",
            node_connectivity,
            "Node connectivity",
            InterpretabilityScore(4),
        )

        self.add_feature(
            "edge_connectivity",
            edge_connectivity,
            "Edge connectivity",
            InterpretabilityScore(4),
        )
