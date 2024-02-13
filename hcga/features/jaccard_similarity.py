"""Jaccard Similarity class."""

from functools import lru_cache

import networkx as nx
import numpy as np

from hcga.feature_class import FeatureClass, InterpretabilityScore
from hcga.features.utils import ensure_connected, remove_selfloops

featureclass_name = "JaccardSimilarity"


@lru_cache(maxsize=None)
def jaccard_similarity(graph):
    """Construct a graph from Jaccard similarity matrix

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

    For further information see `https://en.wikipedia.org/wiki/Jaccard_index'

    """
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


def number_of_edges(graph):
    """number_of_edges"""
    return jaccard_similarity(graph).number_of_edges()


def number_of_edges_no_selfloops(graph):
    """number_of_edges_no_selfloops"""
    return remove_selfloops(jaccard_similarity(graph)).number_of_edges()


def connectance(graph):
    """connectance"""
    return nx.density(jaccard_similarity(graph))


def diameter(graph):
    """diameter"""
    return nx.diameter(jaccard_similarity(ensure_connected(graph)))


def radius(graph):
    """radius"""
    return nx.radius(jaccard_similarity(ensure_connected(graph)))


def degree_assortativity_coeff(graph):
    """degree_assortativity_coeff"""
    return nx.degree_assortativity_coefficient(jaccard_similarity(graph))


def transitivity(graph):
    """transitivity"""
    return nx.transitivity(jaccard_similarity(graph))


def is_connected(graph):
    """is_connected"""
    return nx.is_connected(jaccard_similarity(graph)) * 1


def num_connected_components(graph):
    """num_connected_components"""
    return nx.number_connected_components(jaccard_similarity(graph))


def largest_connected_component(graph):
    """largest_connected_component"""
    return jaccard_similarity(ensure_connected(graph)).number_of_nodes()


def global_efficiency(graph):
    """global_efficiency"""
    return nx.global_efficiency(jaccard_similarity(graph))


class JaccardSimilarity(FeatureClass):
    """Jaccard Similarity class."""

    modes = ["fast", "medium", "slow"]
    shortname = "JS"
    name = "jaccard_similarity"
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
