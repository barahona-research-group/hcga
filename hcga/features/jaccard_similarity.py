"""Jaccard Similarity class."""

from functools import lru_cache

import networkx as nx
import numpy as np

from hcga.feature_class import FeatureClass, InterpretabilityScore
from hcga.features.utils import ensure_connected

featureclass_name = "JaccardSimilarity"


@lru_cache(maxsize=1)
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

    The diagonal of the Jaccard similarity (the similarity of a node with itself) is dropped,
    so the Jaccard similarity graph has no selfloops. For directed graphs, neighbours are
    successors.

    For further information see `https://en.wikipedia.org/wiki/Jaccard_index'

    """
    adjacency = nx.to_scipy_sparse_array(graph, weight=None, format="csr").astype(float)
    intersection = (adjacency @ adjacency.T).toarray()
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    union = degrees[:, None] + degrees[None, :] - intersection
    jsm = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    np.fill_diagonal(jsm, 0.0)
    return nx.Graph(jsm)


def number_of_edges(graph):
    """number_of_edges"""
    return jaccard_similarity(graph).number_of_edges()


def number_of_edges_no_selfloops(graph):
    """number_of_edges_no_selfloops (the Jaccard similarity graph has no selfloops)"""
    return jaccard_similarity(graph).number_of_edges()


def connectance(graph):
    """connectance"""
    return nx.density(jaccard_similarity(graph))


def diameter(graph):
    """diameter of the largest connected component of the Jaccard similarity graph"""
    return nx.diameter(ensure_connected(jaccard_similarity(graph)))


def radius(graph):
    """radius of the largest connected component of the Jaccard similarity graph"""
    return nx.radius(ensure_connected(jaccard_similarity(graph)))


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
    return len(max(nx.connected_components(jaccard_similarity(graph)), key=len))


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
            "Diameter of the largest connected component of Jaccard similarity graph",
            InterpretabilityScore(5),
        )
        self.add_feature(
            "radius",
            radius,
            "Radius of the largest connected component of Jaccard similarity graph",
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
            "The number of connected components of Jaccard similarity graph",
            InterpretabilityScore(5),
        )

        self.add_feature(
            "largest_connected_component",
            largest_connected_component,
            "The size of the largest connected component of Jaccard similarity graph",
            InterpretabilityScore(4),
        )

        # Efficiency
        self.add_feature(
            "global_efficiency",
            global_efficiency,
            "The global efficiency",
            InterpretabilityScore(4),
        )
