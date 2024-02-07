"""Centralities class."""

import networkx as nx
import numpy as np
from networkx.algorithms import centrality

from hcga.feature_class import FeatureClass, InterpretabilityScore
from hcga.features.utils import ensure_connected

featureclass_name = "CentralitiesBasic"


def degree_centrality(graph):
    """degree_centrality"""
    return list(centrality.degree_centrality(graph).values())


def betweenness_centrality(graph):
    """betweenness_centrality"""
    return list(centrality.betweenness_centrality(graph).values())


def weighted_betweenness_centrality(graph):
    """weighted_betweenness_centrality"""
    return list(centrality.betweenness_centrality(graph, weight="weight").values())


def closeness_centrality(graph):
    """closeness_centrality"""
    return list(centrality.closeness_centrality(graph).values())


def edge_betweenness_centrality(graph):
    """edge_betweenness_centrality"""
    if graph.edges:
        return list(centrality.edge_betweenness_centrality(graph).values())
    return [np.nan]


def weighted_edge_betweenness_centrality(graph):
    """weighted_edge_betweenness_centrality"""
    if graph.edges:
        return list(centrality.edge_betweenness_centrality(graph, weight="weight").values())
    return [np.nan]


def harmonic_centrality(graph):
    """harmonic_centrality"""
    return list(centrality.harmonic_centrality(graph).values())


def subgraph_centrality(graph):
    """subgraph_centrality"""
    return list(centrality.subgraph_centrality(graph).values())


def second_order_centrality(graph):
    """second_order_centrality"""
    return list(centrality.second_order_centrality(ensure_connected(graph)).values())


def eigenvector_centrality(graph):
    """eigenvector_centrality"""
    return list(centrality.eigenvector_centrality_numpy(ensure_connected(graph)).values())


def weighted_eigenvector_centrality(graph):
    """weighted_eigenvector_centrality"""
    return list(
        centrality.eigenvector_centrality_numpy(ensure_connected(graph), weight="weight").values()
    )


def katz_centrality(graph):
    """katz_centrality"""
    return list(centrality.katz_centrality_numpy(ensure_connected(graph)).values())


def pagerank(graph):
    """pagerank"""
    return list(nx.pagerank(graph).values())


def weighted_pagerank(graph):
    """weighted_pagerank"""
    return list(nx.pagerank(graph, weight="weight").values())


class CentralitiesBasic(FeatureClass):
    """Centralities class.

    Uses networkx, see 'https://networkx.github.io/documentation/stable/reference/\
        algorithms/centrality.html`

    Here we implement:
    Degree Centrality
    Eigenvector Centrality [1]_ [2]_
    Closeness Centrality [3]_ [4]_
    Betweenness Centrality [5]_ [6]_ [7]_ [8]_
    Harmonic Centrality [9]_

    References
    ----------
    .. [1] Phillip Bonacich.
       "Power and Centrality: A Family of Measures."
       *American Journal of Sociology* 92(5):1170–1182, 1986
       <http://www.leonidzhukov.net/hse/2014/socialnetworks/papers/Bonacich-Centrality.pdf>
    .. [2] Mark E. J. Newman.
       *Networks: An Introduction.*
       Oxford University Press, USA, 2010, pp. 169.
    .. [3] Linton C. Freeman: Centrality in networks: I.
       Conceptual clarification. Social Networks 1:215-239, 1979.
       http://leonidzhukov.ru/hse/2013/socialnetworks/papers/freeman79-centrality.pdf
    .. [4] pg. 201 of Wasserman, S. and Faust, K.,
       Social Network Analysis: Methods and Applications, 1994,
       Cambridge University Press.
    .. [5] Ulrik Brandes:
       A Faster Algorithm for Betweenness Centrality.
       Journal of Mathematical Sociology 25(2):163-177, 2001.
       http://www.inf.uni-konstanz.de/algo/publications/b-fabc-01.pdf
    .. [6] Ulrik Brandes:
       On Variants of Shortest-Path Betweenness
       Centrality and their Generic Computation.
       Social Networks 30(2):136-145, 2008.
       http://www.inf.uni-konstanz.de/algo/publications/b-vspbc-08.pdf
    .. [7] Ulrik Brandes and Christian Pich:
       Centrality Estimation in Large Networks.
       International Journal of Bifurcation and Chaos 17(7):2303-2318, 2007.
       http://www.inf.uni-konstanz.de/algo/publications/bp-celn-06.pdf
    .. [8] Linton C. Freeman:
       A set of measures of centrality based on betweenness.
       Sociometry 40: 35–41, 1977
       http://moreno.ss.uci.edu/23.pdf
    .. [9] Boldi, Paolo, and Sebastiano Vigna. "Axioms for centrality."
           Internet Mathematics 10.3-4 (2014): 222-262.
    """

    modes = ["fast", "medium", "slow"]
    shortname = "CB"
    name = "centralities_basic"
    encoding = "networkx"

    def compute_features(self):
        # Degree centrality

        self.add_feature(
            "degree centrality",
            degree_centrality,
            "The degree centrality distribution",
            InterpretabilityScore(5),
            statistics="centrality",
        )

        # Betweenness Centrality
        self.add_feature(
            "betweenness centrality",
            betweenness_centrality,
            "Betweenness centrality of a node v is the sum of the fraction of \
            all-pairs shortest paths that pass through v",
            InterpretabilityScore(5),
            statistics="centrality",
        )

        # Betweenness Centrality Weighted

        self.add_feature(
            "betweenness centrality_weighted",
            weighted_betweenness_centrality,
            "Betweenness centrality of a node v is the sum of the fraction of \
            all-pairs shortest paths that pass through v",
            InterpretabilityScore(5),
            statistics="centrality",
        )

        # Closeness centrality
        self.add_feature(
            "closeness centrality",
            closeness_centrality,
            "Closeness is the reciprocal of the average shortest path distance",
            InterpretabilityScore(5),
            statistics="centrality",
        )

        # Edge betweenness centrality
        self.add_feature(
            "edge betweenness centrality",
            edge_betweenness_centrality,
            "Betweenness centrality of an edge e is the sum of the fraction of \
            all-pairs shortest paths that pass through e",
            InterpretabilityScore(4),
            statistics="centrality",
        )

        self.add_feature(
            "edge betweenness centrality weighted",
            weighted_edge_betweenness_centrality,
            "Betweenness centrality of an edge e is the sum of the fraction of \
            all-pairs shortest paths that pass through e",
            InterpretabilityScore(4),
            statistics="centrality",
        )

        # Harmonic centrality
        self.add_feature(
            "harmonic centrality",
            harmonic_centrality,
            "Harmonic centrality of a node u is the sum of the reciprocal \
            of the shortest path distances from all other nodes to u",
            InterpretabilityScore(4),
            statistics="centrality",
        )

        # Subgraph centrality
        self.add_feature(
            "subgraph centrality",
            subgraph_centrality,
            "The subgraph centrality for a node is the sum of weighted closed walks \
            of all lengths starting and ending at that node.",
            InterpretabilityScore(3),
            statistics="centrality",
        )

        # Second order centrality
        self.add_feature(
            "second order centrality",
            second_order_centrality,
            "The second order centrality of a given node is the standard deviation \
            of the return times to that node of a perpetual random walk on G",
            InterpretabilityScore(4),
            statistics="centrality",
        )

        # Eigenvector centrality
        self.add_feature(
            "eigenvector centrality",
            eigenvector_centrality,
            "Eigenvector centrality computes the centrality for a node based \
            on the centrality of its neighbors",
            InterpretabilityScore(4),
            statistics="centrality",
        )

        # weighted eigenvector centrality
        self.add_feature(
            "eigenvector centrality weighted",
            eigenvector_centrality,
            "Eigenvector centrality computes the centrality for a node based \
            on the centrality of its neighbors",
            InterpretabilityScore(4),
            statistics="centrality",
        )

        # Katz centrality
        self.add_feature(
            "katz centrality",
            katz_centrality,
            "Generalisation of eigenvector centrality - Katz centrality computes the \
            centrality for a node based on the centrality of its neighbors",
            InterpretabilityScore(4),
            statistics="centrality",
        )

        # Page Rank
        self.add_feature(
            "pagerank",
            pagerank,
            "The pagerank computes a ranking of the nodes in the graph based on \
            the structure of the incoming links. ",
            InterpretabilityScore(4),
            statistics="centrality",
        )

        self.add_feature(
            "pagerank weighted",
            weighted_pagerank,
            "The pagerank computes a ranking of the nodes in the graph based on \
            the structure of the incoming links. ",
            InterpretabilityScore(4),
            statistics="centrality",
        )
