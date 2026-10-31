"""In Out degrees class."""

import networkx as nx
import numpy as np

from hcga.feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "InOutDegrees"


def _ratio(numerators, denominators):
    """Per-node ratios, over the nodes where the denominator is non-zero.

    Nodes where the ratio is undefined (zero denominator) are left out of the distribution,
    and None is returned if it is undefined for every node.
    """
    numerators = np.array(list(numerators), dtype=float)
    denominators = np.array(list(denominators), dtype=float)
    defined = denominators > 0
    if not defined.any():
        return None
    return list(numerators[defined] / denominators[defined])


def in_degree(graph):
    """in_degree (None for undirected graphs)"""
    if nx.is_directed(graph):
        return list(dict(graph.in_degree).values())
    return None


def out_degree(graph):
    """out_degree (None for undirected graphs)"""
    if nx.is_directed(graph):
        return list(dict(graph.out_degree).values())
    return None


def in_deg_n(graph):
    """in_deg_n: in/total degree, over non-isolated nodes (None for undirected graphs)"""
    if nx.is_directed(graph):
        return _ratio(dict(graph.in_degree).values(), dict(graph.degree).values())
    return None


def out_deg_n(graph):
    """out_deg_n: out/total degree, over non-isolated nodes (None for undirected graphs)"""
    if nx.is_directed(graph):
        return _ratio(dict(graph.out_degree).values(), dict(graph.degree).values())
    return None


def in_out_deg(graph):
    """in_out_deg: in/out degree, over nodes with out-degree > 0 (None for undirected graphs)"""
    if nx.is_directed(graph):
        return _ratio(dict(graph.in_degree).values(), dict(graph.out_degree).values())
    return None


def in_degree_centrality(graph):
    """in_degree_centrality (None for undirected graphs)"""
    if nx.is_directed(graph):
        return list(nx.in_degree_centrality(graph).values())
    return None


def out_degree_centrality(graph):
    """out_degree_centrality (None for undirected graphs)"""
    if nx.is_directed(graph):
        return list(nx.out_degree_centrality(graph).values())
    return None


class InOutDegrees(FeatureClass):
    """In Out degrees class.

    Features based on the in and out degrees of directed networks.
    No features are computed for undirected graphs. The degree ratios are taken over the nodes
    where they are defined (non-zero denominator).

    """

    modes = ["fast", "medium", "slow"]
    shortname = "IOD"
    name = "in_out_degrees"
    encoding = "networkx"

    def compute_features(self):
        self.add_feature(
            "in_degree",
            in_degree,
            "The distribution of in degrees of each node",
            InterpretabilityScore(3),
            statistics="centrality",
        )

        self.add_feature(
            "in_degree_normed",
            in_deg_n,
            "The distribution of the ratio of in and total degrees of each node",
            InterpretabilityScore(3),
            statistics="centrality",
        )

        self.add_feature(
            "out_degree",
            out_degree,
            "The distribution of out degrees of each node",
            InterpretabilityScore(3),
            statistics="centrality",
        )

        self.add_feature(
            "out_degree_normed",
            out_deg_n,
            "The distribution of the ratio of out and total degrees of each node",
            InterpretabilityScore(3),
            statistics="centrality",
        )
        self.add_feature(
            "in_out_degree",
            in_out_deg,
            "The distribution of the ratio of in and out degrees of each node",
            InterpretabilityScore(3),
            statistics="centrality",
        )

        self.add_feature(
            "in_degree_centrality",
            in_degree_centrality,
            "The distribution of in degree centralities",
            InterpretabilityScore(3),
            statistics="centrality",
        )

        self.add_feature(
            "out_degree_centrality",
            out_degree_centrality,
            "The distribution of out degree centralities",
            InterpretabilityScore(3),
            statistics="centrality",
        )
