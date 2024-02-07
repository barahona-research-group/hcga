"""Basal nodes class."""

from functools import lru_cache

import networkx as nx

from hcga.feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "BasalNodes"

"""
Basal nodes are nodes which have in degree equal to zero. Attracting nodes are
nodes which have out degree equal to zero
"""


@lru_cache(maxsize=None)
def basal_nodes_func(graph):
    """"""
    in_degrees = dict(graph.in_degree)
    return [i for i in in_degrees if in_degrees[i] == 0]


def n_basal_nodes(graph):
    """n_basal_nodes."""
    if nx.is_directed(graph):
        return len(basal_nodes_func(graph))
    return 0


def basal_degrees(graph):
    """basal_degrees"""
    if nx.is_directed(graph):
        return [dict(graph.out_degree)[i] for i in basal_nodes_func(graph)]
    return [0]


def n_basal_edges(graph):
    """n_basal_edges"""
    if nx.is_directed(graph):
        return sum(dict(graph.out_degree)[i] for i in basal_nodes_func(graph))
    return 0


def exp_basal_edge(graph):
    """exp_basal_edge"""
    if nx.is_directed(graph):
        in_degs = list(dict(graph.in_degree).values())
        r = sum(dict(graph.out_degree)[i] for i in basal_nodes_func(graph)) / (
            graph.number_of_edges()
        )
        return [i * r for i in in_degs]
    return [0]


@lru_cache(maxsize=None)
def attracting_nodes_func(graph):
    """"""
    out_degrees = dict(graph.out_degree)
    return [i for i in out_degrees if out_degrees[i] == 0]


def n_attracting_nodes(graph):
    """n_attracting_nodes"""
    if nx.is_directed(graph):
        return len(attracting_nodes_func(graph))
    return 0


def attracting_degrees(graph):
    """attracting_degrees"""
    if nx.is_directed(graph):
        return [dict(graph.in_degree)[i] for i in attracting_nodes_func(graph)]
    return [0]


def n_attracting_edges(graph):
    """n_attracting_edges"""
    if nx.is_directed(graph):
        return sum(dict(graph.in_degree)[i] for i in attracting_nodes_func(graph))
    return 0


def exp_attracting_edge(graph):
    """exp_attracting_edge"""
    if nx.is_directed(graph):
        out_degs = list(dict(graph.out_degree).values())
        r = sum(dict(graph.in_degree)[i] for i in attracting_nodes_func(graph)) / (
            graph.number_of_edges()
        )
        return [i * r for i in out_degs]
    return [0]


class BasalNodes(FeatureClass):
    """Basal nodes class.

    Basal nodes are nodes which have in degree equal to zero. Attracting nodes are
    nodes which have out degree equal to zero

    References
    ----------
    .. [1]Johnson, Samuel, and Nick S. Jones. "Looplessness in networks is linked to trophic\
        coherence.",
     Proceedings of the National Academy of Sciences 114.22 (2017): 5618-5623.
    """

    modes = ["fast", "medium", "slow"]
    shortname = "BN"
    name = "basal_nodes"
    encoding = "networkx"

    def compute_features(self):
        self.add_feature(
            "n_basal_nodes",
            n_basal_nodes,
            "The number of basal nodes",
            InterpretabilityScore(3),
        )

        self.add_feature(
            "basal_degrees",
            basal_degrees,
            "The distribution of degrees of basal nodes",
            InterpretabilityScore(3),
            statistics="centrality",
        )

        self.add_feature(
            "n_basal_edges",
            n_basal_edges,
            "The total number of edges connected to basal nodes",
            InterpretabilityScore(3),
        )

        self.add_feature(
            "exp_basal_edge",
            exp_basal_edge,
            "The distribution of the expected number of in-edges of each node with basal nodes",
            InterpretabilityScore(3),
            statistics="centrality",
        )

        self.add_feature(
            "n_attracting_nodes",
            n_attracting_nodes,
            "The number of basal nodes",
            InterpretabilityScore(3),
        )

        self.add_feature(
            "attracting_degrees",
            attracting_degrees,
            "The distribution of degrees of attracting nodes",
            InterpretabilityScore(3),
            statistics="centrality",
        )

        self.add_feature(
            "n_attracting_edges",
            n_attracting_edges,
            "The total number of edges connected to attracting nodes",
            InterpretabilityScore(3),
        )
        self.add_feature(
            "exp_attracting_edge",
            exp_attracting_edge,
            "The distribution of the expected number of out-edges \
            of each node with attracting nodes",
            InterpretabilityScore(3),
            statistics="centrality",
        )
