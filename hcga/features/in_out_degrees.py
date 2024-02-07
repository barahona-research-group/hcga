"""In Out degrees class."""

import networkx as nx

from hcga.feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "InOutDegrees"


def in_degree(graph):
    """in_degree"""
    if nx.is_directed(graph):
        return list(dict(graph.in_degree).values())
    return [0]


def out_degree(graph):
    """out_degree"""
    if nx.is_directed(graph):
        return list(dict(graph.out_degree).values())
    return [0]


def in_deg_n(graph):
    """in_deg_n"""
    if nx.is_directed(graph):
        return [
            i / d
            for i, d in zip(list(dict(graph.in_degree).values()), list(dict(graph.degree).values()))
        ]
    return [0]


def out_deg_n(graph):
    """out_deg_n"""
    if nx.is_directed(graph):
        return [
            o / d
            for o, d in zip(
                list(dict(graph.out_degree).values()), list(dict(graph.degree).values())
            )
        ]
    return [0]


def in_out_deg(graph):
    """in_out_deg"""
    if nx.is_directed(graph):
        return [
            i / o
            for i, o in zip(
                list(dict(graph.in_degree).values()),
                list(dict(graph.out_degree).values()),
            )
        ]
    return [0]


def in_degree_centrality(graph):
    """in_degree_centrality"""
    return list(nx.in_degree_centrality(graph).values())


def out_degree_centrality(graph):
    """out_degree_centrality"""
    return list(nx.out_degree_centrality(graph).values())


class InOutDegrees(FeatureClass):
    """In Out degrees class.

    Features based on the in and out degrees of directed networks.

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
