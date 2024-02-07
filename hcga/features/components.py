"""Components class."""

from functools import lru_cache

import networkx as nx

from hcga.feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "Components"


@lru_cache(maxsize=None)
def eval_connected_components(graph):
    """this evaluates the main function and cach it for speed up."""
    return list(nx.connected_components(graph))


def num_connected_components(graph):
    """num_connected_components"""
    return len(eval_connected_components(graph))


def largest_connected_component(graph):
    """largest_connected_components"""
    return len(eval_connected_components(graph)[0])


def ratio_largest(graph):
    """ratio_largest"""
    if len(eval_connected_components(graph)) == 1:
        return 0
    return len(eval_connected_components(graph)[0]) / len(eval_connected_components(graph)[1])


def ratio_min_max(graph):
    """ratio_min_max"""
    if len(eval_connected_components(graph)) == 1:
        return 0
    return len(eval_connected_components(graph)[0]) / len(eval_connected_components(graph)[-1])


def strongly_connected_component_sizes(graph):
    """strongly_connected_component_sizes"""
    return [len(i) for i in nx.strongly_connected_components(graph)]


def condensation_nodes(graph):
    """condensation_nodes"""
    return nx.condensation(graph).number_of_nodes()


def condensation_edges(graph):
    """condensation_edges"""
    return nx.condensation(graph).number_of_edges()


def weakly_connected_component_sizes(graph):
    """weakly_connected_component_sizes"""
    return [len(i) for i in nx.weakly_connected_components(graph)]


def attracting_component_sizes(graph):
    """attracting_component_sizes"""
    return [len(i) for i in nx.attracting_components(graph)]


def number_basal_components(graph):
    """number_basal_components"""
    if nx.is_directed(graph):
        return nx.number_attracting_components(nx.reverse(graph))
    return 0


def basal_component_sizes(graph):
    """basal_component_sizes"""
    if nx.is_directed(graph):
        return [len(i) for i in nx.attracting_components(nx.reverse(graph))]
    return [0]


class Components(FeatureClass):
    """Components class.

    Components calculations using network: `Networkx_components <https://networkx.github.io/\
        documentation/stable/reference/algorithms/component.html>`_
    """

    modes = ["fast", "medium", "slow"]
    shortname = "C"
    name = "components"
    encoding = "networkx"

    def compute_features(self):
        self.add_feature(
            "is_connected",
            nx.is_connected,
            "Whether the graph is connected or not",
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

        self.add_feature(
            "ratio_largest_connected_components",
            ratio_largest,
            "The size ratio of the two largest connected components",
            InterpretabilityScore(4),
        )

        self.add_feature(
            "ratio_maxmin_connected_components",
            ratio_min_max,
            "The size ratio of the max and min largest connected components",
            InterpretabilityScore(4),
        )

        self.add_feature(
            "number_strongly_connected_components",
            nx.number_strongly_connected_components,
            "A strongly connected component is a set of nodes in a directed graph such \
            that each node in the set is reachable from any other node in that set",
            InterpretabilityScore(3),
        )

        self.add_feature(
            "strongly_connected_component_sizes",
            strongly_connected_component_sizes,
            "the distribution of strongly connected component sizes",
            InterpretabilityScore(3),
            statistics="centrality",
        )

        self.add_feature(
            "condensation_nodes",
            condensation_nodes,
            "number of nodes in the condensation of the graph",
            InterpretabilityScore(3),
        )

        self.add_feature(
            "condensation_edges",
            condensation_edges,
            "number of edges in the condensation of the graph",
            InterpretabilityScore(3),
        )

        self.add_feature(
            "number_weakly_connected_components",
            nx.number_weakly_connected_components,
            "A weakly connected component is a set of nodes in a directed graph such that \
            there exists as edge between each node and at least one other node in the set",
            InterpretabilityScore(3),
        )

        self.add_feature(
            "weakly_connected_component_sizes",
            weakly_connected_component_sizes,
            "the distribution of weakly connected component sizes",
            InterpretabilityScore(3),
            statistics="centrality",
        )

        self.add_feature(
            "number_attracting_components",
            nx.number_attracting_components,
            "An attracting component is a set of nodes in a directed graph such that that \
            once in that set, all other nodes outside that set are not reachable",
            InterpretabilityScore(3),
        )

        self.add_feature(
            "attracting_component_sizes",
            attracting_component_sizes,
            "the distribution of attracting component sizes",
            InterpretabilityScore(3),
            statistics="centrality",
        )

        self.add_feature(
            "number_basal_components",
            number_basal_components,
            "An basal component is a set of nodes in a directed graph such that there are no \
            edges pointing into that set",
            InterpretabilityScore(3),
        )

        self.add_feature(
            "basal_component_sizes",
            basal_component_sizes,
            "the distribution of basal component sizes",
            InterpretabilityScore(3),
            statistics="centrality",
        )
