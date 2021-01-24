"""Components class."""
from functools import lru_cache

import networkx as nx

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "Components"


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
            lambda graph: nx.is_connected(graph) * 1,
            "Whether the graph is connected or not",
            InterpretabilityScore(5),
        )

        self.add_feature(
            "num_connected_components",
            lambda graph: len(list(nx.connected_components(graph))),
            "The number of connected components",
            InterpretabilityScore(5),
        )

        @lru_cache(maxsize=None)
        def eval_connectedcomponents(graph):
            """this evaluates the main function and cach it for speed up."""
            return list(nx.connected_components(graph))

        self.add_feature(
            "largest_connected_component",
            lambda graph: len(eval_connectedcomponents(graph)[0]),
            "The size of the largest connected component",
            InterpretabilityScore(4),
        )

        def ratio_largest(graph):
            if len(eval_connectedcomponents(graph)) == 1:
                return 0
            return len(eval_connectedcomponents(graph)[0]) / len(eval_connectedcomponents(graph)[1])

        self.add_feature(
            "ratio_largest_connected_components",
            ratio_largest,
            "The size ratio of the two largest connected components",
            InterpretabilityScore(4),
        )

        def ratio_min_max(graph):
            if len(eval_connectedcomponents(graph)) == 1:
                return 0
            return len(eval_connectedcomponents(graph)[0]) / len(
                eval_connectedcomponents(graph)[-1]
            )

        self.add_feature(
            "ratio_maxmin_connected_components",
            ratio_min_max,
            "The size ratio of the max and min largest connected components",
            InterpretabilityScore(4),
        )

        self.add_feature(
            "number_strongly_connected_components",
            lambda graph: nx.number_strongly_connected_components(graph),
            "A strongly connected component is a set of nodes in a directed graph such \
            that each node in the set is reachable from any other node in that set",
            InterpretabilityScore(3),
        )

        self.add_feature(
            "strongly_connected_component_sizes",
            lambda graph: [len(i) for i in nx.strongly_connected_components(graph)],
            "the distribution of strongly connected component sizes",
            InterpretabilityScore(3),
            statistics="centrality",
        )

        self.add_feature(
            "condensation_nodes",
            lambda graph: nx.condensation(graph).number_of_nodes(),
            "number of nodes in the condensation of the graph",
            InterpretabilityScore(3),
        )

        self.add_feature(
            "condensation_edges",
            lambda graph: nx.condensation(graph).number_of_edges(),
            "number of edges in the condensation of the graph",
            InterpretabilityScore(3),
        )

        self.add_feature(
            "number_weakly_connected_components",
            lambda graph: nx.number_weakly_connected_components(graph),
            "A weakly connected component is a set of nodes in a directed graph such that \
            there exists as edge between each node and at least one other node in the set",
            InterpretabilityScore(3),
        )

        self.add_feature(
            "weakly_connected_component_sizes",
            lambda graph: [len(i) for i in nx.weakly_connected_components(graph)],
            "the distribution of weakly connected component sizes",
            InterpretabilityScore(3),
            statistics="centrality",
        )

        self.add_feature(
            "number_attracting_components",
            lambda graph: nx.number_attracting_components(graph),
            "An attracting component is a set of nodes in a directed graph such that that \
            once in that set, all other nodes outside that set are not reachable",
            InterpretabilityScore(3),
        )

        self.add_feature(
            "attracting_component_sizes",
            lambda graph: [len(i) for i in nx.attracting_components(graph)],
            "the distribution of attracting component sizes",
            InterpretabilityScore(3),
            statistics="centrality",
        )

        self.add_feature(
            "number basal_components",
            lambda graph: nx.number_attracting_components(nx.reverse(graph)),
            "An basal component is a set of nodes in a directed graph such that there are no \
            edges pointing into that set",
            InterpretabilityScore(3),
        )

        self.add_feature(
            "basal_component_sizes",
            lambda graph: [len(i) for i in nx.attracting_components(nx.reverse(graph))],
            "the distribution of basal component sizes",
            InterpretabilityScore(3),
            statistics="centrality",
        )
