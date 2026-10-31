"""Dominating sets class."""

import networkx as nx
from networkx.algorithms.approximation import min_edge_dominating_set

from hcga.feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "DominatingSets"


def min_weighted_dominating_set(graph):
    """Greedy approximation of a minimum dominating set (Vazirani, Algorithm 2.2).

    This is networkx's ``approximation.min_weighted_dominating_set`` as corrected in
    networkx 3.7 (networkx PR #8531): the cost of a candidate node is its weight divided by
    the number of still-uncovered nodes in its closed neighbourhood. Versions before 3.7
    divided by the number of neighbours not in the dominating set, which keeps selecting
    nodes whose neighbourhoods are already covered and can return more than twice as many
    nodes as needed. The algorithm is vendored so the value does not depend on the
    installed networkx version. (networkx is BSD-3 licensed.) Like networkx, it is defined
    for undirected graphs only; FeatureClass evaluates it on the undirected copy of a digraph.
    """
    if graph.is_directed():
        raise nx.NetworkXNotImplemented("not implemented for directed type")
    uncovered = set(graph)
    neighbourhoods = {v: {v} | set(graph[v]) for v in graph}
    dom_set = set()

    def _cost(item):
        node, neighbourhood = item
        n_new = len(neighbourhood & uncovered)
        if n_new == 0:
            return float("inf")
        return graph.nodes[node].get("weight", 1) / n_new

    while uncovered:
        node, neighbourhood = min(neighbourhoods.items(), key=_cost)
        dom_set.add(node)
        del neighbourhoods[node]
        uncovered -= neighbourhood
    return dom_set


def size_dominating_set(graph):
    """size_dominating_set"""
    return len(list(nx.dominating_set(graph)))


def size_min_dominating_set(graph):
    """size_min_dominating_set"""
    return len(min_weighted_dominating_set(graph))


def size_min_edge_dominating_set(graph):
    """size_min_edge_dominating_set

    networkx's approximation is a greedy maximal matching, so this equals
    MaximalMatching.maximal_matching.
    """
    if graph.is_directed():
        graph = graph.to_undirected(as_view=True)
    return len(min_edge_dominating_set(graph))


class DominatingSets(FeatureClass):
    """Dominating sets class.

    Features based on dominating sets. Where  *dominating set* for a graph
    with node set *V* is a subset *D* of
    *V* such that every node not in *D* is adjacent to at least one
    member of *D* [1]_.

    Uses networkx: `Networkx_dominating_set <https://networkx.github.io/documentation/\
        stable/reference/algorithms/dominating.html>`_

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Dominating_set

    .. [2] Abdol-Hossein Esfahanian. Connectivity Algorithms.
        http://www.cse.msu.edu/~cse835/Papers/Graph_connectivity_revised.pdf

    """

    modes = ["fast", "medium", "slow"]
    shortname = "DS"
    name = "dominating_sets"
    encoding = "networkx"

    def compute_features(self):
        self.add_feature(
            "size_dominating_set",
            size_dominating_set,
            "The number of nodes in the dominating set",
            InterpretabilityScore(3),
        )

        self.add_feature(
            "size_min_dominating_set",
            size_min_dominating_set,
            "The number of nodes in the minimum weighted dominating set",
            InterpretabilityScore(3),
        )

        self.add_feature(
            "size_min_edge_dominating_set",
            size_min_edge_dominating_set,
            "The number of edges in the minimum edge dominating set (2-approximation)",
            InterpretabilityScore(3),
        )

