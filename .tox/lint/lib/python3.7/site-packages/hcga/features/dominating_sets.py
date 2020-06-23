"""Dominating sets class."""
import networkx as nx

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "DominatingSets"


class DominatingSets(FeatureClass):
    """Dominating sets class.

    Uses networkx: `Networkx_dominating_set <https://networkx.github.io/documentation/\
        stable/reference/algorithms/dominating.html>`_
    """

    modes = ["fast", "medium", "slow"]
    shortname = "DS"
    name = "dominating_sets"
    encoding = "networkx"

    def compute_features(self):

        self.add_feature(
            "size_dominating_set",
            lambda graph: len(list(nx.dominating_set(graph))),
            "The number of nodes in the dominating set",
            InterpretabilityScore(3),
        )

        self.add_feature(
            "size_min_dominating_set",
            lambda graph: len(list(min_weighted_dominating_set(graph))),
            "The number of nodes in the minimum weighted dominating set",
            InterpretabilityScore(3),
        )

        self.add_feature(
            "size_min_edge_dominating_set",
            lambda graph: len(list(min_edge_dominating_set(graph))),
            "The number of nodes in the minimum edge dominating set",
            InterpretabilityScore(3),
        )


def min_weighted_dominating_set(G, weight=None):
    """Taken from netoworkx."""
    # The unique dominating set for the null graph is the empty set.
    if len(G) == 0:  # pylint: disable=len-as-condition
        return set()

    # This is the dominating set that will eventually be returned.
    dom_set = set()

    def _cost(node_and_neighborhood):
        """Returns the cost-effectiveness of greedily choosing the given node.

        `node_and_neighborhood` is a two-tuple comprising a node and its
        closed neighborhood.
        """
        v, neighborhood = node_and_neighborhood
        return G.nodes[v].get(weight, 1) / len(neighborhood - dom_set)

    # This is a set of all vertices not already covered by the
    # dominating set.
    vertices = set(G)
    # This is a dictionary mapping each node to the closed neighborhood
    # of that node.
    neighborhoods = {v: {v} | set(G[v]) for v in G}

    # Continue until all vertices are adjacent to some node in the
    # dominating set.
    while vertices:
        # Find the most cost-effective node to add, along with its
        # closed neighborhood.
        dom_node, min_set = min(neighborhoods.items(), key=_cost)
        # Add the node to the dominating set and reduce the remaining
        # set of nodes to cover.
        dom_set.add(dom_node)
        del neighborhoods[dom_node]
        vertices -= min_set

    return dom_set


def min_edge_dominating_set(G):
    """Taken from networkx."""
    if not G:
        raise ValueError("Expected non-empty NetworkX graph!")
    return maximal_matching(G)


def maximal_matching(G):
    """Taken from networkx."""
    matching = set()
    nodes = set()
    for u, v in G.edges():
        # If the edge isn't covered, add it to the matching
        # then remove neighborhood of u and v from consideration.
        if u not in nodes and v not in nodes and u != v:
            matching.add((u, v))
            nodes.add(u)
            nodes.add(v)
    return matching
