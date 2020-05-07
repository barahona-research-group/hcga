"""Dominating sets class."""
import networkx as nx

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "DominatingSets"


class DominatingSets(FeatureClass):
    """Dominating sets class.

    A *dominating set* for a graph with node set *V* is a subset *D* of
    *V* such that every node not in *D* is adjacent to at least one
    member of *D* [1]_.

    Parameters
    ----------


    Notes
    -----
    Core number calculations using networkx:
        `Networkx_dominating_set <https://networkx.github.io/documentation/\
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
        """Compute the dominating sets of the network.

        Computed statistics
        -----
        Put here the list of things that are computed, with corresponding names
        """

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
    r"""Returns a dominating set that approximates the minimum weight node
    dominating set.

    Parameters
    ----------
    G : NetworkX graph
        Undirected graph.

    weight : string
        The node attribute storing the weight of an node. If provided,
        the node attribute with this key must be a number for each
        node. If not provided, each node is assumed to have weight one.

    Returns
    -------
    min_weight_dominating_set : set
        A set of nodes, the sum of whose weights is no more than `(\log
        w(V)) w(V^*)`, where `w(V)` denotes the sum of the weights of
        each node in the graph and `w(V^*)` denotes the sum of the
        weights of each node in the minimum weight dominating set.

    Notes
    -----
    This algorithm computes an approximate minimum weighted dominating
    set for the graph `G`. The returned solution has weight `(\log
    w(V)) w(V^*)`, where `w(V)` denotes the sum of the weights of each
    node in the graph and `w(V^*)` denotes the sum of the weights of
    each node in the minimum weight dominating set for the graph.

    This implementation of the algorithm runs in $O(m)$ time, where $m$
    is the number of edges in the graph.

    References
    ----------
    .. [1] Vazirani, Vijay V.
           *Approximation Algorithms*.
           Springer Science & Business Media, 2001.

    """
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
    r"""Returns minimum cardinality edge dominating set.

    Parameters
    ----------
    G : NetworkX graph
      Undirected graph

    Returns
    -------
    min_edge_dominating_set : set
      Returns a set of dominating edges whose size is no more than 2 * OPT.

    Notes
    -----
    The algorithm computes an approximate solution to the edge dominating set
    problem. The result is no more than 2 * OPT in terms of size of the set.
    Runtime of the algorithm is $O(|E|)$.
    """
    if not G:
        raise ValueError("Expected non-empty NetworkX graph!")
    return maximal_matching(G)


def maximal_matching(G):
    r"""Find a maximal matching in the graph.

    A matching is a subset of edges in which no node occurs more than once.
    A maximal matching cannot add more edges and still be a matching.

    Parameters
    ----------
    G : NetworkX graph
        Undirected graph

    Returns
    -------
    matching : set
        A maximal matching of the graph.

    Notes
    -----
    The algorithm greedily selects a maximal matching M of the graph G
    (i.e. no superset of M exists). It runs in $O(|E|)$ time.
    """
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
