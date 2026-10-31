"""Small worldness class."""

import networkx as nx

from hcga.feature_class import FeatureClass, InterpretabilityScore
from hcga.features.utils import ensure_connected

featureclass_name = "SmallWorldness"

SEED = 0


def omega(graph):
    """Small-world coefficient omega of the largest connected component (undirected).

    Returns None where the reference graphs cannot be built by edge rewiring: fewer than 4
    nodes, complete graphs (no swap is possible), and stars (every pair of edges shares the
    centre, so nx.omega would loop forever).
    """
    graph = nx.Graph(graph)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    graph = ensure_connected(graph)
    n_nodes, n_edges = graph.number_of_nodes(), graph.number_of_edges()
    if n_nodes < 4 or n_edges == n_nodes * (n_nodes - 1) // 2:
        return None
    if n_edges == n_nodes - 1 and max(d for _, d in graph.degree()) == n_nodes - 1:
        return None
    return nx.omega(graph, seed=SEED)


class SmallWorldness(FeatureClass):
    """Small worldness class.

    Fetures based on small-worldness, where the small-world coefficient of a graph G is:

    omega = Lr/L - C/Cl

    where C and L are respectively the average clustering coefficient and
    average shortest path length of G. Lr is the average shortest path length
    of an equivalent random graph and Cl is the average clustering coefficient
    of an equivalent lattice graph.

    Small world calculations using networkx:
        `Networkx_omega <https://networkx.github.io/documentation/latest/\
        _modules/networkx/algorithms/smallworld.html#omega>`_

    References
    ----------
    .. [1] Telesford, Joyce, Hayasaka, Burdette, and Laurienti (2011).
           "The Ubiquity of Small-World Networks".
           Brain Connectivity. 1 (0038): 367-75.  PMC 3604768. PMID 22432451.
           doi:10.1089/brain.2011.0038.

    """

    modes = ["slow"]
    shortname = "SW"
    name = "small_worldness"
    encoding = "networkx"

    def compute_features(self):
        # omega metric
        self.add_feature(
            "omega",
            omega,
            "The small world coefficient omega",
            InterpretabilityScore(4),
        )
