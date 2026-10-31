"""Vitality class."""

import networkx as nx
import numpy as np
from scipy.sparse.csgraph import shortest_path

from hcga.feature_class import FeatureClass, InterpretabilityScore
from hcga.features.utils import ensure_connected

featureclass_name = "Vitality"


def _wiener_index(adjacency, directed):
    """Unweighted Wiener index (inf if some pair is not connected)."""
    distances = shortest_path(adjacency, method="D", unweighted=True, directed=directed)
    if directed:
        return distances[~np.eye(len(distances), dtype=bool)].sum()
    return distances[np.triu_indices_from(distances, 1)].sum()


def closeness_vitality(graph):
    """Same values as nx.closeness_vitality(graph), using scipy shortest paths."""
    directed = graph.is_directed()
    adjacency = nx.to_scipy_sparse_array(graph, weight=None, format="csr")
    wiener = _wiener_index(adjacency, directed)
    vitalities = []
    for i in range(len(graph)):
        keep = np.r_[0:i, i + 1 : len(graph)]
        with np.errstate(invalid="ignore"):  # inf - inf on digraphs that are not strongly connected
            vitalities.append(wiener - _wiener_index(adjacency[keep][:, keep], directed))
    return np.array(vitalities)


def vitality(graph):
    """Closeness vitality on the largest connected component.

    Removing an articulation point disconnects the graph and gives a vitality of -inf, so
    non-finite values are dropped (articulation points are counted in n_articulation_points).
    """
    vitalities = closeness_vitality(ensure_connected(graph))
    vitalities = vitalities[np.isfinite(vitalities)]
    if len(vitalities) == 0:
        return None
    return list(vitalities)


def n_articulation_points(graph):
    """Number of articulation points of the largest connected component (undirected)."""
    graph = ensure_connected(graph)
    if graph.is_directed():
        graph = graph.to_undirected()
    return len(list(nx.articulation_points(graph)))


class Vitality(FeatureClass):
    """Vitality measures class.

    Features based on the closeness vitality.
    The *closeness vitality* of a node, defined in Section 3.6.2 of [1],
    is the change in the sum of distances between all node pairs when
    excluding that node.

    References
    ----------
    .. [1] Ulrik Brandes, Thomas Erlebach (eds.).
           *Network Analysis: Methodological Foundations*.
           Springer, 2005.
           <http://books.google.com/books?id=TTNhSm7HYrIC>
    """

    modes = ["slow"]
    shortname = "V"
    name = "vitality"
    encoding = "networkx"

    def compute_features(self):
        # distribution of vitality
        self.add_feature(
            "vitality",
            vitality,
            "The closeness vitality of a node is the change in the sum of distances between \
            all node pairs when excluding that node",
            InterpretabilityScore(3),
            statistics="centrality",
        )

        self.add_feature(
            "n_articulation_points",
            n_articulation_points,
            "The number of articulation points (nodes whose removal disconnects the largest \
            connected component, and whose vitality is therefore infinite)",
            InterpretabilityScore(3),
        )
