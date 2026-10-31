"""Chemical theory class."""

from functools import partial

import networkx as nx
import numpy as np
from scipy.special import logsumexp

from hcga.feature_class import FeatureClass, InterpretabilityScore
from hcga.features.utils import ensure_connected

featureclass_name = "ChemicalTheory"


def _undirected(graph):
    """Undirected view of a directed graph."""
    if graph.is_directed():
        return graph.to_undirected(as_view=True)
    return graph


def wiener_index(graph, weight=None):
    """Wiener index of the largest connected component (undirected view of a directed graph).

    Restricting to the largest connected component keeps the index finite on disconnected graphs.
    """
    return nx.wiener_index(ensure_connected(_undirected(graph)), weight=weight)


def log_estrada_index(graph):
    """Natural log of the Estrada index, log(sum_j exp(lambda_j)) over adjacency eigenvalues.

    Computed with logsumexp so that it stays finite when the Estrada index itself overflows.
    As in networkx, the adjacency matrix is binary (weights ignored).
    """
    adjacency = nx.to_numpy_array(_undirected(graph), weight=None)
    adjacency[adjacency != 0] = 1.0
    return float(logsumexp(np.linalg.eigvalsh(adjacency)))


def estrada_index(graph):
    """Estrada index, exp(log_estrada_index): inf rather than an error when it overflows."""
    with np.errstate(over="ignore"):
        return float(np.exp(log_estrada_index(graph)))


class ChemicalTheory(FeatureClass):
    """Chemical theory class.

    Here we implement the wiener and estrada indexes.

    The *Wiener index* of a graph is the sum of the shortest-path
    distances between each pair of reachable nodes [1]_. For pairs of nodes
    in undirected graphs, only one orientation of the pair is counted.
    Here it is computed on the largest connected component, so it is finite for disconnected
    graphs. The weighted version treats edge weights as distances.

    The Estrada Index is a topological index of folding or 3D “compactness” [2]_.
    It is also given as its logarithm, which does not overflow on large or dense graphs.

    Directed graphs are converted to undirected graphs for both indices.

    References
    ----------
    .. [1] Rouvray, Dennis H.
     "The rich legacy of half a century of the Wiener index.",
      Topology in Chemistry. Woodhead Publishing, 2002. 16-37.
    .. [2] E. Estrada,  Characterization of 3D molecular structure,
       Chem. Phys. Lett. 319, 713 (2000).

    """

    modes = ["fast", "medium", "slow"]
    shortname = "CT"
    name = "chemical_theory"
    encoding = "networkx"

    def compute_features(self):
        self.add_feature(
            "wiener index",
            wiener_index,
            "Sum of the lengths of the shortest paths between all pairs of vertices",
            InterpretabilityScore(4),
        )

        self.add_feature(
            "wiener index weighted",
            partial(wiener_index, weight="weight"),
            "Sum of the lengths of the shortest paths between all pairs of vertices",
            InterpretabilityScore(4),
        )

        self.add_feature(
            "estrada_index",
            estrada_index,
            "Topological index of protein folding or 3D compactness",
            InterpretabilityScore(4),
        )

        self.add_feature(
            "log_estrada_index",
            log_estrada_index,
            "Logarithm of the Estrada index (finite where the Estrada index overflows)",
            InterpretabilityScore(4),
        )
