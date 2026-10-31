"""HITS hubs class."""

from functools import lru_cache

import networkx as nx
import numpy as np

from hcga.feature_class import FeatureClass, InterpretabilityScore
from hcga.features.utils import ensure_connected

featureclass_name = "Hits"


def hits_power_iteration(graph, max_iter=1000, tol=1.0e-8):
    """HITS hubs and authorities by power iteration from a uniform start.

    This is networkx 3.7's default ``nx.hits`` (networkx PR #8620, BSD-3 licensed). Earlier
    networkx 3.x versions used ``scipy.sparse.linalg.eigs`` instead, which on graphs with a
    degenerate leading eigenvalue (every bipartite graph) returns an arbitrary vector from the
    eigenspace, including negative "hub" scores; power iteration from the uniform vector
    gives the usual nonnegative solution. Vendored so the value does not depend on the
    installed networkx version.
    """
    nodelist = list(graph)
    n_nodes = len(nodelist)
    if n_nodes == 0:
        return {}, {}
    adjacency = nx.adjacency_matrix(graph, nodelist=nodelist, dtype=float)
    hubs = np.full(n_nodes, 1.0 / n_nodes)
    for _ in range(max_iter):
        previous = hubs
        authorities = hubs @ adjacency
        hubs = adjacency @ authorities
        hubs = hubs / hubs.max()
        if np.abs(hubs - previous).sum() < tol:
            break
    else:
        raise nx.PowerIterationFailedConvergence(max_iter)
    hubs = hubs / hubs.sum()
    authorities = authorities / authorities.sum()
    return (
        dict(zip(nodelist, map(float, hubs))),
        dict(zip(nodelist, map(float, authorities))),
    )


@lru_cache(maxsize=1)
def eval_hits(graph):
    """Hubs and authorities (nx.hits_numpy was removed in networkx 3.0)."""
    return hits_power_iteration(ensure_connected(graph), max_iter=1000)


def hits(graph):
    """hits hubs"""
    return list(eval_hits(graph)[0].values())


def hits_authorities(graph):
    """hits authorities"""
    return list(eval_hits(graph)[1].values())


class Hits(FeatureClass):
    """HITS hubs class.

    Returns features based on the HITS hubs and authorities.
    The HITS algorithm computes two numbers for a node.
    Authorities estimates the node value based on the incoming links.
    Hubs estimates the node value based on outgoing links.

    Hits calculations using networkx:
            `Networkx_hits <https://networkx.github.io/documentation/stable/reference/\
                algorithms/generated/networkx.algorithms.link_analysis.hits_alg.hits.html\
                #networkx.algorithms.link_analysis.hits_alg.hits>`_

    References
    ----------
    .. [1] A. Langville and C. Meyer,
       "A survey of eigenvector methods of web information retrieval."
       http://citeseer.ist.psu.edu/713792.html
    .. [2] Jon Kleinberg,
       Authoritative sources in a hyperlinked environment
       Journal of the ACM 46 (5): 604-32, 1999.
       doi:10.1145/324133.324140.
       http://www.cs.cornell.edu/home/kleinber/auth.pdf.

    """

    modes = ["medium", "slow"]
    shortname = "LAH"
    name = "Hits"
    encoding = "networkx"

    def compute_features(self):
        self.add_feature(
            "Hits",
            hits,
            "Hits algorithm",
            InterpretabilityScore(3),
            statistics="centrality",
        )

        self.add_feature(
            "hits_authorities",
            hits_authorities,
            "Hits algorithm authorities",
            InterpretabilityScore(3),
            statistics="centrality",
        )
