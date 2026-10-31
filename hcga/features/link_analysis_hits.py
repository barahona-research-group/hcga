"""HITS hubs class."""

from functools import lru_cache

import networkx as nx

from hcga.feature_class import FeatureClass, InterpretabilityScore
from hcga.features.utils import ensure_connected

featureclass_name = "Hits"


@lru_cache(maxsize=1)
def eval_hits(graph):
    """Hubs and authorities (nx.hits_numpy was removed in networkx 3.0)."""
    return nx.hits(ensure_connected(graph), max_iter=1000)


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
