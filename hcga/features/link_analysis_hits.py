"""HITS hubs class."""
import networkx as nx

from ..feature_class import FeatureClass, InterpretabilityScore
from .utils import ensure_connected

featureclass_name = "Hits"


@ensure_connected
def hits(graph):
    h, _ = nx.hits_numpy(graph)
    return list(h.values())


class Hits(FeatureClass):
    """HITS hubs class.

    Hits calculations using networkx:
            `Networkx_hits <https://networkx.github.io/documentation/stable/reference/\
                algorithms/generated/networkx.algorithms.link_analysis.hits_alg.hits.html\
                #networkx.algorithms.link_analysis.hits_alg.hits>`_
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
