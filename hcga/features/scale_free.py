"""Scale Free class."""
import networkx as nx

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "ScaleFree"


class ScaleFree(FeatureClass):
    """Scale Free class.

    Scale free calculations using networkx:
        `Networkx_scale free <https://networkx.github.io/documentation/stable/\
        _modules/networkx/algorithms/smetric.html#s_metric>`_
    """

    modes = ["fast", "medium", "slow"]
    shortname = "SF"
    name = "scale_free"
    encoding = "networkx"

    def compute_features(self):

        # s metric
        self.add_feature(
            "s_metric",
            lambda graph: nx.s_metric(graph, normalized=False),
            "The s-metric is defined as the sum of the products deg(u)*deg(v) for every edge (u,v) in G",
            InterpretabilityScore(4),
        )
