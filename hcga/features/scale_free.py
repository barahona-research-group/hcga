"""Scale Free class."""
from functools import partial

import networkx as nx

from hcga.feature_class import FeatureClass, InterpretabilityScore

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
            partial(nx.s_metric, normalized=False),
            "Sum of the products deg(u)*deg(v) for every edge (u,v) in G",
            InterpretabilityScore(4),
        )
