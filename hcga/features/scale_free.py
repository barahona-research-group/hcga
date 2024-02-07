"""Scale Free class."""

from functools import partial

import networkx as nx

from hcga.feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "ScaleFree"


class ScaleFree(FeatureClass):
    """Scale Free class.

    Features based on the scalefreeness of a graph. It uses the s-metric.
    The s-metric is defined as the sum of the products deg(u)*deg(v)
    for every edge (u,v) in G. If norm is provided construct the
    s-max graph and compute it's s_metric, and return the normalized
    s value

    Scale free calculations using networkx:
        `Networkx_scale free <https://networkx.github.io/documentation/stable/\
        _modules/networkx/algorithms/smetric.html#s_metric>`_

    References
    ----------
    .. [1] Lun Li, David Alderson, John C. Doyle, and Walter Willinger,
           Towards a Theory of Scale-Free Graphs:
           Definition, Properties, and  Implications (Extended Version), 2005.
           https://arxiv.org/abs/cond-mat/0501169

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
