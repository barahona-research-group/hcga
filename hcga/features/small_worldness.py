"""Small worldness class."""
import networkx as nx

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "SmallWorldness"


class SmallWorldness(FeatureClass):
    """Small worldness class.

    Small world calculations using networkx:
        `Networkx_omega <https://networkx.github.io/documentation/latest/\
        _modules/networkx/algorithms/smallworld.html#omega>`_
    """

    modes = ["slow"]
    shortname = "SW"
    name = "small_worldness"
    encoding = "networkx"

    def compute_features(self):

        # omega metric
        self.add_feature(
            "omega",
            nx.omega,
            "The small world coefficient omega",
            InterpretabilityScore(4),
        )
