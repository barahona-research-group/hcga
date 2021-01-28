"Connectance class"
import networkx as nx

from hcga.feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "Connectance"


class Connectance(FeatureClass):
<<<<<<< HEAD
    """Connectance class.

    Features based on the connectivity of the graph.

    For now we compute only the density:
    .. math::

       d = \frac{2m}{n(n-1)},

    and for directed graphs is

    .. math::

       d = \frac{m}{n(n-1)},

    """
=======
    """Connectance class."""
>>>>>>> parent of 47d844a... rebuilt docs

    modes = ["fast", "medium", "slow"]
    shortname = "Cns"
    name = "connectance"
    encoding = "networkx"

    def compute_features(self):

        self.add_feature(
            "connectance",
            nx.density,
            "ratio of number of edges to maximum possible number of edges",
            InterpretabilityScore(3),
        )
