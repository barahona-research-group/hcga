"""Independent sets class."""
import networkx as nx

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "IndependentSets"


class IndependentSets(FeatureClass):
    """Independent sets class.

    Parameters
    ----------



    Notes
    -----


    References
    ----------
    """

    modes = ["fast", "medium", "slow"]
    shortname = "IS"
    name = "independent_sets"
    encoding = "networkx"

    def compute_features(self):
        """Compute the independent sets of the network.

        Computed statistics
        -----
        Put here the list of things that are computed, with corresponding names
        """

        self.add_feature(
            "size_max_indep_set",
            lambda graph: len(nx.maximal_independent_set(graph)),
            "The number of nodes in the maximal independent set",
            InterpretabilityScore(3),
        )
