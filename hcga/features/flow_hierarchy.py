"""Flow hierarchy class."""
from functools import partial

import networkx as nx

from hcga.feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "FlowHierarchy"


class FlowHierarchy(FeatureClass):
    """Flow hierarchy class.

    Features based on flow hierarchy. Flow hierarchy is defined as the fraction of edges not
    participating in cycles in a directed graph [1]_.

    Uses networkx, see 'https://networkx.org/documentation/stable/reference/algorithms/\
        hierarchy.html`


    References
    ----------
    .. [1] Luo, J.; Magee, C.L. (2011),
       Detecting evolving patterns of self-organizing networks by flow
       hierarchy measurement, Complexity, Volume 16 Issue 6 53-61.
       DOI: 10.1002/cplx.20368
       http://web.mit.edu/~cmagee/www/documents/28-DetectingEvolvingPatterns_FlowHierarchy.pdf

    """

    modes = ["fast", "medium", "slow"]
    shortname = "FH"
    name = "flow_hierarchy"
    encoding = "networkx"

    def compute_features(self):

        # graph clique number
        self.add_feature(
            "flow_hierarchy",
            nx.flow_hierarchy,
            "fraction of edges not participating in cycles",
            InterpretabilityScore(3),
        )

        self.add_feature(
            "flow_hierarchy_weighted",
            partial(nx.flow_hierarchy, weight="weight"),
            "fraction of edges not participating in cycles",
            InterpretabilityScore(3),
        )
