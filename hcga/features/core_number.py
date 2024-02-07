"""Core number class."""

import networkx as nx
import numpy as np

from hcga.feature_class import FeatureClass, InterpretabilityScore
from hcga.features.utils import remove_selfloops

featureclass_name = "CoreNumber"


def core_number(graph):
    """core_number"""
    return list(np.asarray(list(nx.core_number(remove_selfloops(graph)).values())))


class CoreNumber(FeatureClass):
    """Core number class.

    Features based on a k-core analysis.

    A k-core is a maximal subgraph that contains nodes of degree k or more.
    The core number of a node is the largest value k of a k-core containing that node.

    Uses networkx, see 'https://networkx.org/documentation/stable/reference/algorithms/core.html`

    References
    ----------
    .. [1] An O(m) Algorithm for Cores Decomposition of Networks
       Vladimir Batagelj and Matjaz Zaversnik, 2003.
       https://arxiv.org/abs/cs.DS/0310049

    """

    modes = ["fast", "medium", "slow"]
    shortname = "CoN"
    name = "core_number"
    encoding = "networkx"

    def compute_features(self):
        self.add_feature(
            "core number",
            core_number,
            "The core number distribution",
            InterpretabilityScore(5),
            statistics="centrality",
        )
