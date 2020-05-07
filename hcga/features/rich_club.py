"""Rich Club class."""
from functools import lru_cache

import networkx as nx
import numpy as np

from ..feature_class import FeatureClass, InterpretabilityScore
from .utils import remove_selfloops

featureclass_name = "RichClub"


class RichClub(FeatureClass):
    """Rich Club class."""

    modes = ["fast", "medium", "slow"]
    shortname = "RC"
    name = "rich_club"
    encoding = "networkx"

    def compute_features(self):
        """Compute the rich club measures of the network.

        Computed statistics
        -----
        Put here the list of things that are computed, with corresponding names
        """

        @lru_cache(maxsize=None)
        def eval_rich_club(graph):
            # extracting feature matrix
            return list(
                nx.rich_club_coefficient(
                    remove_selfloops(graph), normalized=False
                ).values()
            )

        # k = 1
        self.add_feature(
            "rich_club_k=1",
            lambda graph: eval_rich_club(graph)[0],
            "The rich-club coefficient is the ratio of the number of actual to the \
            number of potential edges for nodes with degree greater than k",
            InterpretabilityScore(4),
        )

        #
        self.add_feature(
            "rich_club_k=max",
            lambda graph: eval_rich_club(graph)[-1],
            "The rich-club coefficient is the ratio of the number of actual to the number \
            of potential edges for nodes with degree greater than k",
            InterpretabilityScore(4),
        )

        self.add_feature(
            "rich_club_maxminratio",
            lambda graph: np.min(eval_rich_club(graph)) / np.max(eval_rich_club(graph)),
            "The ratio of the smallest to largest rich club coefficients",
            InterpretabilityScore(4),
        )

        self.add_feature(
            "rich_club",
            lambda graph: eval_rich_club(graph),
            "The distribution of rich club coefficients",
            InterpretabilityScore(4),
            statistics="centrality",
        )
