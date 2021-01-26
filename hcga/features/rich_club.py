"""Rich Club class."""
from functools import lru_cache

import networkx as nx
import numpy as np

from ..feature_class import FeatureClass, InterpretabilityScore
from .utils import remove_selfloops

featureclass_name = "RichClub"


class RichClub(FeatureClass):
    """Rich Club class.

    Features based on the Rich Club of a graph.

    For each degree *k*, the *rich-club coefficient* is the ratio of the
    number of actual to the number of potential edges for nodes with
    degree greater than *k*:

    Rich club calculations using networkx:
            `Rich club <https://networkx.org/documentation/stable/reference/algorithms/rich_club.html>`_

    References
    ----------
    .. [1] Julian J. McAuley, Luciano da Fontoura Costa,
       and Tibério S. Caetano,
       "The rich-club phenomenon across complex network hierarchies",
       Applied Physics Letters Vol 91 Issue 8, August 2007.
       https://arxiv.org/abs/physics/0701290
    .. [2] R. Milo, N. Kashtan, S. Itzkovitz, M. E. J. Newman, U. Alon,
       "Uniform generation of random graphs with arbitrary degree
       sequences", 2006. https://arxiv.org/abs/cond-mat/0312028

    """

    modes = ["fast", "medium", "slow"]
    shortname = "RC"
    name = "rich_club"
    encoding = "networkx"

    def compute_features(self):
        @lru_cache(maxsize=None)
        def eval_rich_club(graph):
            # extracting feature matrix
            return list(
                nx.rich_club_coefficient(remove_selfloops(graph), normalized=False).values()
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
