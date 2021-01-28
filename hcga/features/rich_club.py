"""Rich Club class."""
from functools import lru_cache

import networkx as nx
import numpy as np

from hcga.feature_class import FeatureClass, InterpretabilityScore
from hcga.features.utils import remove_selfloops

featureclass_name = "RichClub"


@lru_cache(maxsize=None)
def eval_rich_club(graph):
    """eval_rich_club"""
    return list(nx.rich_club_coefficient(remove_selfloops(graph), normalized=False).values())


def rich_club_k_1(graph):
    """rich_club_k_1"""
    return eval_rich_club(graph)[0]


def rich_club_k_max(graph):
    """rich_club_k_max"""
    return eval_rich_club(graph)[-1]


def rich_club_maxminratio(graph):
    """rich_club_maxminratio"""
    return np.min(eval_rich_club(graph)) / np.max(eval_rich_club(graph))


class RichClub(FeatureClass):
    """Rich Club class."""

    modes = ["fast", "medium", "slow"]
    shortname = "RC"
    name = "rich_club"
    encoding = "networkx"

    def compute_features(self):
        # k = 1
        self.add_feature(
            "rich_club_k_1",
            rich_club_k_1,
            "The rich-club coefficient is the ratio of the number of actual to the \
            number of potential edges for nodes with degree greater than k",
            InterpretabilityScore(4),
        )

        self.add_feature(
            "rich_club_k_max",
            rich_club_k_max,
            "The rich-club coefficient is the ratio of the number of actual to the number \
            of potential edges for nodes with degree greater than k",
            InterpretabilityScore(4),
        )

        self.add_feature(
            "rich_club_maxminratio",
            rich_club_maxminratio,
            "The ratio of the smallest to largest rich club coefficients",
            InterpretabilityScore(4),
        )

        self.add_feature(
            "rich_club",
            eval_rich_club,
            "The distribution of rich club coefficients",
            InterpretabilityScore(4),
            statistics="centrality",
        )
