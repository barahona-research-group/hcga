"""Communities Modularity propagation class."""
from functools import lru_cache, partial

from networkx.algorithms.community import greedy_modularity_communities

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "CommunitiesModularity"


@lru_cache(maxsize=None)
def eval_modularity(graph, weight=None):
    """this evaluates the main function and cach it for speed up."""
    communities = [set(comm) for comm in greedy_modularity_communities(graph, weight=weight)]
    communities.sort(key=len, reverse=True)
    return communities


def num_communities(graph):
    """num_communities"""
    return len(eval_modularity(graph))


def largest_commsize(graph):
    """largest_commsize"""
    return len(eval_modularity(graph)[0])


def ratio_commsize_maxmin(graph):
    """ratio_commsize_maxmin"""
    return len(eval_modularity(graph)[0]) / len(eval_modularity(graph)[-1])


def num_communities_weighted(graph):
    """num_communities_weighted"""
    return len(eval_modularity(graph, weight="weight"))


def largest_commsize_weighted(graph):
    """largest_commsize_weighted"""
    return len(eval_modularity(graph, weight="weight")[0])


def ratio_commsize_maxmin_weighted(graph):
    """ratio_commsize_maxmin_weighted"""
    return len(eval_modularity(graph, weight="weight")[0]) / len(
        eval_modularity(graph, weight="weight")[-1]
    )


class CommunitiesModularity(FeatureClass):
    """Communities Modularity propagation class."""

    modes = ["fast", "medium", "slow"]
    shortname = "CM"
    name = "communities_modularity"
    encoding = "networkx"

    def compute_features(self):
        # add unweighted features
        self.add_feature(
            "num_communities",
            num_communities,
            "Number of communities",
            InterpretabilityScore(4),
        )

        self.add_feature(
            "largest_commsize",
            largest_commsize,
            "The ratio of the largest and second largest communities using greedy modularity",
            InterpretabilityScore(4),
        )

        self.add_feature(
            "ratio_commsize_maxmin",
            ratio_commsize_maxmin,
            "The ratio of the largest and second largest communities using greedy modularity",
            InterpretabilityScore(3),
        )

        self.add_feature(
            "communities",
            eval_modularity,
            "The optimal partition using greedy modularity algorithm",
            InterpretabilityScore(3),
            statistics="clustering",
        )

        # add weighted features
        self.add_feature(
            "num_communities_weighted",
            num_communities_weighted,
            "Number of communities",
            InterpretabilityScore(4),
        )

        self.add_feature(
            "largest_commsize_weighted",
            largest_commsize_weighted,
            "The ratio of the largest and second largest communities using greedy modularity",
            InterpretabilityScore(4),
        )

        self.add_feature(
            "ratio_commsize_maxmin_weighted",
            ratio_commsize_maxmin_weighted,
            "The ratio of the largest and second largest communities using greedy modularity",
            InterpretabilityScore(3),
        )

        self.add_feature(
            "communities_weighted",
            partial(eval_modularity, weight="weight"),
            "The optimal partition using greedy modularity algorithm",
            InterpretabilityScore(3),
            statistics="clustering",
        )
