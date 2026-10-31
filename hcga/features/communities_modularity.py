"""Communities Modularity propagation class."""

from functools import lru_cache, partial

from networkx.algorithms.community import greedy_modularity_communities, quality

from hcga.feature_class import FeatureClass, InterpretabilityScore

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


def _modularity_unweighted(graph, communities):
    """Modularity of a partition, ignoring edge weights."""
    return quality.modularity(graph, communities, weight=None)


class CommunitiesModularity(FeatureClass):
    """Communities Modularity propagation class.

    Features based on the community detection using modularity.

    Uses networkx, see 'https://networkx.org/documentation/stable/reference/algorithms/\
        community.html`

    Find communities in graph using Clauset-Newman-Moore greedy modularity maximization,
    once ignoring edge weights and once using them (the ``_weighted`` features).
    The modularity of each partition is computed in the same way (unweighted or weighted);
    coverage and performance ignore edge weights.

    Greedy modularity maximization begins with each node in its own community and joins
    the pair of communities that most increases modularity until no such pair exists.

    References
    ----------
    .. [1] M. E. J Newman 'Networks: An Introduction', page 224
       Oxford University Press 2011.
    .. [2] Clauset, A., Newman, M. E., & Moore, C.
       "Finding community structure in very large networks."
       Physical Review E 70(6), 2004.

    """

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
            "The size of the largest community using greedy modularity",
            InterpretabilityScore(4),
        )

        self.add_feature(
            "ratio_commsize_maxmin",
            ratio_commsize_maxmin,
            "The size ratio of the largest and smallest communities using greedy modularity",
            InterpretabilityScore(3),
        )

        # Partition quality of the unweighted partition. Done here rather than with
        # statistics="clustering" so that its modularity also ignores edge weights.
        partition = self.evaluate_feature(eval_modularity, "communities", statistics="clustering")
        compl_desc = " of the partition of the optimal partition using greedy modularity algorithm"
        self.add_feature(
            "communities_modularity",
            partial(_modularity_unweighted, communities=partition),
            "Modularity" + compl_desc,
            InterpretabilityScore(3),
        )
        self.add_feature(
            "communities_coverage",
            lambda graph: quality.partition_quality(graph, partition=partition)[0],
            "Coverage" + compl_desc,
            InterpretabilityScore(3),
        )
        self.add_feature(
            "communities_performance",
            lambda graph: quality.partition_quality(graph, partition=partition)[1],
            "Performance" + compl_desc,
            InterpretabilityScore(3),
        )

        # add weighted features
        self.add_feature(
            "num_communities_weighted",
            num_communities_weighted,
            "Number of communities using weighted greedy modularity",
            InterpretabilityScore(4),
        )

        self.add_feature(
            "largest_commsize_weighted",
            largest_commsize_weighted,
            "The size of the largest community using weighted greedy modularity",
            InterpretabilityScore(4),
        )

        self.add_feature(
            "ratio_commsize_maxmin_weighted",
            ratio_commsize_maxmin_weighted,
            "The size ratio of the largest and smallest communities using weighted greedy modularity",
            InterpretabilityScore(3),
        )

        self.add_feature(
            "communities_weighted",
            partial(eval_modularity, weight="weight"),
            "The optimal partition using weighted greedy modularity algorithm",
            InterpretabilityScore(3),
            statistics="clustering",
        )
