"""Communities Modularity propagation class."""
from functools import lru_cache
from networkx.algorithms.community import greedy_modularity_communities

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "CommunitiesModularity"


class CommunitiesModularity(FeatureClass):
    """Communities Modularity propagation class."""

    modes = ["fast", "medium", "slow"]
    shortname = "CM"
    name = "communities_modularity"
    encoding = "networkx"

    def compute_features(self):
        """Compute the measures about community detection using the modularity algorithm.

        Notes
        -----
        """

        @lru_cache(maxsize=None)
        def eval_modularity(graph):
            """this evaluates the main function and cach it for speed up."""
            communities = [set(comm) for comm in greedy_modularity_communities(graph)]
            communities.sort(key=len, reverse=True)
            return communities

        self.add_feature(
            "num_communities",
            lambda graph: len(eval_modularity(graph)),
            "Number of communities",
            InterpretabilityScore(4),
        )

        self.add_feature(
            "largest_commsize",
            lambda graph: len(eval_modularity(graph)[0]),
            "The ratio of the largest and second largest communities using greedy modularity",
            InterpretabilityScore(4),
        )

        self.add_feature(
            "ratio_commsize_maxmin",
            lambda graph: len(eval_modularity(graph)[0])
            / len(eval_modularity(graph)[-1]),
            "The ratio of the largest and second largest communities using greedy modularity",
            InterpretabilityScore(3),
        )

        self.add_feature(
            "communities",
            lambda graph: eval_modularity(graph),
            "The optimal partition using greedy modularity algorithm",
            InterpretabilityScore(3),
            statistics="clustering",
        )
