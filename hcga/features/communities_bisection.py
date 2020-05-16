"""Communities Bisection class."""
from functools import lru_cache
from networkx.algorithms.community import kernighan_lin_bisection

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "CommunitiesBisection"


class CommunitiesBisection(FeatureClass):
    """Communities Bisection class."""

    modes = ["medium", "slow"]
    shortname = "CBI"
    name = "communities_bisection"
    encoding = "networkx"

    def compute_features(self):
        @lru_cache(maxsize=None)
        def eval_bisection(graph):
            """this evaluates the main function and cach it for speed up."""
            communities = list(kernighan_lin_bisection(graph))
            communities.sort(key=len, reverse=True)

            return communities

        self.add_feature(
            "largest_commsize",
            lambda graph: len(eval_bisection(graph)[0]),
            "The ratio of the largest and second largest communities using bisection algorithm",
            InterpretabilityScore(4),
        )

        self.add_feature(
            "partition",
            eval_bisection,
            "The optimal partition for kernighan lin bisection algorithm",
            InterpretabilityScore(4),
            statistics="clustering",
        )
