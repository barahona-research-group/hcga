"""Communities Bisection class."""

from functools import lru_cache

from networkx.algorithms.community import kernighan_lin_bisection

from hcga.feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "CommunitiesBisection"


@lru_cache(maxsize=None)
def eval_bisection(graph):
    """this evaluates the main function and cach it for speed up."""
    communities = list(kernighan_lin_bisection(graph))
    communities.sort(key=len, reverse=True)

    return communities


def largest_commsize(graph):
    """largest_commsize"""
    return len(eval_bisection(graph)[0])


class CommunitiesBisection(FeatureClass):
    """Communities Bisection class.

    This algorithm partitions a network into two sets by iteratively
    swapping pairs of nodes to reduce the edge cut between the two sets.  The
    pairs are chosen according to a modified form of Kernighan-Lin, which
    moves node individually, alternating between sides to keep the bisection
    balanced.

    References
    ----------
    .. [1] Kernighan, B. W.; Lin, Shen (1970).
       "An efficient heuristic procedure for partitioning graphs."
       *Bell Systems Technical Journal* 49: 291--307.
       Oxford University Press 2011.

    """

    modes = ["medium", "slow"]
    shortname = "CBI"
    name = "communities_bisection"
    encoding = "networkx"

    def compute_features(self):
        self.add_feature(
            "largest_commsize",
            largest_commsize,
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
