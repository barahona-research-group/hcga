"""Communities Label propagation class."""

from functools import lru_cache

from networkx.algorithms.community import label_propagation_communities

from hcga.feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "CommunitiesLabelPropagation"


@lru_cache(maxsize=None)
def eval_labelprop(graph):
    """this evaluates the main function and cach it for speed up."""
    communities = list(label_propagation_communities(graph))
    communities.sort(key=len, reverse=True)

    return communities


def largest_commsize(graph):
    """largest_commsize"""
    return len(eval_labelprop(graph)[0])


def ratio_commsize_maxmin(graph):
    """ratio_commsize_maxmin"""
    return len(eval_labelprop(graph)[0]) / len(eval_labelprop(graph)[-1])


class CommunitiesLabelPropagation(FeatureClass):
    """Communities Label propagation class.

    Features based on the community detection using label propagation.

    Uses networkx, see 'https://networkx.org/documentation/stable/reference/algorithms/\
        community.html`

    Finds communities in `G` using a semi-synchronous label propagation
    method[1]_. This method combines the advantages of both the synchronous
    and asynchronous models.

    References
    ----------
    .. [1] Cordasco, G., & Gargano, L. (2010, December). Community detection
       via semi-synchronous label propagation algorithms. In Business
       Applications of Social Network Analysis (BASNA), 2010 IEEE International
       Workshop on (pp. 1-8). IEEE.
    """

    modes = ["medium", "slow"]
    shortname = "CLP"
    name = "communities_labelprop"
    encoding = "networkx"

    def compute_features(self):
        self.add_feature(
            "largest_commsize",
            largest_commsize,
            "The ratio of the largest and second largest communities using label propagation",
            InterpretabilityScore(4),
        )

        self.add_feature(
            "ratio_commsize_maxmin",
            ratio_commsize_maxmin,
            "The ratio of the largest and second largest communities using label propagation",
            InterpretabilityScore(3),
        )

        self.add_feature(
            "communities",
            eval_labelprop,
            "The optimal partition using label propagation algorithm",
            InterpretabilityScore(3),
            statistics="clustering",
        )
