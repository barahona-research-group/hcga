"""Communities Label propagation class."""
from functools import lru_cache
from networkx.algorithms.community import label_propagation_communities

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "CommunitiesLabelPropagation"


class CommunitiesLabelPropagation(FeatureClass):
    """Communities Label propagation class."""

    modes = ["medium", "slow"]
    shortname = "CLP"
    name = "communities_labelprop"
    encoding = "networkx"

    def compute_features(self):
        @lru_cache(maxsize=None)
        def eval_labelprop(graph):
            """this evaluates the main function and cach it for speed up."""
            communities = list(label_propagation_communities(graph))
            communities.sort(key=len, reverse=True)

            return communities

        self.add_feature(
            "largest_commsize",
            lambda graph: len(eval_labelprop(graph)[0]),
            "The ratio of the largest and second largest communities using label propagation",
            InterpretabilityScore(4),
        )

        self.add_feature(
            "ratio_commsize_maxmin",
            lambda graph: len(eval_labelprop(graph)[0])
            / len(eval_labelprop(graph)[-1]),
            "The ratio of the largest and second largest communities using label propagation",
            InterpretabilityScore(3),
        )

        self.add_feature(
            "communities",
            lambda graph: eval_labelprop(graph),
            "The optimal partition using label propagation algorithm",
            InterpretabilityScore(3),
            statistics="clustering",
        )
