"""Force centrality class."""
import logging
import numpy as np

try:
    from fa2 import ForceAtlas2

    fa2_error = False
except ImportError:
    fa2_error = True

from ..feature_class import FeatureClass, InterpretabilityScore

L = logging.getLogger(__name__)
if fa2_error:
    L.info("Install ForceAtlas2 if you want to use force centrality")

featureclass_name = "ForceCentrality"


class ForceCentrality(FeatureClass):
    """Force centrality class."""

    modes = ["medium", "slow"]
    shortname = "CF"
    name = "centralities_force"
    encoding = "networkx"

    def compute_features(self):

        # number of times to average force centrality
        n_force = 20

        # find node position with force atlas, and distance to the center is the centrality
        forceatlas2 = ForceAtlas2(
            # Tuning
            scalingRatio=2.0,
            strongGravityMode=False,
            gravity=1.0,
            # Log
            verbose=False,
        )

        def force_centrality(graph):
            c = []
            for _ in range(n_force):
                pos = forceatlas2.forceatlas2_networkx_layout(graph, pos=None,)
                c.append(np.linalg.norm(np.array(list(pos.values())), axis=1))

            return (np.vstack(c).mean(axis=0) / np.max(np.vstack(c))).tolist()

        self.add_feature(
            "force_centrality",
            force_centrality,
            "Force centrality is the distance from the centre of mass of the network \
            , larger values indicate further from the centre",
            InterpretabilityScore(4),
            statistics="centrality",
        )
