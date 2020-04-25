# -*- coding: utf-8 -*-
# This file is part of hcga.
#
# Copyright (C) 2019,
# Robert Peach (r.peach13@imperial.ac.uk),
# Alexis Arnaudon (alexis.arnaudon@epfl.ch),
# https://github.com/ImperialCollegeLondon/hcga.git
#
# hcga is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# hcga is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with hcga.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from fa2 import ForceAtlas2

from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "ForceCentrality"


class ForceCentrality(FeatureClass):
    """Basic stats class"""

    modes = ["medium", "slow"]
    shortname = "CF"
    name = "centralities_force"
    encoding = 'networkx' 

    def compute_features(self):
        """
        Compute some standard centrality measures for the network

        Computed statistics
        -----
        Put here the list of things that are computed, with corresponding names

        """

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
            "Force centrality is the distance from the centre of mass of the network - larger values indicate further from the centre",
            InterpretabilityScore(4),
            statistics="centrality",
        )
