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

from .feature_class import FeatureClass
from .feature_class import InterpretabilityScore
from ..feature_utils import summary_statistics
import numpy as np
from networkx.algorithms import centrality
import networkx as nx
from fa2 import ForceAtlas2


featureclass_name = 'ForceCentrality'

class ForceCentrality(FeatureClass):
    """Basic stats class"""

    modes = ['medium', 'slow']
    shortname = 'CF'
    name = 'centralities_force'
    keywords = []
    normalize_features = True

    def compute_features(self):
        """
        Compute some standard centrality measures for the network

        Computed statistics    
        -----
        Put here the list of things that are computed, with corresponding names

        """

        # number of times to average force centrality
        n_force = 20
        
        #find node position with force atlas, and distance to the center is the centrality
        forceatlas2 = ForceAtlas2(
                # Tuning
                scalingRatio=2.0,
                strongGravityMode=False,
                gravity=1.0,
                # Log
                verbose=False)


        # this changes each time so we must average of n_force times
        c = []
        for i in range(n_force):
            try:
                pos = forceatlas2.forceatlas2_networkx_layout(self.graph, pos=None, iterations=2000)
                c.append(np.linalg.norm(np.array(list(pos.values())), axis=1))                

            except Exception as e:
                print('Exception for centrality_force', e)
        
        # force centrality - larger values indicate further from centre of mass
        force_centrality = np.vstack(c).mean(axis=0)/np.max(np.vstack(c))
        summary_statistics(self.add_feature, force_centrality, 
                'force centrality', 
                'Force centrality is the distance from the centre of mass of the network - larger values indicate further from the centre', 
                InterpretabilityScore(5))               
         
      
        
        

