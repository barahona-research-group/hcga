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

import networkx as nx


from ..feature_class import FeatureClass, InterpretabilityScore

featureclass_name = "Efficiency"


class Efficiency(FeatureClass):
    """ Distance Measures class """

    modes = ["fast", "medium", "slow"]
    shortname = "EF"
    name = "efficiency"
    keywords = []
    normalize_features = True

    def compute_features(self):
        """
        Compute the efficiency measures of the network

        Computed statistics
        -----
        Put here the list of things that are computed, with corresponding names

        """

        # local effiency
        self.add_feature(
            "local_efficiency",
            lambda graph: nx.local_efficiency(graph),
            "The local efficiency",
            InterpretabilityScore(4),            
        )    
        
        # global effiency
        self.add_feature(
            "global_efficiency",
            lambda graph: nx.global_efficiency(graph),
            "The global efficiency",
            InterpretabilityScore(4),            
        )  

  


        
        
        
        
        
        
        
        
        



