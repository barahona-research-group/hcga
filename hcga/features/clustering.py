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
import networkx as nx


featureclass_name = 'Clustering'

class Clustering(FeatureClass):
    """Clustering class"""

    modes = ['fast','medium', 'slow']
    shortname = 'CLU'
    name = 'clustering'
    keywords = []
    normalize_features = True

    def compute_features(self):
        """
        Compute some clustering based measures for the network

        Computed statistics    
        -----
        Put here the list of things that are computed, with corresponding names

        """
        
        # computing the number of triangles that include a node as one vertex
        triangles = np.asarray(list(nx.triangles(self.graph).values()))        
        summary_statistics(self.add_feature, triangles, 
                'triangles', 'the distribution of triangles that include a node as one vertex', InterpretabilityScore(3))  
        
        
        # graph transitivity
        self.add_feature('transitivity', nx.transitivity(self.graph), 
                'Possible triangles are identified by the number of “triads” (two edges with a shared vertex)',
                InterpretabilityScore(3))
        
        # clustering
        triangle_clustering = list(nx.clustering(self.graph).values())
        summary_statistics(self.add_feature, triangle_clustering, 
                'triangle clustering', 'the clustering of a node u is the fraction of possible triangles through that node',
                InterpretabilityScore(3))  
        
        # square clustering
        square_clustering = list(nx.square_clustering(self.graph).values())
        summary_statistics(self.add_feature, square_clustering, 
                'square clustering', 'the clustering of a node u is the fraction of possible squares through that node',
                InterpretabilityScore(3))  
        
        

