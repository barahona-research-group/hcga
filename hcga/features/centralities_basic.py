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

featureclass_name = 'CentralitiesBasic'

class CentralitiesBasic(FeatureClass):
    """Basic stats class"""

    modes = ['fast', 'medium', 'slow']
    shortname = 'CB'
    name = 'centralities_basic'
    keywords = []
    normalize_features = True

    def compute_features(self):
        """
        Compute some standard centrality measures for the network

        Computed statistics    
        -----
        Put here the list of things that are computed, with corresponding names

        """

        # Degree centrality
        degree_centrality = np.asarray(list(centrality.degree_centrality(self.graph).values()))
        summary_statistics(self.add_feature, degree_centrality, 
                'degree centrality', 
                'The degree centrality distribution', 
                InterpretabilityScore(5))               
         
    
        
        # Betweenness Centrality
        betweenness_centrality = np.asarray(list(centrality.betweenness_centrality(self.graph).values()))
        summary_statistics(self.add_feature, betweenness_centrality, 
                'betweenness centrality', 
                'Betweenness centrality of a node v is the sum of the fraction of all-pairs shortest paths that pass through v', 
                InterpretabilityScore(5))       

        # Closeness centrality
        closeness_centrality = np.asarray(list(centrality.closeness_centrality(self.graph).values()))
        summary_statistics(self.add_feature, closeness_centrality, 
                'closeness centrality', 
                'Closeness is the reciprocal of the average shortest path distance', 
                InterpretabilityScore(5))   
        
        # Edge betweenness centrality
        edge_betweenness_centrality = np.asarray(list(centrality.edge_betweenness_centrality(self.graph).values()))
        summary_statistics(self.add_feature, edge_betweenness_centrality, 
                'edge betweenness centrality', 
                'Betweenness centrality of an edge e is the sum of the fraction of all-pairs shortest paths that pass through e', 
                InterpretabilityScore(4))          

        # Harmonic centrality
        harmonic_centrality = np.asarray(list(centrality.harmonic_centrality(self.graph).values()))
        summary_statistics(self.add_feature, harmonic_centrality, 
                'harmonic centrality', 
                'Harmonic centrality of a node u is the sum of the reciprocal of the shortest path distances from all other nodes to u', 
                InterpretabilityScore(4))      

        # Harmonic centrality
        harmonic_centrality = np.asarray(list(centrality.harmonic_centrality(self.graph).values()))
        summary_statistics(self.add_feature, harmonic_centrality, 
                'harmonic centrality', 
                'Harmonic centrality of a node u is the sum of the reciprocal of the shortest path distances from all other nodes to u', 
                InterpretabilityScore(4))    

        # Subgraph centrality
        subgraph_centrality = np.asarray(list(centrality.subgraph_centrality(self.graph).values()))
        summary_statistics(self.add_feature, subgraph_centrality, 
                'subgraph centrality', 
                'The subgraph centrality for a node is the sum of weighted closed walks of all lengths starting and ending at that node.', 
                InterpretabilityScore(3))   

        # Second order centrality
        connected_graph = nx.subgraph(self.graph,max(nx.connected_components(self.graph), key=len))
        second_order_centrality = np.asarray(list(centrality.second_order_centrality(connected_graph).values()))
        summary_statistics(self.add_feature, second_order_centrality, 
                'second order centrality', 
                'The second order centrality of a given node is the standard deviation of the return times to that node of a perpetual random walk on G', 
                InterpretabilityScore(4))   

        
        # Eigenvector centrality
        try:
            eigenvector_centrality = np.asarray(list(centrality.eigenvector_centrality(self.graph,max_iter=100).values()))
        except:
            eigenvector_centrality = np.empty(len(self.graph))
            eigenvector_centrality[:] = 1 
            
        summary_statistics(self.add_feature, eigenvector_centrality, 
                'eigenvector centrality', 
                'Eigenvector centrality computes the centrality for a node based on the centrality of its neighbors', 
                InterpretabilityScore(4))    
        
        # Katz centrality
        try:
            katz_centrality = np.asarray(list(centrality.katz_centrality(self.graph).values()))
        except:
            katz_centrality = np.empty(len(self.graph))
            katz_centrality[:] = 1 
            
        summary_statistics(self.add_feature, katz_centrality, 
                'katz centrality', 
                'Generalisation of eigenvector centrality - Katz centrality computes the centrality for a node based on the centrality of its neighbors', 
                InterpretabilityScore(4))            
        
        

