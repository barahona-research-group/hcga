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
from networkx.algorithms import clique


featureclass_name = 'Cliques'

class Cliques(FeatureClass):
    """Basic stats class"""

    modes = ['fast','medium', 'slow']
    shortname = 'CL'
    name = 'cliques'
    keywords = []
    normalize_features = True

    def compute_features(self):
        """
        Compute some clique based measures for the network

        Computed statistics    
        -----
        Put here the list of things that are computed, with corresponding names

        """
        
        # graph clique number
        self.add_feature('graph_clique_number', clique.graph_clique_number(self.graph), 
                'The clique number of a graph is the size of the largest clique in the graph',
                InterpretabilityScore(3))
        
        # number of maximal cliques
        self.add_feature('num_max_cliques', clique.graph_number_of_cliques(self.graph), 
                'The number of maximal cliques in the graph',
                InterpretabilityScore(3))
        
        cliques = [u for u in list(clique.enumerate_all_cliques(self.graph)) if len(u)>1]            
        
        self.add_feature('num_cliques', len(cliques), 
                'The number of cliques in the graph',
                InterpretabilityScore(3))
        
        clique_sizes = [len(u) for u in cliques]
        summary_statistics(self.add_feature, clique_sizes, 
                'clique sizes', 'the distribution of clique sizes', InterpretabilityScore(3))       

        
           
        
        

