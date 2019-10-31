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

from networkx.algorithms import clique
import numpy as np
import networkx as nx


class Cliques():
    """
    Cliques class
    """
 
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = {}

    def feature_extraction(self):
        """
        Compute the clique number for the graph and the number of maximal 
        cliques
        
        A clique is the maximal subset of nodes such that all the nodes in 
        this set are connected to each other. 
        The clique number is the size of the largest clique in a graph.
        
        
        Parameters
        ----------
        G : graph
          A networkx graph


        Returns
        -------
        feature_list : dict
           Dictionary of features related to cliques.


        Notes
        -----
        Clique number calculations using networkx:
            `Networkx <https://networkx.github.io/documentation/stable/reference/algorithms/clique.html>`_
        """        


        G = self.G

        feature_list = {}
        
        if not nx.is_directed(G):
            #Calculate the clique number for the graph
            feature_list['clique_number']=clique.graph_clique_number(G)
        
            #Calculate the number of maximal cliques in the graph
            feature_list['number_of_cliques']=clique.graph_number_of_cliques(G)
            
        else:
            feature_list['clique_number'] = np.nan
            feature_list['number_of_cliques'] = np.nan
        

        self.features = feature_list
