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
from hcga.Operations import utils
import numpy as np

class Eccentricity():
    """
    Eccentricity class
    """
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = {}
        
    def feature_extraction(self):
        """
        Compute eccentricity for each node.
        
        The eccentricity of a node is the maximum distance of the node from 
        any other node in the graph [1]_.
        
        Parameters
        ----------
        G : graph
          A networkx graph


        Returns
        -------
        feature_list : dict
           Dictionary of features related to eccentricity.


        Notes
        -----
        Eccentricity using networkx:
            `Networkx_eccentricity <https://networkx.github.io/documentation/stable/reference/algorithms/distance_measures.html>`_   
        
        
        References
        ----------
        .. [1] F.W. Takes and W.A. Kosters, Computing the Eccentricity Distribution of
    Large Graphs, Algorithms 6(1): 100-118, 2013.
    doi: https://doi.org/10.3390/a6010100          
            
        

        
        """
        
        # Defining the input arguments
        bins = [10,20,50]
        

        
        G = self.G
        feature_list = {}
        if not nx.is_directed(G) or (nx.is_directed(G) and nx.is_strongly_connected(G)):
            #Calculate the eccentricity of each node
            eccentricity = np.asarray(list(nx.eccentricity(G).values()))
            # Basic stats regarding the eccentricity distribution
            feature_list['mean'] = eccentricity.mean()
            feature_list['std'] = eccentricity.std()
            feature_list['max'] = eccentricity.max()
            feature_list['min'] = eccentricity.min()
            
            for i in range(len(bins)):

                
                # Fitting the eccentricity distribution and finding the optimal
                # distribution according to SSE
                opt_mod,opt_mod_sse = utils.best_fit_distribution(eccentricity,bins=bins[i])
                feature_list['opt_model_{}'.format(bins[i])] = opt_mod
    
                # Fitting power law and finding 'a' and the SSE of fit.
                feature_list['powerlaw_a_{}'.format(bins[i])] = utils.power_law_fit(eccentricity,bins=bins[i])[0][-2]# value 'a' in power law
                feature_list['powerlaw_SSE_{}'.format(bins[i])] = utils.power_law_fit(eccentricity,bins=bins[i])[1] # value sse in power law
        else:
            feature_list['mean'] = np.nan
            feature_list['std'] = np.nan
            feature_list['max'] = np.nan
            feature_list['min'] = np.nan
            for i in range(len(bins)):
                feature_list['opt_model_{}'.format(bins[i])] = np.nan
                feature_list['powerlaw_a_{}'.format(bins[i])] = np.nan
                feature_list['powerlaw_SSE_{}'.format(bins[i])] = np.nan


        self.features = feature_list
       
