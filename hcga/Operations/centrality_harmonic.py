#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 20:13:58 2019

@author: Henry
"""

from networkx.algorithms import centrality
from hcga.Operations import utils
import numpy as np
import networkx as nx



class HarmonicCentrality():
    """
    Harmonic centrality class
    """
 
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = []

    def feature_extraction(self):
        """Compute the harmonic centrality for nodes.

        The harmonic centrality of a node is the sum of the reciprocal
        of the shortest paths from all other nodes in the graph


        Parameters
        ----------
        G : graph
          A networkx graph

        
        Returns
        -------
        feature_list :list
           List of features related to harmonic centrality.


        Notes
        -----
        Harmonic centrality calculations using networkx:
            https://networkx.github.io/documentation/stable/reference/algorithms/centrality.html 
        """
        
        # Defining the input arguments
        bins = [10,20,50]
        
        G = self.G

        feature_list = {}
        
        # Calculate the harmonic centrality of each node
        harmonic_centrality = np.asarray(list(centrality.harmonic_centrality(G).values()))
    
        # Basic stats regarding the harmonic centrality distribution
        feature_list['mean'] = harmonic_centrality.mean()
        feature_list['std'] = harmonic_centrality.std()
        feature_list['max'] = harmonic_centrality.max()
        feature_list['min'] = harmonic_centrality.min()
            
            
        for i in range(len(bins)):
            """# Adding to feature names
            feature_names.append('opt_model_{}'.format(bins[i]))
            feature_names.append('powerlaw_a_{}'.format(bins[i]))
            feature_names.append('powerlaw_SSE_{}'.format(bins[i]))"""
                
            # Fitting the harmonic centrality distribution and finding the optimal
            # distribution according to SSE
            opt_mod,opt_mod_sse = utils.best_fit_distribution(harmonic_centrality,bins=bins[i])
            feature_list['opt_model_{}'.format(bins[i])] = opt_mod
    
            # Fitting power law and finding 'a' and the SSE of fit.
            feature_list['powerlaw_a_{}'.format(bins[i])] = utils.power_law_fit(harmonic_centrality,bins=bins[i])[0][-2]# value 'a' in power law
            feature_list['powerlaw_SSE_{}'.format(bins[i])] = utils.power_law_fit(harmonic_centrality,bins=bins[i])[1] # value sse in power law
            


        self.features = feature_list