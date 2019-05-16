#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 10:41:52 2019

@author: robert
"""

from fa2 import ForceAtlas2
from hcga.Operations import utils
import numpy as np
import scipy as sp


class ForceCentrality():
    """
    Centrality eigenvector
    """
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = []

    def feature_extraction(self,args):

        """Compute the force centrality for the graph `G`.



        Parameters
        ----------
        G : graph
          A networkx graph

        args: list
            Parameters for calculating feature_list
                arg[0]: integer
                    number of bins


        Returns
        -------
        feature_list :list
           List of features related to force centrality.


        Notes
        -----



        """

        # Defining the input arguments
        bins = args[0]

        # Defining featurenames
        self.feature_names = ['mean','std','max','min','opt_mod','power_law_a','power_law_sse']

        G = self.G
        
        feature_list = []
        
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
        
        pos = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=2000)
        c = np.linalg.norm(np.array(list(pos.values())),axis=1)        
        
        # this changes each time so we must average of n_force times
        for i in range(n_force-1):
            pos = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=2000)
            c += np.linalg.norm(np.array(list(pos.values())), axis=1)                    
            c = -c/n_force


        feature_list.append(np.mean(c))
        feature_list.append(np.std(c))
        
        feature_list.append(np.max(abs(c)))
        feature_list.append(np.min(abs(c)))

        
        # Fitting the degree centrality distribution and finding the optimal
        # distribution according to SSE
        opt_mod,opt_mod_sse = utils.best_fit_distribution(c,bins=bins)
        feature_list.append(opt_mod)
        
        # Fitting power law and finding 'a' and the SSE of fit.
        feature_list.append(utils.power_law_fit(c,bins=bins)[0][-2]) # value 'a' in power law
        feature_list.append(utils.power_law_fit(c,bins=bins)[1]) # value sse in power law

        

        self.features = feature_list
