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
    Force centrality class
    """
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = {}

    def feature_extraction(self):

        """Compute the force centrality for the graph `G`.



        Parameters
        ----------
        G : graph
          A networkx graph

        bins:
            Number of bins for calculating pdf of chosen distribution
            for SSE calculation


        Returns
        -------
        feature_list :list
           List of features related to force centrality.


        Notes
        -----



        """

        # Defining the input arguments
        bins = [10]
        
        """
        # Defining featurenames
        self.feature_names = ['mean','std','max','min']
        """
        
        G = self.G
        
        feature_list = {}
        
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
        
	# producing a zero array in case force atlas fails.
	c = np.zeros(G.number_of_nodes())

        try:
            pos = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=2000)
            c = np.linalg.norm(np.array(list(pos.values())),axis=1)        
        except:
            print('An exception occurred in ForceAtlas2')
            
            
        # this changes each time so we must average of n_force times
        for i in range(n_force-1):
            try:
                pos = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=2000)
                c += np.linalg.norm(np.array(list(pos.values())), axis=1)                    
                c = -c/n_force
            except:
                print('An exception occurred in ForceAtlas2')


        feature_list['mean']=np.mean(c)
        feature_list['std']=np.std(c)
        
        feature_list['max']=np.max(abs(c))
        feature_list['min']=np.min(abs(c))

        
        for i in range(len(bins)):
            """# Adding to feature names
            feature_names.append('opt_model_{}'.format(bins[i]))
            feature_names.append('powerlaw_a_{}'.format(bins[i]))
            feature_names.append('powerlaw_SSE_{}'.format(bins[i]))"""
            
            # Fitting the c distribution and finding the optimal
            # distribution according to SSE
            opt_mod,opt_mod_sse = utils.best_fit_distribution(c,bins=bins[i])
            feature_list['opt_model_{}'.format(bins[i])] = opt_mod

            # Fitting power law and finding 'a' and the SSE of fit.
            feature_list['powerlaw_a_{}'.format(bins[i])] = utils.power_law_fit(c,bins=bins[i])[0][-2]# value 'a' in power law
            feature_list['powerlaw_SSE_{}'.format(bins[i])] = utils.power_law_fit(c,bins=bins[i])[1] # value sse in power law

        

        self.features = feature_list
