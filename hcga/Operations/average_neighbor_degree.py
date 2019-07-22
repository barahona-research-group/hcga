
from networkx.algorithms import assortativity
from hcga.Operations import utils
import numpy as np

class AverageNeighborDegree():
    """
    Average neighbor degree class
    """
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = []
        
    def feature_extraction(self):
        """
        Compute average neighbor degree for each node
        
                Parameters
        ----------
        G : graph
          A networkx graph

        args :
            arg[0] Number of bins for calculating pdf of chosen distribution for SSE calculation

        Returns
        -------
        feature_list :list
           List of features related to average neighbor degree.


        Notes
        -----
        Average neighbor degree calculations using networkx:
            https://networkx.github.io/documentation/stable/reference/algorithms/assortativity.html
        
        """
        
        # Defining the input arguments
        bins = [10,20,50]
        """# Defining featurenames
        feature_names = ['mean','std','max','min']
        """
        
        G = self.G
        feature_list = {}
        #Calculate the average neighbor degree of each node
        average_neighbor_degree = np.asarray(list(assortativity.average_neighbor_degree(G).values()))
        # Basic stats regarding the average neighbor degree distribution
        feature_list['mean'] = average_neighbor_degree.mean()
        feature_list['std'] = average_neighbor_degree.std()
        feature_list['max'] = average_neighbor_degree.max()
        feature_list['min'] = average_neighbor_degree.min()
        
        for i in range(len(bins)):
            """# Adding to feature names
            feature_names.append('opt_model_{}'.format(bins[i]))
            feature_names.append('powerlaw_a_{}'.format(bins[i]))
            feature_names.append('powerlaw_SSE_{}'.format(bins[i]))"""
            
            # Fitting the average neighbor degree distribution and finding the optimal
            # distribution according to SSE
            opt_mod,opt_mod_sse = utils.best_fit_distribution(average_neighbor_degree,bins=bins[i])
            feature_list['opt_model_{}'.format(bins[i])] = opt_mod

            # Fitting power law and finding 'a' and the SSE of fit.
            feature_list['powerlaw_a_{}'.format(bins[i])] = utils.power_law_fit(average_neighbor_degree,bins=bins[i])[0][-2]# value 'a' in power law
            feature_list['powerlaw_SSE_{}'.format(bins[i])] = utils.power_law_fit(average_neighbor_degree,bins=bins[i])[1] # value sse in power law

        
        """
        self.feature_names=feature_names
        """
        
        self.features = feature_list
        
