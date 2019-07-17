
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
        
    def feature_extraction(self,args):
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
        bins = args[0]
        # Defining featurenames
        feature_names = ['mean','std','opt_model','powerlaw_a','powerlaw_SSE']
        G = self.G
        feature_list = []
        #Calculate the average neighbor degree of each node
        # Basic stats regarding the average neighbor degree distribution
        average_neighbor_degree = np.asarray(list(assortativity.average_neighbor_degree(G).values()))
        feature_list.append(average_neighbor_degree.mean())
        feature_list.append(average_neighbor_degree.std())
        
        # Fitting the average neighbor degree distribution and finding the optimal
        # distribution according to SSE
        opt_mod,opt_mod_sse = utils.best_fit_distribution(average_neighbor_degree,bins=bins)
        feature_list.append(opt_mod)

        # Fitting power law and finding 'a' and the SSE of fit.
        feature_list.append(utils.power_law_fit(average_neighbor_degree,bins=bins)[0][-2])# value 'a' in power law
        feature_list.append(utils.power_law_fit(average_neighbor_degree,bins=bins)[1])# value sse in power law

        
        self.feature_names=feature_names
        self.features = feature_list
        
