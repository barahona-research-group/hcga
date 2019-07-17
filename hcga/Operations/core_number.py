import networkx as nx
from hcga.Operations import utils
import numpy as np

class CoreNumber():
    """
    Core number class
    """
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = []
        
    def feature_extraction(self,args):
        """
        Compute core number for each node
        
        A k-core is the maximal subgraph that contains nodes of degree k or 
        higher. 
        The core number for a node is the largest value of k such that a 
        k-core containing that node exists.
        
                Parameters
        ----------
        G : graph
          A networkx graph

        args :
            arg[0] Number of bins for calculating pdf of chosen distribution for SSE calculation

        Returns
        -------
        feature_list :list
           List of features related to core number.


        Notes
        -----
        Core number calculations using networkx:
            https://networkx.github.io/documentation/stable/reference/algorithms/core.html
        """
        
        # Defining the input arguments
        bins = args[0]
        # Defining featurenames
        feature_names = ['mean','std','opt_model','powerlaw_a','powerlaw_SSE']
        G = self.G
        feature_list = []
        #Calculate the core number of each node
        core_number = np.asarray(list(nx.core_number(G).values()))
        # Basic stats regarding the core number distribution
        feature_list.append(core_number.mean())
        feature_list.append(core_number.std())
        
        # Fitting the core number distribution and finding the optimal
        # distribution according to SSE
        opt_mod,opt_mod_sse = utils.best_fit_distribution(core_number,bins=bins)
        feature_list.append(opt_mod)

        # Fitting power law and finding 'a' and the SSE of fit.
        feature_list.append(utils.power_law_fit(core_number,bins=bins)[0][-2])# value 'a' in power law
        feature_list.append(utils.power_law_fit(core_number,bins=bins)[1])# value sse in power law

        
        self.feature_names=feature_names
        self.features = feature_list