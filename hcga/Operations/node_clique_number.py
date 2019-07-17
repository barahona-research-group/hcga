
from networkx.algorithms import clique
from hcga.Operations import utils
import numpy as np


class NodeCliqueNumber():
    """
    Node clique number class
    """
 
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = []

    def feature_extraction(self,args):
        """
        Compute the maximal clique containing each node, i.e passing through
        that node.
        
        
        Parameters
        ----------
        G : graph
          A networkx graph

        args :
            arg[0] Number of bins for calculating pdf of chosen distribution for SSE calculation

        Returns
        -------
        feature_list :list
           List of features related to clique number.


        Notes
        -----
        Clique number calculations using networkx:
            https://networkx.github.io/documentation/stable/reference/algorithms/clique.html
        """        
                
        # Defining the input arguments
        bins = args[0]

        # Defining featurenames
        feature_names = ['mean','std','opt_model','powerlaw_a','powerlaw_SSE']

        G = self.G

        feature_list = []
        
        
        #Calculate the largest maximal clique containing each node
        node_clique_number = np.asarray(list(clique.node_clique_number(G).values()))

        # Basic stats regarding the node clique number distribution
        feature_list.append(node_clique_number.mean())
        feature_list.append(node_clique_number.std())

        # Fitting the node clique number distribution and finding the optimal
        # distribution according to SSE
        opt_mod,opt_mod_sse = utils.best_fit_distribution(node_clique_number,bins=bins)
        feature_list.append(opt_mod)

        # Fitting power law and finding 'a' and the SSE of fit.
        feature_list.append(utils.power_law_fit(node_clique_number,bins=bins)[0][-2])# value 'a' in power law
        feature_list.append(utils.power_law_fit(node_clique_number,bins=bins)[1])# value sse in power law

        # Fitting normal distribution and finding...


        # Fitting exponential and finding ...

        self.feature_names=feature_names
        self.features = feature_list
