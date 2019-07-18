
from networkx.algorithms import centrality
from hcga.Operations import utils
import numpy as np

class KatzCentrality():
    """
    Eccentricity class
    """
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = []
        
    def feature_extraction(self,args):
        """
        Compute Katz centrality for each node
        
        Katz centrality is similar to eigenvector centrality with the main
        difference being that each node is given a base value of centrality.
        
        Parameters
        ----------
        G : graph
          A networkx graph

        args :
            arg[0] Number of bins for calculating pdf of chosen distribution for SSE calculation

        Returns
        -------
        feature_list :list
           List of features related to Katz centrality.


        Notes
        -----
        Katz centrality using networkx:
            https://networkx.github.io/documentation/stable/reference/algorithms/centrality.html        """
        
        # Defining the input arguments
        bins = args[0]
        # Defining featurenames
        feature_names = ['mean','std','opt_model','powerlaw_a','powerlaw_SSE']
        G = self.G
        feature_list = []
        #Calculate the Katz centrality of each node
        katz_centrality = np.asarray(list(centrality.katz_centrality(G).values()))
        # Basic stats regarding the Katz centrality distribution
        feature_list.append(katz_centrality.mean())
        feature_list.append(katz_centrality.std())
        
        # Fitting the Katz centrality distribution and finding the optimal
        # distribution according to SSE
        opt_mod,opt_mod_sse = utils.best_fit_distribution(katz_centrality,bins=bins)
        feature_list.append(opt_mod)

        # Fitting power law and finding 'a' and the SSE of fit.
        feature_list.append(utils.power_law_fit(katz_centrality,bins=bins)[0][-2])# value 'a' in power law
        feature_list.append(utils.power_law_fit(katz_centrality,bins=bins)[1])# value sse in power law

        
        self.feature_names=feature_names
        self.features = feature_list
       