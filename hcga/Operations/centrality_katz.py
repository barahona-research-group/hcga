
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
        
    def feature_extraction(self):
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
        bins = [10,20,50]
        
        """
        # Defining featurenames
        feature_names = ['mean','std','max','min']
        """
        
        G = self.G
        feature_list = {}
        
        
        #Calculate the Katz centrality of each node
        katz_centrality = np.asarray(list(centrality.katz_centrality(G).values()))
        # Basic stats regarding the Katz centrality distribution
        feature_list['mean'] = katz_centrality.mean()
        feature_list['std'] = katz_centrality.std()
        feature_list['max'] = katz_centrality.max()
        feature_list['min'] = katz_centrality.min()
        
        for i in range(len(bins)):
            """# Adding to feature names
            feature_names.append('opt_model_{}'.format(bins[i]))
            feature_names.append('powerlaw_a_{}'.format(bins[i]))
            feature_names.append('powerlaw_SSE_{}'.format(bins[i]))"""
                
            # Fitting the katz centrality distribution and finding the optimal
            # distribution according to SSE
            opt_mod,opt_mod_sse = utils.best_fit_distribution(katz_centrality,bins=bins[i])
            feature_list['opt_model_{}'.format(bins[i])] = opt_mod
                
            # Fitting power law and finding 'a' and the SSE of fit.
            feature_list['powerlaw_a_{}'.format(bins[i])] = utils.power_law_fit(katz_centrality,bins=bins[i])[0][-2]# value 'a' in power law
            feature_list['powerlaw_SSE_{}'.format(bins[i])] = utils.power_law_fit(katz_centrality,bins=bins[i])[1] # value sse in power law

        """
        self.feature_names=feature_names
        """
        
        self.features = feature_list
       