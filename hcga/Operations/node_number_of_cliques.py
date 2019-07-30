
from networkx.algorithms import clique
from hcga.Operations import utils
import numpy as np
import networkx as nx

class NodeNumberOfCliques():
    """
    Node number of cliques class
    """
 
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = {}

    def feature_extraction(self):
        """
        Compute the number of maximal cliques for each node in the graph,
        i.e passing through that node
        
        
        Parameters
        ----------
        G : graph
          A networkx graph

        bins :
            Number of bins for calculating pdf of chosen distribution for SSE calculation

        Returns
        -------
        feature_list :list
           List of features related to the number of maximal cliques for
           each node.


        Notes
        -----
        Number of cliques calculations using networkx:
            `Networkx_cliques <https://networkx.github.io/documentation/stable/reference/algorithms/clique.html>`_
        """        
                
        # Defining the input arguments
        bins = [10,20,50]
        
        """
        # Defining featurenames
        feature_names = ['mean','std','max','min','median']
        """

        G = self.G

        feature_list = {}
        
        if not nx.is_directed(G):
            #Calculate the the number of maximal cliques for each node
            number_of_cliques = np.asarray(list(clique.number_of_cliques(G).values()))
    
            # Basic stats regarding the number of cliques distribution
            feature_list['mean'] = number_of_cliques.mean()
            feature_list['std'] = number_of_cliques.std()
            feature_list['max'] = number_of_cliques.max()
            feature_list['min'] = number_of_cliques.min()
            
            for i in range(len(bins)):
                """# Adding to feature names
                feature_names.append('opt_model_{}'.format(bins[i]))
                feature_names.append('powerlaw_a_{}'.format(bins[i]))
                feature_names.append('powerlaw_SSE_{}'.format(bins[i]))"""
                
                # Fitting the number of cliques distribution and finding the optimal
                # distribution according to SSE
                opt_mod,opt_mod_sse = utils.best_fit_distribution(number_of_cliques,bins=bins[i])
                feature_list['opt_model_{}'.format(bins[i])] = opt_mod
    
                # Fitting power law and finding 'a' and the SSE of fit.
                feature_list['powerlaw_a_{}'.format(bins[i])] = utils.power_law_fit(number_of_cliques,bins=bins[i])[0][-2]# value 'a' in power law
                feature_list['powerlaw_SSE_{}'.format(bins[i])] = utils.power_law_fit(number_of_cliques,bins=bins[i])[1] # value sse in power law
        else:
            feature_list['mean'] = np.nan
            feature_list['std'] = np.nan
            feature_list['max'] = np.nan
            feature_list['min'] = np.nan
            for i in range(len(bins)):
                feature_list['opt_model_{}'.format(bins[i])] = np.nan
                feature_list['powerlaw_a_{}'.format(bins[i])] = np.nan
                feature_list['powerlaw_SSE_{}'.format(bins[i])] = np.nan


        # Fitting normal distribution and finding...


        # Fitting exponential and finding ...
        
        """
        self.feature_names=feature_names
        """
        self.features = feature_list
