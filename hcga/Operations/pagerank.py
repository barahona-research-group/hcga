

import networkx as nx
from hcga.Operations import utils
import numpy as np

class PageRank():
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = []

    def feature_extraction(self):

        """
        Compute the PageRank of a network.
        
        The PageRank ranks the nodes in a network based on the number of
        incoming links to that node. in undirected graphs, each edge is 
        as an edge going in both directions.

        Parameters
        ----------
        G : graph
           A networkx graph
        args :
            arg[0] Number of bins for calculating pdf of chosen distribution for SSE calculation

        Returns
        -------
        feature_list :list
           List of features related to PageRank.


        Notes
        -----
        Implementation of networkx code:
            https://networkx.github.io/documentation/stable/reference/algorithms/link_analysis.html        

        """
        bins = [10,20,50]
        
        feature_names = ['mean','std','max','min']

        G = self.G

        feature_list = []
        #Calculate PageRank
        pagerank = np.asarray(list(nx.pagerank(G).values()))
        # Basic stats regarding the PageRank distribution
        feature_list.append(pagerank.mean())
        feature_list.append(pagerank.std())
        feature_list.append(pagerank.max())
        feature_list.append(pagerank.min())
        
        for i in range(len(bins)):
            # Adding to featurenames
            feature_names.append('opt_model_{}'.format(bins[i]))
            feature_names.append('powerlaw_a_{}'.format(bins[i]))
            feature_names.append('powerlaw_SSE_{}'.format(bins[i]))
            
            # Fitting the PageRank distribution and finding the optimal
            # distribution according to SSE
            opt_mod,opt_mod_sse = utils.best_fit_distribution(pagerank,bins=bins[i])
            feature_list.append(opt_mod)

            # Fitting power law and finding 'a' and the SSE of fit.
            feature_list.append(utils.power_law_fit(pagerank,bins=bins[i])[0][-2]) # value 'a' in power law
            feature_list.append(utils.power_law_fit(pagerank,bins=bins[i])[1]) # value sse in power law

        self.features = feature_list
        self.feature_names=feature_names
