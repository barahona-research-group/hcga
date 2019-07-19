from networkx.algorithms import centrality
from hcga.Operations import utils
import numpy as np



class DegreeCentrality():
    """
    Degree centrality class
    """
 
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = []

    def feature_extraction(self):
        """Compute the degree centrality for nodes.

        The degree centrality for a node v is the fraction of nodes it
        is connected to.


        Parameters
        ----------
        G : graph
          A networkx graph

        args :
            arg[0] Number of bins for calculating pdf of chosen distribution for SSE calculation

        Returns
        -------
        feature_list :list
           List of features related to degree centrality.


        Notes
        -----
        Degree centrality calculations using networkx:
            https://networkx.github.io/documentation/stable/_modules/networkx/algorithms/centrality/degree_alg.html#degree_centrality
            
        The degree centrality values are normalized by dividing by the maximum
        possible degree in a simple graph n-1 where n is the number of nodes in G.

        For multigraphs or graphs with self loops the maximum degree might
        be higher than n-1 and values of degree centrality greater than 1
        are possible.
        """
        # Defining the input arguments
        bins = [10,20,50]

        # Defining featurenames
        feature_names = ['mean','std','max','min']

        G = self.G

        feature_list = []

        #Calculate the degree centrality of each node
        degree_centrality = np.asarray(list(centrality.degree_centrality(G).values()))

        # Basic stats regarding the degree centrality distribution
        feature_list.append(degree_centrality.mean())
        feature_list.append(degree_centrality.std())
        feature_list.append(degree_centrality.max())
        feature_list.append(degree_centrality.min())
        
        
        for i in range(len(bins)):
            # Adding to featurenames
            feature_names.append('opt_model_{}'.format(bins[i]))
            feature_names.append('powerlaw_a_{}'.format(bins[i]))
            feature_names.append('powerlaw_SSE_{}'.format(bins[i]))
            
            # Fitting the degree centrality distribution and finding the optimal
            # distribution according to SSE
            opt_mod,opt_mod_sse = utils.best_fit_distribution(degree_centrality,bins=bins[i])
            feature_list.append(opt_mod)

            # Fitting power law and finding 'a' and the SSE of fit.
            feature_list.append(utils.power_law_fit(degree_centrality,bins=bins[i])[0][-2]) # value 'a' in power law
            feature_list.append(utils.power_law_fit(degree_centrality,bins=bins[i])[1]) # value sse in power law

        # Fitting normal distribution and finding...


        # Fitting exponential and finding ...

        self.feature_names=feature_names
        self.features = feature_list
