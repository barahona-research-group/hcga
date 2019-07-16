
from networkx.algorithms import centrality
from hcga.Operations import utils
import numpy as np



class EdgeBetweennessCentrality():
    """
    Edge betweenness centrality class
    """
 
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = []

    def feature_extraction(self,args):
        """
        Compute the edge betweenness centrality for nodes.
        
        The edge betweeness centrality for an edge is the number of shortest
        paths that pass through it.
        Here we calculate the fraction of shortest paths that pass through
        each edge.
        
        Parameters
        ----------
        G : graph
          A networkx graph

        args :
            arg[0] Number of bins for calculating pdf of chosen distribution
            for SSE calculation

        Returns
        -------
        feature_list :list
           List of features related to betweenness centrality.
           
         Notes
        -----
        Betweenness centrality calculations using networkx:
            https://networkx.github.io/documentation/stable/reference/algorithms/centrality.html
        """     
                
        # Defining the input arguments
        bins = args[0]

        # Defining featurenames
        feature_names = ['mean','std','opt_model','powerlaw_a','powerlaw_SSE']

        G = self.G

        feature_list = []

        #Calculate the edge betweenness centrality of each node
        edge_betweenness_centrality = np.asarray(list(centrality.edge_betweenness_centrality(G).values()))

        # Basic stats regarding the edge betweenness centrality distribution
        feature_list.append(edge_betweenness_centrality.mean())
        feature_list.append(edge_betweenness_centrality.std())

        # Fitting the edge betweenness centrality distribution and finding the optimal
        # distribution according to SSE
        opt_mod,opt_mod_sse = utils.best_fit_distribution(edge_betweenness_centrality,bins=bins)
        feature_list.append(opt_mod)

        # Fitting power law and finding 'a' and the SSE of fit.
        feature_list.append(utils.power_law_fit(edge_betweenness_centrality,bins=bins)[0][-2])# value 'a' in power law
        feature_list.append(utils.power_law_fit(edge_betweenness_centrality,bins=bins)[1])# value sse in power law

        # Fitting normal distribution and finding...


        # Fitting exponential and finding ...

        self.feature_names=feature_names
        self.features = feature_list