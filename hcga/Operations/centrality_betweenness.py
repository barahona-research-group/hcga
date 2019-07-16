
from networkx.algorithms import centrality
from hcga.Operations import utils
import numpy as np



class BetweennessCentrality():
    """
    Betweenness centrality class
    """
 
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = []

    def feature_extraction(self,args):
        """Compute the betweenness centrality for nodes.
        """        
                
        # Defining the input arguments
        bins = args[0]

        # Defining featurenames
        feature_names = ['mean','std','opt_model','powerlaw_a','powerlaw_SSE']

        G = self.G

        feature_list = []

        #Calculate the betweenness centrality of each node
        betweenness_centrality = np.asarray(list(centrality.betweenness_centrality(G).values()))

        # Basic stats regarding the betwenness centrality distribution
        feature_list.append(betweenness_centrality.mean())
        feature_list.append(betweenness_centrality.std())

        # Fitting the betweenness centrality distribution and finding the optimal
        # distribution according to SSE
        opt_mod,opt_mod_sse = utils.best_fit_distribution(betweenness_centrality,bins=bins)
        feature_list.append(opt_mod)

        # Fitting power law and finding 'a' and the SSE of fit.
        feature_list.append(utils.power_law_fit(betweenness_centrality,bins=bins)[0][-2])# value 'a' in power law
        feature_list.append(utils.power_law_fit(betweenness_centrality,bins=bins)[1])# value sse in power law

        # Fitting normal distribution and finding...


        # Fitting exponential and finding ...

        self.feature_names=feature_names
        self.features = feature_list
