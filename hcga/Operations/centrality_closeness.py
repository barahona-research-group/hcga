
from networkx.algorithms import centrality
from hcga.Operations import utils
import numpy as np



class ClosenessCentrality():
    """
    Closeness centrality class
    """
 
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = []

    def feature_extraction(self,args):
        """Compute the closeness centrality for nodes.
        """        
                
        # Defining the input arguments
        bins = args[0]

        # Defining featurenames
        feature_names = ['mean','std','opt_model','powerlaw_a','powerlaw_SSE']

        G = self.G

        feature_list = []

        #Calculate the closeness centrality of each node
        closeness_centrality = np.asarray(list(centrality.closeness_centrality(G).values()))

        # Basic stats regarding the closeness centrality distribution
        feature_list.append(closeness_centrality.mean())
        feature_list.append(closeness_centrality.std())

        # Fitting the closeness centrality distribution and finding the optimal
        # distribution according to SSE
        opt_mod,opt_mod_sse = utils.best_fit_distribution(closeness_centrality,bins=bins)
        feature_list.append(opt_mod)

        # Fitting power law and finding 'a' and the SSE of fit.
        feature_list.append(utils.power_law_fit(closeness_centrality,bins=bins)[0][-2])# value 'a' in power law
        feature_list.append(utils.power_law_fit(closeness_centrality,bins=bins)[1])# value sse in power law

        # Fitting normal distribution and finding...


        # Fitting exponential and finding ...

        self.feature_names=feature_names
        self.features = feature_list
