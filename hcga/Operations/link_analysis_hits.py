

import networkx as nx
from hcga.Operations import utils
import numpy as np

class Hits():
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = {}

    def feature_extraction(self):

        """
 

        Parameters
        ----------
        G : graph
           A networkx graph


        Returns
        -------
        feature_list :list
           List of features related to 


        Notes
        -----
        Implementation of networkx code:

        """
        bins = [10,20,50]
        


        G = self.G

        feature_list = {}
        
        #Calculate PageRank
        h,a=nx.hits(G,max_iter=1000)
        h = np.asarray(list(h.values()))
        
        # if undirected h and a are the same
        
        
        # Basic stats regarding the PageRank distribution
        feature_list['mean'] = h.mean()
        feature_list['std'] = h.std()
        feature_list['max'] = h.max()
        feature_list['min'] = h.min()
        feature_list['ratio'] = h.min()/h.max()

            
        for i in range(len(bins)):
            """# Adding to feature names
            feature_names.append('opt_model_{}'.format(bins[i]))
            feature_names.append('powerlaw_a_{}'.format(bins[i]))
            feature_names.append('powerlaw_SSE_{}'.format(bins[i]))"""
                
            # Fitting the PageRank distribution and finding the optimal
            # distribution according to SSE
            opt_mod,opt_mod_sse = utils.best_fit_distribution(h,bins=bins[i])
            feature_list['opt_model_{}'.format(bins[i])] = opt_mod
    
            # Fitting power law and finding 'a' and the SSE of fit.
            feature_list['powerlaw_a_{}'.format(bins[i])] = utils.power_law_fit(h,bins=bins[i])[0][-2]# value 'a' in power law
            feature_list['powerlaw_SSE_{}'.format(bins[i])] = utils.power_law_fit(h,bins=bins[i])[1] # value sse in power law
        
        self.features = feature_list

