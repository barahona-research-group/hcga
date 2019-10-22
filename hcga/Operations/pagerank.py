

import networkx as nx
from hcga.Operations import utils
import numpy as np

class PageRank():
    """
    Page rank class    
    """
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = {}

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

        Returns
        -------
        feature_list :list
           List of features related to PageRank.


        Notes
        -----
        Implementation of networkx code:
            `Networkx_pagerank <https://networkx.github.io/documentation/latest/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html>`_        

        PageRank computes a ranking of the nodes in the graph G based on
        the structure of the incoming links. It was originally designed as
        an algorithm to rank web pages.

        References
        ----------
        .. [1] A. Langville and C. Meyer,
           "A survey of eigenvector methods of web information retrieval."
           http://citeseer.ist.psu.edu/713792.html
        .. [2] Page, Lawrence; Brin, Sergey; Motwani, Rajeev and Winograd, Terry,
           The PageRank citation ranking: Bringing order to the Web. 1999
           http://dbpubs.stanford.edu:8090/pub/showDoc.Fulltext?lang=en&doc=1999-66&format=pdf
    

        """
        bins = [10,20,50]
        

        G = self.G

        feature_list = {}
        
        #Calculate PageRank
        pagerank = np.asarray(list(nx.pagerank(G).values()))
        # Basic stats regarding the PageRank distribution
        feature_list['mean'] = pagerank.mean()
        feature_list['std'] = pagerank.std()
        feature_list['max'] = pagerank.max()
        feature_list['min'] = pagerank.min()
            
        for i in range(len(bins)):

                
            # Fitting the PageRank distribution and finding the optimal
            # distribution according to SSE
            opt_mod,opt_mod_sse = utils.best_fit_distribution(pagerank,bins=bins[i])
            feature_list['opt_model_{}'.format(bins[i])] = opt_mod
    
            # Fitting power law and finding 'a' and the SSE of fit.
            feature_list['powerlaw_a_{}'.format(bins[i])] = utils.power_law_fit(pagerank,bins=bins[i])[0][-2]# value 'a' in power law
            feature_list['powerlaw_SSE_{}'.format(bins[i])] = utils.power_law_fit(pagerank,bins=bins[i])[1] # value sse in power law
        
        self.features = feature_list

