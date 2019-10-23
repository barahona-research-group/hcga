
import networkx as nx
from hcga.Operations import utils
import numpy as np

class Hits():
    """
    HITS hubs class
    """
    
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = {}

    def feature_extraction(self):

        """
        Computes measures based on the HITS hubs 
        
        The HITS algorithm computes two numbers for a node [1]_ [2]_.
        Authorities estimates the node value based on the incoming links.
        Hubs estimates the node value based on outgoing links.

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
                Hits calculations using networkx:
            `Networkx_hits <https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.hits_alg.hits.html#networkx.algorithms.link_analysis.hits_alg.hits>`_
        
 
        The eigenvector calculation is done by the power iteration method
        and has no guarantee of convergence.  The iteration will stop
        after max_iter iterations or an error tolerance of
        number_of_nodes(G)*tol has been reached.
    
        The HITS algorithm was designed for directed graphs but this
        algorithm does not check if the input graph is directed and will
        execute on undirected graphs.
    
        References
        ----------
        .. [1] A. Langville and C. Meyer,
           "A survey of eigenvector methods of web information retrieval."
           http://citeseer.ist.psu.edu/713792.html
        .. [2] Jon Kleinberg,
           Authoritative sources in a hyperlinked environment
           Journal of the ACM 46 (5): 604-32, 1999.
           doi:10.1145/324133.324140.
           http://www.cs.cornell.edu/home/kleinber/auth.pdf.

        """
        bins = [10,20,50]
        


        G = self.G

        feature_list = {}
        
        try:
            # if undirected h and a are the same
            h,a=nx.hits(G,max_iter=1000)
            h = np.asarray(list(h.values()))
        except Exception as e:
            print('Exception for link_analysis_hit', e)
            h = np.array([1,1,1]) # random filling array
        
        
        
        # Basic stats regarding the PageRank distribution
        feature_list['mean'] = h.mean()
        feature_list['std'] = h.std()
        feature_list['max'] = h.max()
        feature_list['min'] = h.min()
        feature_list['ratio'] = h.min()/h.max()

            
        for i in range(len(bins)):

            # distribution according to SSE
            opt_mod,opt_mod_sse = utils.best_fit_distribution(h,bins=bins[i])
            feature_list['opt_model_{}'.format(bins[i])] = opt_mod
    
            # Fitting power law and finding 'a' and the SSE of fit.
            feature_list['powerlaw_a_{}'.format(bins[i])] = utils.power_law_fit(h,bins=bins[i])[0][-2]# value 'a' in power law
            feature_list['powerlaw_SSE_{}'.format(bins[i])] = utils.power_law_fit(h,bins=bins[i])[1] # value sse in power law
        
        self.features = feature_list

