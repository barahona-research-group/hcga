
import pandas as pd
import numpy as np
import networkx as nx

class RichClub():
    """
    Rich club class    
    """
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = {}

    def feature_extraction(self):

        """Returns the rich-club coefficient of the graph `G`.
    
        For each degree *k*, the *rich-club coefficient* is the ratio of the
        number of actual to the number of potential edges for nodes with
        degree greater than *k*:
    
        .. math::
    
            \phi(k) = \frac{2 E_k}{N_k (N_k - 1)}
    
        where `N_k` is the number of nodes with degree larger than *k*, and
        `E_k` is the number of edges among those nodes.
        
        Parameters
        ----------
        G : graph
           A networkx graph

        Returns
        -------
        feature_list : dict
           Dictionary of features related to the rich club coefficients.

        Notes
        -----
        Rich club calculated using networkx:
            `Networkx_centrality <https://networkx.github.io/documentation/stable/reference/algorithms/rich_club.html>`_
        
        The rich club definition and algorithm are found in [1]_.  This
        algorithm ignores any edge weights and is not defined for directed
        graphs or graphs with parallel edges or self loops.
    
        Estimates for appropriate values of `Q` are found in [2]_.
    
        References
        ----------
        .. [1] Julian J. McAuley, Luciano da Fontoura Costa,
           and Tibério S. Caetano,
           "The rich-club phenomenon across complex network hierarchies",
           Applied Physics Letters Vol 91 Issue 8, August 2007.
           https://arxiv.org/abs/physics/0701290
        .. [2] R. Milo, N. Kashtan, S. Itzkovitz, M. E. J. Newman, U. Alon,
           "Uniform generation of random graphs with arbitrary degree
           sequences", 2006. https://arxiv.org/abs/cond-mat/0312028

        """
        

        

        G = self.G

        feature_list = {}
        if not nx.is_directed(G) and len(G)>5:            
            # Calculating the shortest paths stats
            for attempts in range(10):
                try:
                    
                    rich_club = list(nx.rich_club_coefficient(G).values())       
                    
                    # calculate number of nodes that qualify according to degree k
                    feature_list['num_rich']=len(rich_club)
                    
                    feature_list['mean_rich_coef']=np.mean(rich_club)
                    feature_list['std_rich_coef']=np.std(rich_club)
                    feature_list['max_rich_coef']=np.max(rich_club)      
                    
                    feature_list['ratio_rich_coef']=np.min(rich_club)/np.max(rich_club)
                    if rich_club[-2]>0:
                        feature_list['ratio_top2_coef']=rich_club[-1]/rich_club[-2]
                    else:
                        feature_list['ratio_top2_coef']=0
            
                    # top ten degree rich club coefficients
                    if len(rich_club)>=10:
                        top10 = rich_club[-10:]
                    else:
                        top10 = [1] * (10-len(rich_club)) + rich_club
                        
                    """
                    feature_list = feature_list + top10
                    """
                    
                    l=[10,9,8,7,6,5,4,3,2,1]
                    for i in range(len(top10)):
                        feature_list['top10_{}'.format(l[i])]=top10[i]
                    
                    break
                
                except Exception as e:
                    print('Exception for rich_club:', e)
    
                    feature_names = ['num_rich','mean_rich_coef','std_rich_coef','max_rich_coef','ratio_rich_coef','ratio_top2_coef',
                                      'top10_10','top10_9','top10_8','top10_7','top10_6','top10_5','top10_4','top10_3',
                                      'top10_2','top10_1']
                    for j in range(len(feature_names)):
                        feature_list[feature_names[j]]=0
        else:
            feature_names = ['num_rich','mean_rich_coef','std_rich_coef','max_rich_coef','ratio_rich_coef','ratio_top2_coef',
                                  'top10_10','top10_9','top10_8','top10_7','top10_6','top10_5','top10_4','top10_3',
                                  'top10_2','top10_1']
            for j in range(len(feature_names)):
                    feature_list[feature_names[j]]=np.nan
            
            

        self.features = feature_list
