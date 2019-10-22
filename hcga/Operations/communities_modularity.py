import numpy as np
from hcga.Operations.utils import clustering_quality
import networkx as nx

from networkx.algorithms.community import greedy_modularity_communities

class ModularityCommunities():
    """
    Modularity communities class
    """
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = {}

    def feature_extraction(self):

        """
        
        Find communities in graph using Clauset-Newman-Moore greedy modularity
        maximization [1]_ [2]_. This method currently supports the Graph class and does not
        consider edge weights.

        Greedy modularity maximization begins with each node in its own community
        and joins the pair of communities that most increases modularity until no
        such pair exists.

        Parameters
        ----------
        G : NetworkX graph

        Returns
        -------
        feature_list:  dict
            Dictionary of features related to modularity

        Notes
        -----
        Greedy modularity implemented using networkx:
            `Networkx_communities <https://networkx.github.io/documentation/stable/_modules/networkx/algorithms/community/modularity_max.html#greedy_modularity_communities>`_

        References
        ----------
        .. [1] M. E. J Newman 'Networks: An Introduction', page 224
           Oxford University Press 2011.
        .. [2] Clauset, A., Newman, M. E., & Moore, C.
           "Finding community structure in very large networks."
           Physical Review E 70(6), 2004.
        """
        
        
        G = self.G

        feature_list = {}
        
        if not nx.is_directed(G):

    
            # The optimised number of communities using greedy modularity
            c = list(greedy_modularity_communities(G))
            
            # calculate number of communities
            feature_list['num_comms_greedy_mod']=len(c)  
        
            # calculate ratio of largest to smallest community
            feature_list['ratio_max_min_num_nodes']=(len(c[0])/len(c[-1]))      
        
            # calculate ratio of largest to 2nd largest community
            if len(c)>1:
                feature_list['ratio_max_2max_num_nodes']=(len(c[0])/len(c[1]))
            else:
                feature_list['ratio_max_2max_num_nodes']=np.nan
        
            # clustering quality functions       
            qual_names,qual_vals = clustering_quality(G,c)      
                
            for i in range(len(qual_names)):
                feature_list[qual_names[i]]=qual_vals[i]   
        else:
            feature_list['num_comms_greedy_mod'] =np.nan
            feature_list['ratio_max_min_num_nodes'] =np.nan
            feature_list['ratio_max_2max_num_nodes']=np.nan
            feature_list['node_ratio']=np.nan
            feature_list['mod']=np.nan
            feature_list['coverage']=np.nan
            feature_list['performance']=np.nan
            feature_list['inter_comm_edge']=np.nan
            feature_list['inter_comm_nedge']=np.nan
            feature_list['intra_comm_edge']=np.nan


        self.features = feature_list
