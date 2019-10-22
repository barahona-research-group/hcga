

import numpy as np
from hcga.Operations.utils import clustering_quality
import networkx as nx

from networkx.algorithms.community import label_propagation_communities

class LabelpropagationCommunities():
    """
    Label propagation communities class
    """
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = {}

    def feature_extraction(self):

        """Compute the measures based on the Label propagation communities algorithm.

        Parameters
        ----------
        G : graph
           A networkx graph

        Returns
        -------
        feature_list : dict
           Dictionary of features related to the label propagation communities algorithm.


        Notes
        -----
        Implementation of networkx code:
            `Networkx_label_propagation <https://networkx.github.io/documentation/networkx-2.2/_modules/networkx/algorithms/community/label_propagation.html#label_propagation_communities>`_
    

        
        Finds communities in `G` using a semi-synchronous label propagation
    method[1]_. This method combines the advantages of both the synchronous
    and asynchronous models. Not implemented for directed graphs.

        References
        ----------
        .. [1] Cordasco, G., & Gargano, L. (2010, December). Community detection
           via semi-synchronous label propagation algorithms. In Business
           Applications of Social Network Analysis (BASNA), 2010 IEEE International
           Workshop on (pp. 1-8). IEEE.

        """


        G = self.G

        feature_list = {}
        
        if not nx.is_directed(G):
    
            c = list(label_propagation_communities(G))              
    
            
            # calculate ratio of the two communities
            if len(c)>1:
                feature_list['node_ratio']=(len(c[0])/len(c[1]))
            else:
                feature_list['node_ratio']=0
            
            # clustering quality functions       
            qual_names,qual_vals = clustering_quality(G,c)  
            
            for i in range(len(qual_names)):
                feature_list[qual_names[i]]=qual_vals[i]         
    

        else:
            feature_list['node_ratio']=np.nan
            feature_list['node_ratio']=np.nan
            feature_list['mod']=np.nan
            feature_list['coverage']=np.nan
            feature_list['performance']=np.nan
            feature_list['inter_comm_edge']=np.nan
            feature_list['inter_comm_nedge']=np.nan
            feature_list['intra_comm_edge']=np.nan
            
            

        self.features = feature_list
