
import numpy as np
from hcga.Operations.utils import clustering_quality
import networkx as nx

from networkx.algorithms.community import kernighan_lin_bisection

class BisectionCommunities():
    """
    Bisection communities class
    """
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = {}

    def feature_extraction(self):
        
        """Compute the measures based on the Kernighan–Lin Bisection communities algorithm.

        Parameters
        ----------
        G : graph
           A networkx graph

        Returns
        -------
        feature_list : dict
           Dictionary of features related to the Kernighan–Lin communities algorithm.


        Notes
        -----
        Implementation of networkx code:
            `Networkx_kernighan_lin_bisection <https://networkx.github.io/documentation/latest/reference/algorithms/generated/networkx.algorithms.community.kernighan_lin.kernighan_lin_bisection.html>`_
    
        Partition a graph into two blocks using the Kernighan–Lin algorithm described in
        [1]_.
        
        This algorithm paritions a network into two sets by iteratively swapping pairs of nodes to reduce the edge cut between the two sets.

        References
        ----------
        .. [1] Kernighan, B. W.; Lin, Shen (1970).
           "An efficient heuristic procedure for partitioning graphs."
           *Bell Systems Technical Journal* 49: 291--307.
           Oxford University Press 2011.

        """

        G = self.G

        feature_list = {}
        
        if not nx.is_directed(G):


            c = list(kernighan_lin_bisection(G))        
        

        
            # calculate ratio of the two communities
            feature_list['node_ratio']=(len(c[0])/len(c[1]))
        
            # clustering quality functions       
            qual_names,qual_vals = clustering_quality(G,c)           

            for i in range(len(qual_names)):
                feature_list[qual_names[i]]=qual_vals[i]
            

        else:
            feature_list['node_ratio']=np.nan
            feature_list['mod']=np.nan
            feature_list['coverage']=np.nan
            feature_list['performance']=np.nan
            feature_list['inter_comm_edge']=np.nan
            feature_list['inter_comm_nedge']=np.nan
            feature_list['intra_comm_edge']=np.nan


        self.features = feature_list
