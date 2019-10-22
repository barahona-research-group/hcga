
import numpy as np
import networkx as nx

from hcga.Operations import utils

class NodeConnectivity():
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = {}

    def feature_extraction(self):

        """Compute node connectivity measures.

        Node connectivity is equal to the minimum number of nodes that
            must be removed to disconnect G or render it trivial. If source
            and target nodes are provided, this function returns the local node
            connectivity: the minimum number of nodes that must be removed to break
            all paths from source to target in G.

        Parameters
        ----------
        G : graph
           A networkx graph

        Returns
        -------
        feature_list :list
           List of features related to node connectivity.


        Notes
        -----
        Implementation of networkx code:
            `Networkx_node_connectivity <https://networkx.github.io/documentation/stable/_modules/networkx/algorithms/connectivity/connectivity.html#node_connectivity>`_

        This is a flow based implementation of node connectivity. The
        algorithm works by solving $O((n-\delta-1+\delta(\delta-1)/2))$
        maximum flow problems on an auxiliary digraph. Where $\delta$
        is the minimum degree of G. For details about the auxiliary
        digraph and the computation of local node connectivity see
        :meth:`local_node_connectivity`. This implementation is based
        on algorithm 11 in [1]_.
        
        References
        ----------
        .. [1] White, Douglas R., and Mark Newman. 2001 A Fast Algorithm for 
        Node-Independent Paths. Santa Fe Institute Working Paper #01-07-035
        http://eclectic.ss.uci.edu/~drwhite/working.pdf

        """
        

        # Defining the input arguments
        bins=[10]
        
        G = self.G

        feature_list = {}

        # calculating node connectivity
        node_connectivity = nx.all_pairs_node_connectivity(G)
        N = G.number_of_nodes()
        
        node_conn = np.zeros([N,N])
        for key1, value1 in node_connectivity.items():    
            for key2, value2 in value1.items():
                node_conn[key1,key2] = value2

        # mean and median minimum number of nodes to remove connectivity
        feature_list = utils.summary_statistics(feature_list,list(np.triu(node_conn).flatten()),'node_conn')

        for i in range(len(bins)):
            
            # fitting the node connectivity histogram distribution
            opt_mod_mean,_ =  utils.best_fit_distribution(node_conn.mean(axis=1),bins=bins[i])
            feature_list['opt_model_mean_{}'.format(bins[i])]=opt_mod_mean
        
            # fitting the node connectivity histogram distribution
            opt_mod_std,_ =  utils.best_fit_distribution(node_conn.std(axis=1),bins=bins[i])
            feature_list['opt_model_std{}'.format(bins[i])]=opt_mod_std        
        
            # fitting the node connectivity histogram distribution
            opt_mod_max,_ =  utils.best_fit_distribution(node_conn.max(axis=1),bins=bins[i])
            feature_list['opt_model_max{}'.format(bins[i])]=opt_mod_max              
        
        # Calculate connectivity
        feature_list['node_connectivity']=nx.node_connectivity(G)
        feature_list['average_node_connectivity']=nx.average_node_connectivity(G)
        feature_list['edge_connectivity']=nx.edge_connectivity(G)
        
        # calculate the wiener index 
        feature_list['wiener_index']=nx.wiener_index(G)
        

        self.features = feature_list
