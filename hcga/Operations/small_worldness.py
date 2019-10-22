
import networkx as nx

class SmallWorld():
    """
    Small world class
    """
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = {}

    def feature_extraction(self):

        """ 
    The small-world coefficient of a graph G is:
    
        omega = Lr/L - C/Cl
    
        where C and L are respectively the average clustering coefficient and
        average shortest path length of G. Lr is the average shortest path length
        of an equivalent random graph and Cl is the average clustering coefficient
        of an equivalent lattice graph.
    
        The small-world coefficient (omega) ranges between -1 and 1. Values close
        to 0 means the G features small-world characteristics. Values close to -1
        means G has a lattice shape whereas values close to 1 means G is a random
        graph.
        
        Parameters
        ----------
        G : graph
           A networkx graph

        Returns
        -------
        feature_list : dict
           Dictionary of features related to small worldness


        Notes
        -----
        Small world calculations using networkx:
            `Networkx_omega <https://networkx.github.io/documentation/latest/_modules/networkx/algorithms/smallworld.html#omega>`_
        
        The implementation is adapted from the algorithm by Telesford et al. [1]_.
    
        References
        ----------
        .. [1] Telesford, Joyce, Hayasaka, Burdette, and Laurienti (2011).
               "The Ubiquity of Small-World Networks".
               Brain Connectivity. 1 (0038): 367-75.  PMC 3604768. PMID 22432451.
               doi:10.1089/brain.2011.0038.

        """


        G = self.G

        feature_list = {}

        #C = nx.transitivity(G)
        #L = nx.average_shortest_path_length(G)
        
        # sigma requires recalculating the average clustering effiient 
        # and the average shortest path length. These can be reused from
        # other features.

        #feature_list.append(nx.sigma(G))
        #feature_list.append(nx.omega(G))
        # small worldness sigma
#        nrand=20
#        
#        randMetrics = {"C": [], "L": []}
#        for i in range(nrand):
#            Gr = nx.dense_gnm_random_graph(N,E)
#            if nx.is_connected(Gr):
#                randMetrics["C"].append(nx.transitivity(Gr))
#                randMetrics["L"].append(nx.average_shortest_path_length(Gr))
#            
#        Cr = np.mean(randMetrics["C"])
#        Lr = np.mean(randMetrics["L"])
#        
#        sigma = (C / Cr) / (L / Lr)
        
        #feature_list['sigma']=nx.sigma(G)


        #omega = 0
        feature_list['omega']=nx.omega(G)

        self.features = feature_list
