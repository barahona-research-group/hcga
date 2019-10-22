

import networkx as nx

class ScaleFree():
    """
    Scale free class
    
    """
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = {}

    def feature_extraction(self):

        """ Calculating metrics about scale-free networks



        Parameters
        ----------
        G : graph
           A networkx graph

        Returns
        -------
        feature_list : dict
           Dictionary of features related to scale freeness.


        Notes
        -----
                Scale free calculations using networkx:
            `Networkx_scale free <https://networkx.github.io/documentation/stable/_modules/networkx/algorithms/smetric.html#s_metric>`_
        
        The s-metric is defined as the sum of the products deg(u)*deg(v)
        for every edge (u,v) in G. If norm is provided construct the
        s-max graph and compute it's s_metric, and return the normalized
        s value      
        
        
        References
        ----------
        .. [1] Lun Li, David Alderson, John C. Doyle, and Walter Willinger,
               Towards a Theory of Scale-Free Graphs:
               Definition, Properties, and  Implications (Extended Version), 2005.
               https://arxiv.org/abs/cond-mat/0501169

        """

        G = self.G

        feature_list = {}

              

        # 
        feature_list['s_metric']=nx.s_metric(G,normalized=False)        
        

        self.features = feature_list
