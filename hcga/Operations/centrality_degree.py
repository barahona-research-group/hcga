from networkx.algorithms import centrality
from hcga.Operations import utils
import numpy as np



class DegreeCentrality():
    def __init__(self, G):
        self.G = G
        self.feature_names = []
        self.features = []

    def feature_extraction(self,bins=10):
        """
        Input: bins - the number of bins for the calculating the SSE of fits. Default = 10 bins

        """

        self.feature_names = ['mean','std','opt_model','powerlaw_a','powerlaw_SSE']

        G = self.G

        feature_list = []

        #Calculate the degree centrality of each node
        degree_centrality = np.asarray(list(centrality.degree_centrality(G).values()))

        # Basic stats regarding the degree centrality distribution
        feature_list.append(degree_centrality.mean())
        feature_list.append(degree_centrality.std())

        # Fitting the degree centrality distribution and finding the optimal
        # distribution according to SSE
        opt_mod,opt_mod_sse = utils.best_fit_distribution(degree_centrality,bins=bins)
        feature_list.append(opt_mod)

        # Fitting power law and finding 'a' and the SSE of fit.
        feature_list.append(utils.power_law_fit(degree_centrality,bins=bins)[0][-2]) # value 'a' in power law
        feature_list.append(utils.power_law_fit(degree_centrality,bins=bins)[1]) # value sse in power law

        # Fitting normal distribution and finding...


        # Fitting exponential and finding ...


        self.features = feature_list


"""
def DegreeCentrality(G,bins):

    feature_names = ['mean','std','opt_model','powerlaw_a','powerlaw_SSE']
    feature_list = []

    degree_centrality = np.asarray(list(centrality.degree_centrality(G).values()))
    #degree_centrality_in = centrality.in_degree_centrality(G)
    #degree_centrality_out = centrality.out_degree_centrality(G)

    # Basic stats regarding the degree centrality distribution
    feature_list.append(degree_centrality.mean())
    feature_list.append(degree_centrality.std())

    # Fitting the degree centrality distribution and finding the optimal
    # distribution according to SSE
    opt_mod,opt_mod_sse = DistributionFitting.best_fit_distribution(degree_centrality,bins=bins)
    feature_list.append(opt_mod)

    # Fitting power law and finding 'a' and the SSE of fit.
    feature_list.append(DistributionFitting.power_law_fit(degree_centrality,bins=bins)[0][-2]) # value 'a' in power law
    feature_list.append(DistributionFitting.power_law_fit(degree_centrality,bins=bins)[1]) # value sse in power law

    # Fitting normal distribution and finding...


    # Fitting exponential and finding ...


    return (feature_names,feature_list)
"""
