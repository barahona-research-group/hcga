from networkx.algorithms import centrality
from hcga.Operations import DistributionFitting
import numpy as np

def DegreeCentrality(G,bins):
    """
    Input: Graph networkx

    Output: Dictionary of heuristic graph stats

    Stats:



    """
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
    feature_list.append(DistributionFitting.power_law_fit(degree_centrality,bins=bins)[0][0][-2]) # value 'a' in power law
    feature_list.append(DistributionFitting.power_law_fit(degree_centrality,bins=bins)[0][1]) # value sse in power law

    # Fitting normal distribution and finding...


    # Fitting exponential and finding ...


    return (feature_names,feature_list)
