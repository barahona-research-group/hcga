from networkx.algorithms import centrality
from hcga.Operations import DistributionFitting

def DegreeCentrality(G):
    """
    Input: Graph networkx

    Output: Dictionary of heuristic graph stats

    Stats:



    """

    degree_centrality_dict = {'degree':None}


    # basic normalisation parameters
    N = G.number_of_nodes()
    E = G.number_of_edges()

    degree_centrality = np.asarray(list(centrality.degree_centrality(G).values()))
    #degree_centrality_in = centrality.in_degree_centrality(G)
    #degree_centrality_out = centrality.out_degree_centrality(G)

    # Basic stats regarding the degree centrality distribution
    degree_centrality_dict['degree_centrality_mean'] = degree_centrality.mean()
    degree_centrality_dict['degree_centrality_std'] = degree_centrality.std()


    # Fitting the degree centrality distribution and finding the optimal
    # distribution according to SSE

    # Looping over bin sizes for distribution.
    bin_sizes = [5,25,200]
    opt_mod = []
    for bins in bin_sizes:
        opt_mod.append(DistributionFitting.best_fit_distribution(degree_centrality,bins=bins))

    degree_centrality_dict['degree_centrality_opt_mod_5'] = opt_mod[0][0]
    degree_centrality_dict['degree_centrality_opt_mod_25'] = opt_mod[1][0]
    degree_centrality_dict['degree_centrality_opt_mod_200'] = opt_mod[2][0]

    # Looping over bin sizes for fitting power law distribution and storing the powers
    bin_sizes = [5,25,200]
    powerlaw_model = []
    for bins in bin_sizes:
        opt_mod.append(DistributionFitting.power_law_fit(degree_centrality,bins))

    degree_centrality_dict['degree_centrality_opt_mod_5_params'] = opt_mod[0][0]
    degree_centrality_dict['degree_centrality_opt_mod_25_'] = opt_mod[1][0]
    degree_centrality_dict['degree_centrality_opt_mod_200'] = opt_mod[2][0]




    return heuristics_dict
