
import scipy.stats as st
import statsmodels as sm
import warnings
import numpy as np
from networkx.algorithms.community import quality

# Identifying optimal model with Sum of square error (SSE)

def power_law_fit(data,bins=10):
    y, x = np.histogram(data, bins=bins)
    x = (x + np.roll(x, -1))[:-1] / 2.0
    try:
        # Ignore warnings from data that can't be fit
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            params = st.powerlaw.fit(data)

            # Separate parts of parameters
            arg = params[:-2]
            loc = params[-2]
            scale = params[-1]

            # Calculate fitted PDF and error with fit in distribution
            pdf = st.powerlaw.pdf(x, loc=loc, scale=scale, *arg)
            sse = np.sum(np.power(y - pdf, 2.0))
    except Exception:
        pass

    return (params, sse)





def best_fit_distribution(data, bins=10, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check

    #DISTRIBUTIONS = [st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,
        #st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,
        #st.foldcauchy,st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.gennorm,st.genexpon,
        #st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,
        #st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
        #st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,st.laplace,st.levy,st.levy_l,st.levy_stable,
        #st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,
        #st.nct,st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,
        #st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,
        #st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy
    #]

    DISTRIBUTIONS = [st.powerlaw, st.expon, st.norm, st.lognorm, st.beta]

    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS:

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                    end
                except Exception:
                    pass

                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse

        except Exception:
            pass

    dist_ID = DISTRIBUTIONS.index(best_distribution)
    # we could 1 hot encode this?

    return (dist_ID, best_params)




def clustering_quality(G,c):
    """Method for calculating the quality of parition"""

    quality_names = ['mod','coverage','performance','inter_comm_edge','inter_comm_nedge','intra_comm_edge']
    quality_values = []

    quality_values.append(quality.modularity(G,c))
    quality_values.append(quality.coverage(G,c))
    quality_values.append(quality.performance(G,c))
    quality_values.append(quality.inter_community_edges(G,c))
    quality_values.append(quality.inter_community_non_edges(G,c))
    quality_values.append(quality.intra_community_edges(G,c))

    return quality_names,quality_values



def summary_statistics(feature_list,dist,feat_name):
    
    """ Computes summary statistics of distribution """
        
    feature_list[feat_name + '_mean'] = np.mean(dist)
    feature_list[feat_name + '_min'] = np.min(dist)
    feature_list[feat_name + '_max'] = np.max(dist)
    feature_list[feat_name + '_median'] = np.median(dist)
    feature_list[feat_name + '_std'] = np.std(dist)
    feature_list[feat_name + '_gmean'] = st.gmean(dist)
    
    try:
        feature_list[feat_name + '_hmean'] = st.hmean(np.abs(dist)+1e-8)

    except Exception as e:
        print('Exception for utils:', e)

        feature_list[feat_name + '_hmean'] = np.nan
        
    feature_list[feat_name + '_kurtosis'] = st.kurtosis(dist)
    feature_list[feat_name + '_mode'] = st.mode(dist)[0][0]
    feature_list[feat_name + '_kstat'] = st.kstat(dist)
    feature_list[feat_name + '_kstatvar'] = st.kstatvar(dist)
    feature_list[feat_name + '_tmean'] = st.tmean(dist)   
    feature_list[feat_name + '_tvar'] = st.tvar(dist)   
    feature_list[feat_name + '_tmin'] = st.tmin(dist)   
    feature_list[feat_name + '_tmax'] = st.tmax(dist)   
    feature_list[feat_name + '_tstd'] = st.tstd(dist)   
    feature_list[feat_name + '_tsem'] = st.tsem(dist)   
    feature_list[feat_name + '_variation'] = st.variation(dist)   
    feature_list[feat_name + '_mean_repeats'] = np.mean(st.find_repeats(dist)[1])
    #feature_list[feat_name + '_max_repeat'] = np.max(st.find_repeats(dist)[0])
    feature_list[feat_name + '_entropy'] = st.entropy(dist)   
    feature_list[feat_name + '_sem'] = st.sem(dist)
    try:
        feature_list[feat_name + '_bayes_confint'] = st.bayes_mvs(dist)[0][1][1] - st.bayes_mvs(dist)[0][1][0]

    except Exception as e:
        print('Exception for utils, bayes_confing:', e)
        feature_list[feat_name + '_bayes_confint'] = np.nan 

    return feature_list
