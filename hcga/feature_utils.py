"""utils functions for feature extractions"""
import scipy.stats as st
import numpy as np

#np.seterr(all='raise')

def normalize_features(feature_instance):
    """triple the number of features by normalising by node and edges"""

    interpretability_downgrade = 1
    for feature_name in list(feature_instance.features.keys()):
        feat_interpret = feature_instance.get_feature_interpretability(feature_name)
        feat_desc = feature_instance.get_feature_description(feature_name)
        feature_instance.add_feature(
                feature_name + "_N", 
                feature_instance.features[feature_name] / len(feature_instance.graph.nodes),
                feat_desc + ", normalised by the number of nodes in the graph", 
                feat_interpret - interpretability_downgrade,
                )
        feature_instance.add_feature(
                feature_name + "_E", 
                feature_instance.features[feature_name] / len(feature_instance.graph.edges),
                feat_desc + ", normalised by the number of edges in the graph",
                feat_interpret - interpretability_downgrade,
                )

def summary_statistics(add_feature, feat_dist, feat_name, feat_desc, feat_interpret):
    """Computes summary statistics of distributions"""
    compl_desc = " of the distribution of " + feat_desc

    add_feature(feat_name + "_mean", np.mean(feat_dist), 
            "Mean" + compl_desc, feat_interpret - 0)
    add_feature(feat_name + "_min", np.min(feat_dist), 
            "Minimum" + compl_desc, feat_interpret - 0)
    add_feature(feat_name + "_max", np.max(feat_dist), 
            "Maximum" + compl_desc, feat_interpret - 0)
    add_feature(feat_name + "_median", np.median(feat_dist), 
            "Median" + compl_desc, feat_interpret - 0)
    add_feature(feat_name + "_std", np.std(feat_dist), 
            "Standard deviation" + compl_desc, feat_interpret - 0)
    add_feature(feat_name + "_gmean", st.gmean(feat_dist), 
            "G. Mean" + compl_desc, feat_interpret - 1)
    add_feature(feat_name + "_hmean", st.hmean(np.abs(feat_dist) + 1e-8),
        "G. Mean" + compl_desc, feat_interpret - 1)
    add_feature(feat_name + "_kurtosis", st.kurtosis(feat_dist),
        "Kurtosis" + compl_desc, feat_interpret - 1)
    add_feature(feat_name + "_mode", st.mode(feat_dist)[0][0],
        "Mode" + compl_desc, feat_interpret - 1)
    add_feature(feat_name + "_kstat", st.kstat(feat_dist),
        "Kstat" + compl_desc, feat_interpret - 1)
    add_feature(feat_name + "_kstatvar", st.kstatvar(feat_dist),
        "Kstat variance" + compl_desc, feat_interpret - 1)
    add_feature(feat_name + "_tmean", st.tmean(feat_dist),
        "Tmean" + compl_desc, feat_interpret - 1)
    add_feature(feat_name + "_tmean", st.tvar(feat_dist),
        "Tvariance" + compl_desc, feat_interpret - 1)
    add_feature(feat_name + "_tmin", st.tmin(feat_dist),
        "Tminimum" + compl_desc, feat_interpret - 1)
    add_feature(feat_name + "_tmax", st.tmax(feat_dist),
        "Tmaximum" + compl_desc, feat_interpret - 1)
    add_feature(feat_name + "_tstd", st.tstd(feat_dist),
        "T standard deviation" + compl_desc, feat_interpret - 1)
    add_feature(feat_name + "_tsem", st.tsem(feat_dist),
        "T sem" + compl_desc, feat_interpret - 1)
    add_feature(feat_name + "_variation", st.variation(feat_dist),
        "Variation" + compl_desc, feat_interpret - 1)
    add_feature(feat_name + "_mean_repeats", np.mean(st.find_repeats(feat_dist)[1]),
        "Mean repeats" + compl_desc, feat_interpret - 1)
    add_feature(feat_name + "_entropy", st.entropy(feat_dist),
        "Entropy" + compl_desc, feat_interpret - 1)
    add_feature(feat_name + "_sem", st.sem(feat_dist),
        "SEM" + compl_desc, feat_interpret - 1)
    if len(feat_dist)>1:
        add_feature(feat_name + "_bayes_confint", 
            st.bayes_mvs(feat_dist)[0][1][1] - st.bayes_mvs(feat_dist)[0][1][0],
            "Bayes confidance interval" + compl_desc, feat_interpret - 1)
    else:
        add_feature(feat_name + "_bayes_confint", np.nan,
            "Bayes confidance interval" + compl_desc, feat_interpret - 1)

