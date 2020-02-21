"""utils functions for feature extractions"""
import scipy.stats as st
import numpy as np


def normalize_features(feature_instance):
    """triple the number of features by normalising by node and edges"""

    for feature_name in list(feature_instance.features.keys()):
        feature_instance.add_feature(
                feature_name + "_N", 
                feature_instance.features[feature_name] / len(feature_instance.graph.nodes),
                (feature_instance.__class__.feature_descriptions[feature_name] 
                    + ", normalised by the number of nodes in the graph"),
                )
        feature_instance.add_feature(
                feature_name + "_E", 
                feature_instance.features[feature_name] / len(feature_instance.graph.edges),
                (feature_instance.__class__.feature_descriptions[feature_name] 
                    + ", normalised by the number of edges in the graph"),
                )

def summary_statistics(add_feature, feat_dist, feat_name, feat_desc):
    """Computes summary statistics of distributions"""
    compl_desc = " of the distribution of " + feat_desc

    add_feature(feat_name + "_mean", np.mean(feat_dist), "Mean" + compl_desc)
    add_feature(feat_name + "_min", np.min(feat_dist), "Minimum" + compl_desc)
    add_feature(feat_name + "_max", np.max(feat_dist), "Maximum" + compl_desc)
    add_feature(feat_name + "_median", np.median(feat_dist), "Median" + compl_desc)
    add_feature(feat_name + "_std", np.std(feat_dist), "Standard deviation" + compl_desc)
    add_feature(feat_name + "_gmean", st.gmean(feat_dist), "G. Mean" + compl_desc)
    add_feature(feat_name + "_hmean", st.hmean(np.abs(feat_dist) + 1e-8),
        "G. Mean" + compl_desc)
    add_feature(feat_name + "_kurtosis", st.kurtosis(feat_dist),
        "Kurtosis" + compl_desc)
    add_feature(feat_name + "_mode", st.mode(feat_dist)[0][0],
        "Mode" + compl_desc)
    add_feature(feat_name + "_kstat", st.kstat(feat_dist),
        "Kstat" + compl_desc)
    add_feature(feat_name + "_kstatvar", st.kstatvar(feat_dist),
        "Kstat variance" + compl_desc)
    add_feature(feat_name + "_tmean", st.tmean(feat_dist),
        "Tmean" + compl_desc)
    add_feature(feat_name + "_tmean", st.tvar(feat_dist),
        "Tvariance" + compl_desc)
    add_feature(feat_name + "_tmin", st.tmin(feat_dist),
        "Tminimum" + compl_desc)
    add_feature(feat_name + "_tmax", st.tmax(feat_dist),
        "Tmaximum" + compl_desc)
    add_feature(feat_name + "_tstd", st.tstd(feat_dist),
        "T standard deviation" + compl_desc)
    add_feature(feat_name + "_tsem", st.tsem(feat_dist),
        "T sem" + compl_desc)
    add_feature(feat_name + "_variation", st.variation(feat_dist),
        "Variation" + compl_desc)
    add_feature(feat_name + "_mean_repeats", np.mean(st.find_repeats(feat_dist)[1]),
        "Mean repeats" + compl_desc)
    add_feature(feat_name + "_entropy", st.entropy(feat_dist),
        "Entropy" + compl_desc)
    add_feature(feat_name + "_sem", st.sem(feat_dist),
        "SEM" + compl_desc)
    add_feature(feat_name + "_bayes_confint", 
        st.bayes_mvs(feat_dist)[0][1][1] - st.bayes_mvs(feat_dist)[0][1][0],
        "Bayes confidance interval" + compl_desc)

