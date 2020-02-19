"""utils functions for feature extractions"""
import scipy.stats as st
import numpy as np


def normalize_features(feature_list, graph):
    """triple the number of features by normalising by node and edges"""

    for feature in list(feature_list.keys()):
        feature_list[feature + '_N'] = feature_list[feature] / len(graph.nodes)
        feature_list[feature + '_E'] = feature_list[feature] / len(graph.edges)

def summary_statistics(feature_list, dist, feat_name):
    """Computes summary statistics of distributions"""

    feature_list[feat_name + "_mean"] = np.mean(dist)
    feature_list[feat_name + "_min"] = np.min(dist)
    feature_list[feat_name + "_max"] = np.max(dist)
    feature_list[feat_name + "_median"] = np.median(dist)
    feature_list[feat_name + "_std"] = np.std(dist)
    feature_list[feat_name + "_gmean"] = st.gmean(dist)
    feature_list[feat_name + "_hmean"] = st.hmean(np.abs(dist) + 1e-8)
    feature_list[feat_name + "_kurtosis"] = st.kurtosis(dist)
    feature_list[feat_name + "_mode"] = st.mode(dist)[0][0]
    feature_list[feat_name + "_kstat"] = st.kstat(dist)
    feature_list[feat_name + "_kstatvar"] = st.kstatvar(dist)
    feature_list[feat_name + "_tmean"] = st.tmean(dist)
    feature_list[feat_name + "_tvar"] = st.tvar(dist)
    feature_list[feat_name + "_tmin"] = st.tmin(dist)
    feature_list[feat_name + "_tmax"] = st.tmax(dist)
    feature_list[feat_name + "_tstd"] = st.tstd(dist)
    feature_list[feat_name + "_tsem"] = st.tsem(dist)
    feature_list[feat_name + "_variation"] = st.variation(dist)
    feature_list[feat_name + "_mean_repeats"] = np.mean(st.find_repeats(dist)[1])
    feature_list[feat_name + "_entropy"] = st.entropy(dist)
    feature_list[feat_name + "_sem"] = st.sem(dist)
    feature_list[feat_name + "_bayes_confint"] = (
            st.bayes_mvs(dist)[0][1][1] - st.bayes_mvs(dist)[0][1][0]
        )

    return feature_list
