"""plotting functions"""
import os

import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt

# matplotlib.use("agg")


def plot_sklearn_analysis(X, testing_accurary, top_features, folder="."):
    """main function to plot sklearn analysis"""
    # TODO: add other functions, with argument in this one to select what to plot

    plot_feature_importance(X, top_features, folder=folder)
    # plot_dendrogram_top_features(X, top_featres, folder=folder)
    # ...


def plot_feature_importance(X, top_features, folder=".", ext=".svg"):
    """plot the feature importances from sklearn computation"""

    mean_features = np.mean(np.array(top_features), axis=0)
    rank_features = np.argsort(mean_features)[::-1]

    plt.figure()
    plt.bar(np.arange(0, len(X.columns)), mean_features[rank_features])
    plt.xticks(
        np.arange(0, len(X.columns)),
        [X.columns.values.tolist()[i] for i in rank_features],
        rotation="vertical",
    )
    plt.savefig(
        os.path.join(folder, "feature_importance_sklern" + ext), bbox_inches="tight"
    )
    plt.show()


def plot_dendogram_top_features(
    X, top_features, top_feat_indices, image_folder, top_N=40
):
    df_topN = pd.DataFrame(
        columns=top_features_list[:top_N], data=X[:, top_feat_indices[:top_N]]
    )
    cor = np.abs(df_topN.corr())
    Z = linkage(cor, "ward")

    plt.figure()
    dn = dendrogram(Z)
    plt.savefig(
        image_folder + "/endogram_top{}_features.svg".format(top_N), bbox_inches="tight"
    )


def plot_heatmap_top_features():
    new_index = [int(i) for i in dn["ivl"]]
    top_feats_names = [top_features_list[i] for i in new_index]
    df = df_top40[top_feats_names]
    cor2 = np.abs(df.corr())
    plt.figure()
    sns.heatmap(cor2, linewidth=0.5)
    plt.savefig(
        image_folder + "/heatmap_top40_feature_dependencies.svg", bbox_inches="tight"
    )


def plot_top_features(X, top_feats, feature_names, image_folder, threshold=0.9):
    """
    Plot the dendogram, heatmap and importance distribution of top features
    """

    mean_importance = np.mean(np.asarray(top_feats), 0)
    sorted_mean_importance = np.sort(mean_importance)[::-1]

    top_feat_indices = np.argsort(mean_importance)[::-1]
    top_features_list = []
    for i in range(len(top_feat_indices)):
        top_features_list.append(feature_names[top_feat_indices[i]])

    top_features_list = list(dict.fromkeys(top_features_list))

    df_top40 = pd.DataFrame(
        columns=top_features_list[:40], data=X[:, top_feat_indices[:40]]
    )
    cor = np.abs(df_top40.corr())
    Z = linkage(cor, "ward")

    plt.figure()
    dn = dendrogram(Z)
    plt.savefig(image_folder + "/endogram_top40_features.svg", bbox_inches="tight")

    new_index = [int(i) for i in dn["ivl"]]
    top_feats_names = [top_features_list[i] for i in new_index]
    df = df_top40[top_feats_names]
    cor2 = np.abs(df.corr())
    plt.figure()
    sns.heatmap(cor2, linewidth=0.5)
    plt.savefig(
        image_folder + "/heatmap_top40_feature_dependencies.svg", bbox_inches="tight"
    )

    # Taking only features till we have reached 90% importance
    sum_importance = 0
    final_index = 0
    for i in range(len(sorted_mean_importance)):
        sum_importance = sum_importance + sorted_mean_importance[i]
        if sum_importance > threshold:
            final_index = i
            break
    if final_index < 3:  # take top 2 if no features are selected
        final_index = 3

    plt.figure()
    plt.plot(np.sort(mean_importance)[::-1])

    plt.xlabel("Features")
    plt.ylabel("Feature Importance")
    plt.xscale("log")
    plt.yscale("symlog", nonposy="clip", linthreshy=0.001)
    plt.axvline(x=final_index, color="r")
    plt.savefig(
        image_folder + "/feature_importance_distribution.svg", bbox_inches="tight"
    )

    # import pickle as pkl
    # pkl.dump(np.sort(mean_importance)[::-1], open('importance_data/'+image_folder+'.pkl','wb'), pkl.HIGHEST_PROTOCOL)


def top_features_importance_plot(
    g, X, top_feat_indices, feature_names, y, name="xgboost", image_folder="images"
):
    """ 
    Plot the top feature importances
    """

    import matplotlib.cm as cm
    import random

    # mean_importance = np.mean(np.asarray(top_feats),0)
    # top_feat_indices = np.argsort(mean_importance)[::-1]
    plt.figure()
    cm = cm.get_cmap("RdYlBu")
    sc = plt.scatter(X[:, top_feat_indices[0]], X[:, top_feat_indices[1]], cmap=cm, c=y)
    plt.xlabel(feature_names[top_feat_indices[0]])
    plt.ylabel(feature_names[top_feat_indices[1]])
    plt.colorbar(sc)
    plt.savefig(
        image_folder + "/scatter_top2_feats_" + g.dataset + "_" + name + ".svg",
        bbox_inches="tight",
    )


def plot_violin_feature(
    g, X, y, feature_id, feature_names, name="xgboost", image_folder="images"
):
    """
    Plot the violins of a feature
    """

    import random

    feature_data = X[:, feature_id]

    data_split = []
    for k in np.unique(y):
        indices = np.argwhere(y == k)
        data_split.append(feature_data[indices])

    import seaborn as sns

    plt.figure()
    sns.set(style="whitegrid")
    ax = sns.violinplot(data=data_split, palette="muted", width=1)
    ax.set(xlabel="Class label", ylabel=feature_names[feature_id])
    plt.savefig(
        image_folder
        + "/violin_plot_"
        + g.dataset
        + "_"
        + str(feature_names[feature_id])
        + "_"
        + name
        + ".svg",
        bbox_inches="tight",
    )
