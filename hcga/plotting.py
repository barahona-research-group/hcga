"""plotting functions."""
import logging
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import shap
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib.backends.backend_pdf import PdfPages

# pylint: disable-all

L = logging.getLogger(__name__)


def shap_plots(X, y, shap_values, folder, graphs, analysis_type, max_feats=20):
    """plot summary."""
    with PdfPages(os.path.join(folder, "analysis_report.pdf")) as pp:

        pp = custom_bar_ranking_plot(shap_values, X, folder, pp, max_feats=max_feats)
        pp = custom_dot_summary_plot(shap_values, X, folder, pp, max_feats=max_feats)
        pp = plot_dendogram_shap(shap_values, X, folder, pp, max_feats=max_feats)

        if analysis_type == "classification":
            pp = plot_shap_violin(shap_values, X, y, folder, pp, max_feats=max_feats)
        elif analysis_type == "regression":
            pp = plot_trend(shap_values, X, y, folder, pp, max_feats=max_feats)

        pp = plot_feature_summary(X, graphs, folder, pp, shap_values)


def custom_bar_ranking_plot(shap_vals, data, folder, pp, max_feats):
    """Function for customizing and saving SHAP summary bar plot.

    Arguments:
    shap_vals = SHAP values list generated from explainer
    data      = data to explain
    max_feats = number of features to display
    """
    plt.rcParams.update({"font.size": 14})
    shap.summary_plot(
        shap_vals, data, plot_type="bar", max_display=max_feats, show=False
    )
    fig = plt.gcf()
    fig.set_figheight(20)
    fig.set_figwidth(15)
    plt.tight_layout()
    plt.title("Feature Rankings-All Classes")
    plt.savefig(os.path.join(folder, "shap_bar_rank.png"), dpi=200, bbox_inches="tight")
    pp.savefig(fig, bbox_inches="tight")
    return pp


def custom_dot_summary_plot(shap_vals, data, folder, pp, max_feats):
    """Function for customizing and saving SHAP summary dot plot.

    Arguments:
    shap_vals = SHAP values list generated from explainer
    data      = data to explain
    max_feats = number of features to display
    """
    num_classes = len(shap_vals)
    for i in range(num_classes):
        plt.figure()
        L.info("Sample Expanded Feature Summary for Class %s", i)
        plt.rcParams.update({"font.size": 14})
        shap.summary_plot(
            shap_vals[i], data, plot_type="dot", max_display=max_feats, show=False
        )
        fig = plt.gcf()
        fig.set_figheight(20)
        fig.set_figwidth(15)
        plt.tight_layout()
        plt.title("Sample Expanded Feature Summary for Class " + str(i))
        plt.savefig(
            os.path.join(folder, "shap_class_{}_summary.png".format(i)),
            dpi=200,
            bbox_inches="tight",
        )
        pp.savefig(fig, bbox_inches="tight")
    return pp


def plot_dendogram_shap(shap_vals, data, folder, pp, max_feats=20):
    from matplotlib.gridspec import GridSpec

    plt.figure()

    shap_mean = np.sum(np.mean(np.abs(shap_vals), axis=1), axis=0)

    top_feat_idx = shap_mean.argsort()[::-1][:max_feats]

    df_topN = data[data.columns[top_feat_idx]]

    cor = np.abs(df_topN.corr())
    Z = linkage(cor, "ward")

    dn = dendrogram(Z, labels=data.columns[top_feat_idx])

    top_feats_names = [df_topN.columns[i] for i in dn["leaves"]]

    df = df_topN[top_feats_names]
    cor2 = np.abs(df.corr())

    f, ax = plt.subplots(3, 1, figsize=(20, 15))

    dn = dendrogram(Z, ax=ax[0])
    ax[0].xaxis.set_ticklabels([])
    sns.heatmap(
        cor2,
        ax=ax[2],
        linewidth=0.5,
        cbar_ax=ax[1],
        cbar_kws={"label": "Absolute Correlation Coefficient"},
    )
    gs = GridSpec(2, 2, height_ratios=[1, 3], width_ratios=[15, 1])

    ax[0].set_position(gs[0, 0].get_position(f))
    ax[1].set_position(gs[1, 1].get_position(f))
    ax[2].set_position(gs[1, 0].get_position(f))

    ax[0].set_ylabel("Euclidean Distance")

    ax[0].title.set_text("Top {} features heatmap and dendogram".format(max_feats))
    plt.savefig(
        os.path.join(folder, "shap_dendogram_top20.png"), dpi=200, bbox_inches="tight"
    )
    pp.savefig(f, bbox_inches="tight")
    return pp


def plot_feature_summary(data, graphs, folder, pp, shap_vals=None, feat_name=None):
    """for a given feature id, plot the feature summary."""

    if not feat_name:
        shap_mean = np.sum(np.mean(np.abs(shap_vals), axis=1), axis=0)

    feature_data = data.iloc[:, shap_mean.argmax()].sort_values()
    samples = np.round(np.linspace(0, len(feature_data) - 1, 5)).astype(int)

    fig = plt.figure(figsize=(20, 15))
    grid = plt.GridSpec(5, 3, wspace=0.4, hspace=0.3)
    ax = []
    ax.append(plt.subplot(grid[0:, 0:2]))
    ax.append(plt.subplot(grid[4, 2]))
    ax.append(plt.subplot(grid[3, 2]))
    ax.append(plt.subplot(grid[2, 2]))
    ax.append(plt.subplot(grid[1, 2]))
    ax.append(plt.subplot(grid[0, 2]))
    g = sns.violinplot(data=feature_data, ax=ax[0], palette="muted", width=1)

    c = sns.color_palette("hls", 5)
    for i, sample in enumerate(samples):
        graph_id = feature_data.index[sample]  # .index.tolist()
        graph_to_plot = graphs.graphs[graph_id]
        g.axhline(feature_data.iloc[sample], ls="--", color=c[i])

        graph = graph_to_plot.get_graph("networkx")
        pos = nx.spring_layout(graph)
        nx.draw(
            graph,
            pos,
            ax=ax[i + 1],
            node_size=5,
            node_color=[c[i] for n in range(len(graph))],
        )
        ax[i + 1].set_title(
            "Graph ID: {}, y-label: {}".format(
                feature_data.index[sample], graph_to_plot.label[0]
            ),
            fontsize="small",
        )

    fig.suptitle(
        "Feature: {}".format(feature_data.name)
    )  # or plt.suptitle('Main title')
    plt.savefig(
        os.path.join(folder, "feature_{}_summary.png".format(feature_data.name[1])),
        dpi=200,
        bbox_inches="tight",
    )

    pp.savefig(
        fig, bbox_inches="tight",
    )
    return pp


def plot_shap_violin(shap_vals, data, labels, folder, pp, max_feats=20):
    """Plot the violins of a feature."""
    shap_mean = np.sum(np.mean(np.abs(shap_vals), axis=1), axis=0)
    top_feat_idx = shap_mean.argsort()[::-1][:max_feats]

    ncols = 4
    nrows = int(np.ceil(len(top_feat_idx) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, dpi=120, figsize=(20, 14))

    for ax, top_feat in zip(axes.flatten(), top_feat_idx):
        feature_data = data[data.columns[top_feat]].values
        data_split = []

        for k in np.unique(labels):
            indices = np.argwhere(labels.values == k)
            data_split.append(feature_data[indices])

        # sns.set(style="whitegrid")
        sns.violinplot(data=data_split, ax=ax, palette="muted", width=1)
        ax.set(xlabel="Class label", ylabel=data.columns[top_feat])

        ax.tick_params(axis="both", which="major", labelsize=5)

        ax.xaxis.get_label().set_fontsize(7)
        ax.yaxis.get_label().set_fontsize(7)

    plt.subplots_adjust(top=0.9, bottom=0.1, hspace=0.3, wspace=0.2)
    plt.savefig(
        os.path.join(folder, "shap_violins_top20.png"), dpi=200, bbox_inches="tight",
    )
    pp.savefig(
        fig, bbox_inches="tight",
    )
    return pp


def plot_trend(shap_vals, data, labels, folder, pp, max_feats=20):
    """Plot the violins of a feature."""
    shap_mean = np.sum(np.mean(np.abs(shap_vals), axis=1), axis=0)
    top_feat_idx = shap_mean.argsort()[::-1][:max_feats]

    ncols = 4
    nrows = int(np.ceil(len(top_feat_idx) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, dpi=120, figsize=(20, 14))

    for ax, top_feat in zip(axes.flatten(), top_feat_idx):
        feature_data = data[data.columns[top_feat]].values

        sns.scatterplot(feature_data, labels, ax=ax, palette="muted")
        ax.set(xlabel=data.columns[top_feat], ylabel="y-label")

        ax.tick_params(axis="both", which="major", labelsize=5)

        ax.xaxis.get_label().set_fontsize(7)
        ax.yaxis.get_label().set_fontsize(7)

    plt.subplots_adjust(top=0.9, bottom=0.1, hspace=0.3, wspace=0.2)
    plt.savefig(
        os.path.join(folder, "shap_trend_top20.png"), dpi=200, bbox_inches="tight",
    )
    pp.savefig(
        fig, bbox_inches="tight",
    )
    return pp


def pca_plot(features, pca):
    """ plot pca of data """
    X = pca.transform(features)
    plt.scatter(X[:, 0], X[:, 1])
    plt.xlabel("PC1")
    plt.ylabel("PC2")
