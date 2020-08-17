"""plotting functions."""
import logging
import os

import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import networkx as nx
import shap

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


L = logging.getLogger(__name__)


def _save_to_pdf(pdf, figs=None):
    if figs is not None:
        for fig in figs:
            pdf.savefig(fig, bbox_inches="tight")
        plt.close("all")
    else:
        pdf.savefig(bbox_inches="tight")
        plt.close()


def plot_prediction(analysis_results, folder, ext=".png", figsize=(6, 6)):
    """Plot the prediction of the trained model."""

    def _plot_pred(analysis_results, model, X, acc):
        train_index, test_index = analysis_results["indices"]
        prediction_test = model.predict(X.iloc[test_index])
        prediction_train = model.predict(X.iloc[train_index])
        plt.figure(figsize=figsize)
        plt.plot(
            analysis_results["y"].iloc[test_index],
            prediction_test,
            "+",
            c="C0",
            label="test samples",
        )
        plt.plot(
            analysis_results["y"].iloc[train_index],
            prediction_train,
            ".",
            c="0.5",
            label="train samples",
        )
        plt.plot(analysis_results["y"], analysis_results["y"], ls="--", c="k")
        plt.legend(loc="best")
        plt.suptitle("Accuracy: " + str(np.round(acc, 2)))
        plt.xlabel("Value")
        plt.ylabel("Predicted value")

    _plot_pred(
        analysis_results,
        analysis_results["model"],
        analysis_results["X"],
        analysis_results["acc_score"],
    )
    plt.savefig(os.path.join(folder, "prediction" + ext), dpi=200, bbox_inches="tight")
    plt.close()
    if "reduced_model" in analysis_results:
        _plot_pred(
            analysis_results,
            analysis_results["reduced_model"],
            analysis_results["X"][analysis_results["reduced_features"]],
            analysis_results["reduced_acc_score"],
        )
        plt.savefig(
            os.path.join(folder, "reduced_prediction" + ext),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()


def plot_analysis(
    analysis_results,
    folder,
    graphs,
    analysis_type,
    max_feats=20,
    max_feats_dendrogram=100,
    ext=".svg",
):
    """Plot summary of hcga analysis."""
    if "reduced_mean_shap_values" in analysis_results:
        L.info("Using reduced features for plotting.")
        shap_values = analysis_results["reduced_mean_shap_values"]
        reduced_features = analysis_results["reduced_features"]
    else:
        shap_values = analysis_results["mean_shap_values"]
        reduced_features = analysis_results["X"].columns

    X = analysis_results["X"][reduced_features]
    y = analysis_results["y"]

    with PdfPages(os.path.join(folder, "analysis_report.pdf")) as pdf:
        _bar_ranking_plot(shap_values, X, folder, max_feats, ext=ext)
        _save_to_pdf(pdf)

        _dot_summary_plot(shap_values, X, folder, max_feats, ext=ext)
        _save_to_pdf(pdf)

        _plot_feature_correlation(
            analysis_results["shap_values"],
            analysis_results["X"],
            reduced_features,
            folder,
            max_feats_dendrogram,
            ext=ext,
        )
        _save_to_pdf(pdf)
        _plot_dendrogram_shap(
            analysis_results["mean_shap_values"],
            analysis_results["X"],
            reduced_features,
            folder,
            max_feats_dendrogram,
            ext=ext,
        )
        _save_to_pdf(pdf)

        if analysis_type == "classification":
            _plot_shap_violin(shap_values, X, y, folder, max_feats, ext=ext)
        elif analysis_type == "regression":
            _plot_trend(shap_values, X, y, folder, max_feats, ext=ext)
        _save_to_pdf(pdf)

        figs = _plot_feature_summary(
            X[reduced_features], y, graphs, folder, shap_values, max_feats, ext=ext
        )
        _save_to_pdf(pdf, figs)


def _bar_ranking_plot(shap_values, data, folder, max_feats, ext=".png"):
    """Function for customizing and saving SHAP summary bar plot."""
    shap.summary_plot(
        shap_values, data, plot_type="bar", max_display=max_feats, show=False
    )
    plt.title("Feature Rankings-All Classes")
    plt.savefig(
        os.path.join(folder, "shap_bar_rank" + ext), dpi=200, bbox_inches="tight"
    )


def _dot_summary_plot(shap_values, data, folder, max_feats, ext=".png"):
    """Function for customizing and saving SHAP summary dot plot."""
    print(np.shape(shap_values))
    num_classes = len(shap_values)
    for i in range(num_classes):
        plt.figure()
        shap.summary_plot(
            shap_values[i], data, plot_type="dot", max_display=max_feats, show=False
        )
        plt.title("Sample Expanded Feature Summary for Class " + str(i))
        plt.savefig(
            os.path.join(folder, "shap_class_{}_summary{}".format(i, ext)),
            dpi=200,
            bbox_inches="tight",
        )


def _plot_dendrogram_shap(
    shap_values, X, reduced_features, folder, max_feats, ext=".png"
):
    """Plot dendrogram witth hierarchical clustering."""
    shap_mean = np.sum(np.mean(np.abs(shap_values), axis=1), axis=0)
    top_feat_idx = shap_mean.argsort()[-max_feats:]
    df_topN = X[X.columns[top_feat_idx]]

    plt.figure(figsize=(20, 1.2 * 20))
    gs = GridSpec(2, 1, height_ratios=[0.2, 1.0])
    gs.update(hspace=0)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[1, 0])

    cor = np.abs(df_topN.corr())
    Z = linkage(cor, "ward")
    dn = dendrogram(Z, labels=X.columns[top_feat_idx], ax=ax1)
    ax1.xaxis.set_ticklabels([])
    ax1.set_ylabel("Euclidean Distance")

    top_feats_names = df_topN.columns[dn["leaves"]]
    cor_sorted = np.abs(df_topN[top_feats_names].corr())

    cb_axis = inset_axes(ax1, width="5%", height="12%", borderpad=2, loc="upper right")
    sns.heatmap(
        cor_sorted,
        square=True,
        ax=ax2,
        linewidth=0.5,
        cbar_ax=cb_axis,
        cbar_kws={"label": "|pearson|"},
    )
    cor_sorted["id"] = np.arange(len(cor_sorted))
    reduced_id = cor_sorted.loc[reduced_features, ("id", "")]
    ax2.scatter(reduced_id + 0.5, reduced_id + 0.5, c="g", s=100)
    ax1.title.set_text("Top {} features heatmap and dendogram".format(max_feats))
    plt.savefig(
        os.path.join(folder, "shap_dendogram" + ext), dpi=200, bbox_inches="tight"
    )


def _plot_feature_correlation(
    shap_values, X, reduced_features, folder, max_feats, ext=".png"
):
    """Plot correlation matrix."""
    shap_mean = np.sum(np.mean(np.abs(shap_values), axis=1), axis=0)
    top_feat_idx = shap_mean.argsort()[-max_feats:]
    df_topN = X[X.columns[top_feat_idx]]
    df_topN = df_topN.sort_index(axis=0).sort_index(axis=1)
    cor_sorted = np.abs(df_topN.corr())

    plt.figure(figsize=(20, 20))
    ax = plt.gca()
    cb_axis = inset_axes(ax, width="5%", height="20%", borderpad=2, loc="upper right")
    sns.heatmap(
        cor_sorted,
        square=True,
        ax=ax,
        linewidth=0.5,
        cbar_ax=cb_axis,
        cbar_kws={"label": "|pearson|"},
    )
    cor_sorted["id"] = np.arange(len(cor_sorted))
    reduced_id = cor_sorted.loc[reduced_features, ("id", "")]
    ax.scatter(reduced_id + 0.5, reduced_id + 0.5, c="g", s=100)
    plt.savefig(
        os.path.join(folder, "correlation_matrix" + ext), dpi=200, bbox_inches="tight"
    )


PERCENTILES = [2, 25, 50, 75, 98]


def _plot_feature_summary(
    data, labels, graphs, folder, shap_values, max_feats, ext=".png"
):
    """for a given feature id, plot the feature summary."""

    shap_mean = np.sum(np.mean(np.abs(shap_values), axis=1), axis=0)
    feature_names = list(data.iloc[:, shap_mean.argsort()[-max_feats:]].columns)

    figs = []
    for feature_name in feature_names:
        p_vals = [np.percentile(data[feature_name], p) for p in PERCENTILES]
        p_ids = [abs(data[feature_name] - p_val).idxmin() for p_val in p_vals]

        figs.append(plt.figure())
        grid = plt.GridSpec(5, 3, wspace=0.4, hspace=0.3)
        ax = []
        ax.append(plt.subplot(grid[0:, 0:2]))
        ax.append(plt.subplot(grid[4, 2]))
        ax.append(plt.subplot(grid[3, 2]))
        ax.append(plt.subplot(grid[2, 2]))
        ax.append(plt.subplot(grid[1, 2]))
        ax.append(plt.subplot(grid[0, 2]))

        g = sns.violinplot(data=data[feature_name], ax=ax[0], palette="muted", bw=0.1)

        c = sns.color_palette("hls", 5)
        for i, p_id in enumerate(p_ids):
            graph_to_plot = graphs.graphs[p_id]
            g.axhline(data.loc[p_id, feature_name], ls="--", color=c[i])

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
                "Graph ID: {}, y-label: {}".format(p_id, np.round(labels[p_id], 3)),
                fontsize="small",
            )

        feature_name = feature_name[0] + "_" + feature_name[1]
        figs[-1].suptitle("Feature: {}".format(feature_name))
        plt.savefig(
            os.path.join(folder, "feature_{}_summary{}".format(feature_name, ext)),
            dpi=200,
            bbox_inches="tight",
        )
    return figs


def _plot_shap_violin(shap_vals, data, labels, folder, max_feats, ext=".png"):
    """Plot the violins of a feature."""
    shap_mean = np.sum(np.mean(np.abs(shap_vals), axis=1), axis=0)
    top_feat_idx = shap_mean.argsort()[::-1][:max_feats]

    ncols = 4
    nrows = int(np.ceil(len(top_feat_idx) / ncols))
    _, axes = plt.subplots(nrows=nrows, ncols=ncols, dpi=120, figsize=(20, 14))

    for ax, top_feat in zip(axes.flatten(), top_feat_idx):
        feature_data = data[data.columns[top_feat]].values
        data_split = []

        for k in np.unique(labels):
            indices = np.argwhere(labels.values == k)
            data_split.append(feature_data[indices])

        sns.violinplot(data=data_split, ax=ax, palette="muted", width=1)
        ax.set(xlabel="Class label", ylabel=data.columns[top_feat])

        ax.tick_params(axis="both", which="major", labelsize=5)

        ax.xaxis.get_label().set_fontsize(7)
        ax.yaxis.get_label().set_fontsize(7)

    plt.subplots_adjust(top=0.9, bottom=0.1, hspace=0.3, wspace=0.2)
    plt.savefig(
        os.path.join(folder, "shap_violins" + ext), dpi=200, bbox_inches="tight",
    )


def _plot_trend(shap_vals, data, labels, folder, max_feats, ext=".png"):
    """Plot the violins of a feature."""
    shap_mean = np.sum(np.mean(np.abs(shap_vals), axis=1), axis=0)
    top_feat_idx = shap_mean.argsort()[::-1][:max_feats]

    ncols = 4
    nrows = int(np.ceil(len(top_feat_idx) / ncols))
    _, axes = plt.subplots(nrows=nrows, ncols=ncols, dpi=120, figsize=(20, 14))

    for ax, top_feat in zip(axes.flatten(), top_feat_idx):
        feature_data = data[data.columns[top_feat]].values

        sns.scatterplot(feature_data, labels, ax=ax, palette="muted")
        ax.set(xlabel=data.columns[top_feat], ylabel="y-label")

        ax.tick_params(axis="both", which="major", labelsize=5)

        ax.xaxis.get_label().set_fontsize(7)
        ax.yaxis.get_label().set_fontsize(7)

    plt.subplots_adjust(top=0.9, bottom=0.1, hspace=0.3, wspace=0.2)
    plt.savefig(
        os.path.join(folder, "shap_trend" + ext), dpi=200, bbox_inches="tight",
    )


def pca_plot(features, pca):
    """ plot pca of data """
    X = pca.transform(features)
    plt.scatter(X[:, 0], X[:, 1])
    plt.xlabel("PC1")
    plt.ylabel("PC2")
