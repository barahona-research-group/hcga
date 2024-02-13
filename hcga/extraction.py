"""Functions necessary for the extraction of graph features."""

import logging
import time
from functools import partial
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from hcga.utils import NestedPool

L = logging.getLogger(__name__)


def extract(
    graphs,
    n_workers,
    mode="fast",
    normalize_features=True,
    statistics_level="basic",
    with_runtimes=False,
    with_node_features=True,
    timeout=10,
    connected=False,
    weighted=True,
):
    """Main function to extract graph features.

    Args:
        graphs (GraphCollection object): GraphCollection object with loaded graphs (see graph.py)
        n_workers (int): number of workers for parallel processing
        mode (str): 'fast', 'medium', 'slow' - only features that are fast to compute will
            be run with 'fast'
        normalize_features (bool): normalise features by number of nodes and number of edges
        statistics_level (str): 'basic', 'advanced' - for features that provide distributions
            we must compute statistics.
        with_runtimes (bool): calculating the run time of each feature.
        with_node_features (bool): include node features in feature extraction
        timeout (int): number of seconds before the calculation for a feature is cancelled
        connected (bool): True will make sure that only the largest connected component of a graph
            is used for feature extraction.
        weighted (bool): calculations will consider edge weights where possible.

    Returns:
        (DataFrame): dataframe of features
        (DataFrame): dataframe of meta information of computed features.
    """
    if not with_node_features:
        graphs.remove_node_features()
        n_node_features = 0
    else:
        n_node_features = graphs.get_n_node_features()

    if connected:
        graphs.maximal_subgraphs()

    if not weighted:
        graphs.remove_edge_weights()

    feat_classes, features_info_df = get_list_feature_classes(
        mode,
        normalize_features=normalize_features,
        statistics_level=statistics_level,
        n_node_features=n_node_features,
        timeout=timeout,
    )
    if with_runtimes:
        L.info(
            "Runtime option enable, we will only use 10 graphs and one worker to estimate \
            the computational time of each feature class.",
        )
        selected_graphs = np.random.randint(0, len(graphs), 10)
        for graph in graphs.graphs:
            if graph.id not in selected_graphs:
                graph.disabled = True

    L.info(
        "Extracting features from %s graphs (we disabled %s graphs).",
        len(graphs),
        graphs.get_num_disabled_graphs(),
    )
    all_features_df = compute_all_features(
        graphs,
        feat_classes,
        n_workers=n_workers,
        with_runtimes=with_runtimes,
    )

    if with_runtimes:
        _print_runtimes(all_features_df)
        return 0.0, 0.0

    _set_graph_labels(all_features_df, graphs)

    L.info("%s feature extracted.", len(all_features_df.columns))
    return all_features_df, features_info_df


def _print_runtimes(all_features_df):
    """Print sorted runtimes."""
    mean = all_features_df["runtimes"].mean(axis=0).to_list()
    std = all_features_df["runtimes"].std(axis=0).to_list()
    sortid = np.argsort(mean)[::-1]

    for i in sortid:
        L.info(
            "Runtime of %s is %s ( std = %s ) seconds per graph.",
            all_features_df["runtimes"].columns[i],
            np.round(mean[i], 3),
            np.round(std[i], 3),
        )


def _set_graph_labels(features, graphs):
    """Set graph labels to features dataframe."""
    for graph in graphs:
        features.loc[graph.id, "label"] = graph.label


def _load_feature_class(feature_name):
    """load the feature class from feature name."""
    feature_module = import_module("hcga.features." + feature_name)
    return getattr(feature_module, feature_module.featureclass_name)


def get_list_feature_classes(
    mode="fast",
    normalize_features=True,
    statistics_level="basic",
    n_node_features=0,
    timeout=10,
):
    """Generates and returns the list of feature classes to compute for a given mode.

    Args:
        mode (str): 'fast', 'medium', 'slow' - only features that are fast to compute will
            be run with 'fast'
        normalize_features (bool): normalise features by number of nodes and number of edges
        statistics_level (str): 'basic', 'advanced' - for features that provide distributions
            we must compute statistics.
        n_node_features (int):  dimension of node features for feature constructors
        timeout (int): number of seconds before the calculation for a feature is cancelled

    Returns:
        (list): list of feature classes instances
        (DataFrame): dataframe with feature information
    """
    feature_path = Path(__file__).parent / "features"
    non_feature_files = ["__init__", "utils"]

    list_feature_classes = []
    column_indexes = pd.MultiIndex(
        levels=[[], []], codes=[[], []], names=["feature_class", "feature_name"]
    )
    feature_info_df = pd.DataFrame(columns=column_indexes)
    feature_names = [
        _f.stem for _f in feature_path.glob("*.py") if _f.stem not in non_feature_files
    ]
    L.info("Setting up feature classes...")
    for feature_name in tqdm(feature_names):
        feature_class = _load_feature_class(feature_name)
        if mode in feature_class.modes or mode == "all":
            list_feature_classes.append(feature_class)
            # runs once update_feature with trivial graph to create class variables
            features_info = feature_class.setup_class(
                normalize_features=normalize_features,
                statistics_level=statistics_level,
                n_node_features=n_node_features,
                timeout=timeout,
            )
            columns = [(feature_class.shortname, col) for col in features_info.columns]
            feature_info_df[columns] = features_info

    return list_feature_classes, feature_info_df


def feature_extraction(graph, list_feature_classes, with_runtimes=False):
    """Extract features for a single graph

    Args:
        graph (Graph object): Graph object (see graph.py)
        list_feature_classes (list): list of feature classes found in ./features
        with_runtimes (bool): compute the run time of each feature

    Returns:
        (DataFrame): dataframe of calculated features for a given graph.
    """
    L.debug("computing %s", graph)
    column_indexes = pd.MultiIndex(
        levels=[[], []], codes=[[], []], names=["feature_class", "feature_name"]
    )
    features_df = pd.DataFrame(columns=column_indexes)
    for i, feature_class in enumerate(list_feature_classes):
        L.debug("computing: %s/ %s, %s", i, len(list_feature_classes), feature_class)
        if with_runtimes:
            start_time = time.time()

        feat_class_inst = feature_class(graph)
        features = pd.DataFrame(feat_class_inst.get_features(), index=[graph.id])
        columns = [(feat_class_inst.shortname, col) for col in features.columns]
        features_df[columns] = features
        del feat_class_inst
        L.debug("done with: %s/ %s, %s", i, len(list_feature_classes), feature_class)

        if with_runtimes:
            features_df[("runtimes", feature_class.name)] = time.time() - start_time

    if with_runtimes:
        return features_df

    return features_df


def compute_all_features(
    graphs,
    list_feature_classes,
    n_workers=1,
    with_runtimes=False,
):
    """Compute features for all graphs

    Args:
        graphs (GraphCollection object): GraphCollection object with loaded graphs (see graph.py)
        list_feature_classes (list): list of feature classes found in ./features
        n_workers (int): number of workers for parallel processing
        with_runtimes (bool): compute the run time of each feature

    Returns:
        (DataDrame): dataframe of calculated features for the graph collection.
    """

    L.info("Computing features for %s graphs:", len(graphs))
    if with_runtimes:
        n_workers = 1

    with NestedPool(n_workers) as pool:
        return pd.concat(
            tqdm(
                pool.imap(
                    partial(
                        feature_extraction,
                        list_feature_classes=list_feature_classes,
                        with_runtimes=with_runtimes,
                    ),
                    graphs,
                ),
                total=len(graphs),
            )
        )
