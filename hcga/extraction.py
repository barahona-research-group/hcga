"""functions to extract features from graphs"""
import multiprocessing
import time
from collections import defaultdict
from functools import partial
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import utils


def extract(
    graphs,
    n_workers,
    mode="fast",
    normalize_features=False,
    statistics_level="basic",
    with_runtimes=False,
    with_node_features=False,
    timeout=10,
):
    """main function to extract features"""
    if not with_node_features:
        graphs.remove_node_features()
        n_node_features = 0
    else:
        n_node_features = graphs.get_n_node_features()

    feat_classes = get_list_feature_classes(
        mode,
        normalize_features=normalize_features,
        statistics_level=statistics_level,
        n_node_features=n_node_features,
        timeout=timeout,
    )

    if with_runtimes:
        print(
            "WARNING: Runtime option enable, we will only use 10 graphs and one worker to estimate",
            "the computational time of each feature class.",
        )
        selected_graphs = np.random.randint(0, len(graphs), 10)
        for graph in graphs.graphs:
            if graph.id not in selected_graphs:
                graph.disabled = True

    print(
        "Extracting features from",
        len(graphs),
        "graphs (we disabled",
        graphs.get_num_disabled_graphs(),
        "graphs).",
    )
    all_features = compute_all_features(
        graphs, feat_classes, n_workers=n_workers, with_runtimes=with_runtimes,
    )

    if with_runtimes:
        runtimes = defaultdict(list)
        for raw_feature in all_features.values():
            for feat in raw_feature[1]:
                runtimes[feat].append(raw_feature[1][feat])
        feature_names, runtimes = list(runtimes.keys()), list(runtimes.values())
        runtime_sortid = np.argsort(np.mean(runtimes, axis=1))[::-1]
        for feat_id in runtime_sortid:
            print(
                "Runtime of",
                feature_names[feat_id],
                "is",
                np.round(np.mean(runtimes[feat_id]), 3),
                "( std = ",
                np.round(np.std(runtimes[feat_id]), 3),
                ") seconds per graph.",
            )
        return 0.0, 0.0

    all_features_df, features_info_df = gather_features(all_features, feat_classes)
    _set_graph_labels(all_features_df, graphs)

    print(len(all_features_df.columns), "feature extracted.")
    print(len(utils.filter_features(all_features_df).columns), "valid features.")

    return all_features_df, features_info_df


def _set_graph_labels(features, graphs):
    """Set graph labels to features dataframe."""
    for graph in graphs:
        features.loc[graph.id, "labels"] = graph.label


def _load_feature_class(feature_name):
    """load the feature class from feature name"""
    feature_module = import_module("hcga.features." + feature_name)
    return getattr(feature_module, feature_module.featureclass_name)


def get_list_feature_classes(
    mode="fast",
    normalize_features=False,
    statistics_level="basic",
    n_node_features=0,
    timeout=10,
):
    """Generates and returns the list of feature classes to compute for a given mode"""
    feature_path = Path(__file__).parent / "features"
    non_feature_files = ["__init__", "utils"]

    list_feature_classes = []
    for f_name in feature_path.glob("*.py"):
        feature_name = f_name.stem
        if feature_name not in non_feature_files:
            feature_class = _load_feature_class(feature_name)
            if mode in feature_class.modes or mode == "all":
                list_feature_classes.append(feature_class)
                # runs once update_feature with trivial graph to create class variables
                feature_class.setup_class(
                    normalize_features=normalize_features,
                    statistics_level=statistics_level,
                    n_node_features=n_node_features,
                    timeout=timeout,
                )
    return list_feature_classes


def feature_extraction(graph, list_feature_classes, with_runtimes=False):
    """extract features from a single graph"""
    if with_runtimes:
        runtimes = {}

    all_features = {}
    for feature_class in list_feature_classes:
        if with_runtimes:
            start_time = time.time()

        feature_class(graph).update_features(all_features)

        if with_runtimes:
            runtimes[feature_class.shortname] = time.time() - start_time

    if with_runtimes:
        return graph.id, [all_features, runtimes]
    return graph.id, all_features


def compute_all_features(
    graphs, list_feature_classes, n_workers=1, with_runtimes=False,
):
    """compute the feature from all graphs"""
    print("Computing features for {} graphs:".format(len(graphs)))
    if with_runtimes:
        n_workers = 1

    with multiprocessing.Pool(n_workers) as pool:
        results = pool.imap_unordered(
            partial(
                feature_extraction,
                list_feature_classes=list_feature_classes,
                with_runtimes=with_runtimes,
            ),
            graphs,
        )
        return {  # pylint: disable=unnecessary-comprehension
            graph_id: features
            for graph_id, features in tqdm(results, total=len(graphs))
        }


def gather_features(all_features_raw, list_feature_classes):
    """Convert the raw feature to a pandas dataframe and a dict with features infos."""
    graphs_id = list(all_features_raw.keys())

    features_data = all_features_raw[graphs_id[0]]
    feature_name_list = [
        (shortname, name)
        for shortname in list(features_data.keys())
        for name in list(features_data[shortname].keys())
    ]
    column_indexes = pd.MultiIndex.from_tuples(
        feature_name_list, names=["feature_class", "feature_name"]
    )

    all_features_df = pd.DataFrame(index=graphs_id, columns=column_indexes)
    for graph_id, class_features in all_features_raw.items():
        for feature_class, features in class_features.items():
            all_features_df.loc[
                graph_id, (feature_class, features.keys())
            ] = features.values()

    feature_class_inst_tmp = list_feature_classes[0]()
    feature_tmp = list(features_data[feature_class_inst_tmp.shortname].keys())[0]
    feature_info_keys = feature_class_inst_tmp.get_feature_info(feature_tmp).keys()

    features_info_df = pd.DataFrame(index=feature_info_keys, columns=column_indexes)
    for feature_class in list_feature_classes:
        feature_class_inst = feature_class()
        for feature in features_data[feature_class_inst.shortname]:
            features_info_df[(feature_class_inst.shortname, feature)].update(
                pd.Series(feature_class_inst.get_feature_info(feature))
            )

    return all_features_df, features_info_df
