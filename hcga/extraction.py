"""functions to extract features from graphs"""
import multiprocessing
import time
from collections import defaultdict
from functools import partial
from importlib import import_module
from pathlib import Path

import networkx as nx
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
    ensure_connectivity=False,
):
    """main function to extract features"""
    n_node_features = graphs.get_n_node_features()

    feat_classes = get_list_feature_classes(
        mode,
        normalize_features=normalize_features,
        statistics_level=statistics_level,
        n_node_features=n_node_features,
    )

    if ensure_connectivity:
        graphs = _ensure_connectivity(graphs)

    if with_runtimes:
        print(
            "WARNING: Runtime option enable, we will only use 10 graphs and one worker to estimate",
            "the computational time of each feature class.",
        )
        graphs = graphs[:10]

    all_features = compute_all_features(
        graphs, feat_classes, n_workers=n_workers, with_runtimes=with_runtimes,
    )

    if with_runtimes:
        runtimes = [raw_feature[1] for raw_feature in all_features.values()]
        list_runtimes = defaultdict(list)
        for runtime in runtimes:
            for feat in runtime:
                list_runtimes[feat].append(runtime[feat])

        for feat in list_runtimes:
            print(
                "Runtime of",
                feat,
                "is",
                np.round(np.mean(list_runtimes[feat]), 3),
                "seconds per graph.",
            )
        return 0.0, 0.0

    else:
        features, features_info = gather_features(all_features, feat_classes)
        _set_graph_labels(features, graphs)

        print(len(features.columns), "feature extracted.")
        good_features = utils.filter_features(features)
        print(len(good_features.columns), "good features")

        return features, features_info


def _set_graph_labels(features, graphs):
    """Set graph labels to features dataframe."""
    for graph in graphs:
        features.loc[graph.id, "labels"] = graph.label


def _load_feature_class(feature_name):
    """load the feature class from feature name"""
    feature_module = import_module("hcga.features." + feature_name)
    return getattr(feature_module, feature_module.featureclass_name)


def get_list_feature_classes(
    mode="fast", normalize_features=False, statistics_level="basic", n_node_features=0
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
        return {
            graph_id: features
            for graph_id, features in tqdm(results, total=len(graphs))
        }


def gather_features(all_features_raw, list_feature_classes):
    """Convert the raw feature to a pandas dataframe and a dict with features infos."""
    features_info = {}
    feature_name_list = []
    for feature_class in list_feature_classes:
        feature_class_inst = feature_class()
        for feature in all_features_raw[list(all_features_raw.keys())[0]][
            feature_class_inst.shortname
        ]:
            feature_info = feature_class_inst.get_feature_info(feature)
            features_info[feature_info["fullname"]] = feature_info
            feature_name_list.append((feature_info["shortname"], feature_info["name"]))

    column_indexes = pd.MultiIndex.from_tuples(
        feature_name_list, names=["feature_class", "feature_name"]
    )
    all_features = pd.DataFrame(columns=column_indexes)
    for graph_id, features in all_features_raw.items():
        all_features.loc[graph_id] = [
            f for feat in features.values() for f in feat.values()
        ]
    return all_features, features_info


def _ensure_connectivity(graphs):
    # take the largest connected component of the graph
    for i, graph in enumerate(graphs):
        if not nx.is_connected(graph):
            print(
                "Graph "
                + str(i)
                + " is not connected. Taking largest subgraph and relabelling the nodes."
            )
            Gcc = sorted(nx.connected_components(graph), key=len, reverse=True)
            G0 = graph.subgraph(Gcc[0])
            mapping = dict(zip(G0.nodes, range(0, len(G0))))
            G0 = nx.relabel_nodes(G0, mapping)
            G0.label = graph.label
            graphs[i] = G0
    return graphs
