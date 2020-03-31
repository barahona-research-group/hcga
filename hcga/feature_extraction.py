"""functions to extract features from graphs"""
import multiprocessing
import time
from importlib import import_module
from pathlib import Path
import networkx as nx
import numpy as np
from collections import defaultdict
from tqdm import tqdm

import pandas as pd

from . import utils


def _set_graph_id(graphs):
    """set graphs ids for logging"""
    for i, graph in enumerate(graphs):
        graph.graph['id'] = i

def extract(
    graphs,
    n_workers,
    mode="fast",
    normalize_features=False,
    statistics_level="basic",
    with_runtimes=False,
):
    """main function to extract features"""
    _set_graph_id(graphs)

    feat_classes = get_list_feature_classes(
        mode, normalize_features=normalize_features, statistics_level=statistics_level
    )
    if with_runtimes:
        print(
            "WARNING: Runtime option enable, we will only use 10 graphs and one worker to estimate",
            "the computational time of each feature class.",
        )
        graphs = graphs[:10]


    raw_features = compute_all_features(
        graphs, feat_classes, n_workers=n_workers, with_runtimes=with_runtimes,
    )

    if with_runtimes:
        runtimes = [raw_feature[1] for raw_feature in raw_features]
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
        features, features_info = gather_features(raw_features, feat_classes)
        features["labels"] = [graph.label for graph in graphs]

        print(len(features.columns), "feature extracted.")
        good_features = utils.filter_features(features)
        print(len(good_features.columns), "good features")

        return features, features_info


def _load_feature_class(feature_name):
    """load the feature class from feature name"""
    feature_module = import_module("hcga.features." + feature_name)
    return getattr(feature_module, feature_module.featureclass_name)


def get_list_feature_classes(
    mode="fast", normalize_features=False, statistics_level="basic"
):
    """Generates and returns the list of feature classes to compute for a given mode"""
    feature_path = Path(__file__).parent / "features"
    non_feature_files = ["__init__", "feature_class", "utils"]

    list_feature_classes = []
    trivial_graph = utils.get_trivial_graph()

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
                )
    return list_feature_classes


class Worker:
    """worker for computing features"""

    def __init__(
        self, list_feature_classes, with_runtimes=False):
        self.list_feature_classes = list_feature_classes
        self.with_runtimes = with_runtimes

    def __call__(self, graph):
        return feature_extraction(graph, self.list_feature_classes, with_runtimes=self.with_runtimes)


def feature_extraction(graph, list_feature_classes, with_runtimes=False):
    """extract features from a single graph"""
    if with_runtimes:
        runtimes = {}

    all_features = {}
    for feature_class in list_feature_classes:
        if with_runtimes:
            start_time = time.time()

        feature_inst = feature_class(graph)
        feature_inst.update_features(all_features)

        if with_runtimes:
            runtimes[feature_class.shortname] = time.time() - start_time

    if with_runtimes:
        return [all_features, runtimes]
    return all_features


def compute_all_features(
    graphs,
    list_feature_classes,
    n_workers=1,
    with_runtimes=False,
):
    """compute the feature from all graphs"""
    print("Computing features for {} graphs:".format(len(graphs)))

    worker = Worker(list_feature_classes, with_runtimes=with_runtimes)
    if with_runtimes:
        n_workers = 1

    if n_workers == 1:
        mapper = map
    else:
        pool = multiprocessing.Pool(n_workers)
        mapper = pool.imap

    return list(tqdm(mapper(worker, graphs), total=len(graphs)))


def gather_features(all_features_raw, list_feature_classes):
    """convert the raw feature to a pandas dataframe and a dict with features infos"""

    features_info = {}
    all_features = {}

    for feature_class in list_feature_classes:
        feature_inst = feature_class()

        for feature in all_features_raw[0][feature_inst.shortname]:
            feature_info = feature_inst.get_feature_info(feature)
            features_info[feature_info["feature_name"]] = feature_info

            all_features[feature_info["feature_name"]] = []
            for i, _ in enumerate(all_features_raw):
                all_features[feature_info["feature_name"]].append(
                    all_features_raw[i][feature_inst.shortname][feature]
                )

    return pd.DataFrame.from_dict(all_features), features_info
