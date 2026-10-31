"""Regenerate the feature baseline in tests/data.

Run from the repository root:

    python tests/tools/generate_baseline.py

The baseline is the full feature table (mode "all", advanced statistics, node/edge
normalisation on, in-process evaluation) for every fixture graph in tests/fixtures.py.
tests/test_baseline_drift.py checks that the current code reproduces it. Regenerate it only
when a value change is intended, and say what changed in the commit message.

Some features are unseeded or depend on node iteration order, which changes with Python's
per-process hash seed. The script therefore computes the table in several separate processes
with different PYTHONHASHSEED values, and lists every column that differs between any two of
them in tests/data/baseline_nondeterministic.txt; the drift test skips those columns. The
baseline itself is the first run.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fixtures import fixture_collection  # noqa: E402

from hcga.extraction import feature_extraction, get_list_feature_classes  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
BASELINE_FILE = DATA_DIR / "baseline_features.npz"
NONDET_FILE = DATA_DIR / "baseline_nondeterministic.txt"

# Classes left out of the baseline because they do not terminate in-process on the fixtures:
# SimpleCycles enumerates all undirected cycles (exponential), and nx.omega loops forever on
# the star graph.
EXCLUDED_CLASSES = {"SimpleCycles", "SmallWorldness"}

HASH_SEEDS = tuple(range(8))

# Suffixes added by FeatureClass: node/edge normalisation and the distribution statistics.
NORMALISATION_SUFFIXES = ("_N", "_E")
STATISTIC_SUFFIXES = (
    "_mean", "_min", "_sum", "_max", "_median", "_std",
    "_gmean", "_hmean", "_kurtosis", "_mode",
    "_kstat", "_kstatvar", "_tmean", "_tvar", "_tmin", "_tmax", "_tstd", "_tsem",
    "_variation", "_mean_repeats", "_entropy", "_sem",
)


def feature_stem(column):
    """Strip normalisation and statistic suffixes, so all statistics of one distribution
    (and their _N/_E copies) share a stem."""
    stem = column
    for suffix in NORMALISATION_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    for suffix in STATISTIC_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def statistic_suffix(column):
    """The summary-statistic suffix of a column ("_mode", ...), ignoring any _N/_E suffix,
    or "" for a scalar feature."""
    stem = column
    for suffix in NORMALISATION_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    for suffix in STATISTIC_SUFFIXES:
        if stem.endswith(suffix):
            return suffix
    return ""


def nondeterministic_mask(columns, stems):
    """Boolean array marking columns whose stem is in the nondeterministic set."""
    return np.array([feature_stem(col) in stems for col in columns])


def baseline_classes():
    """Feature classes used for the baseline, set up for in-process evaluation."""
    classes, _ = get_list_feature_classes(
        mode="all", normalize_features=True, statistics_level="advanced", timeout=0
    )
    return [cls for cls in classes if cls.__name__ not in EXCLUDED_CLASSES]


def column_name(column):
    """Flatten a (feature_class, feature_name) column to one string."""
    return "|".join(column)


def compute_table(classes, graph):
    """Return a Series of feature values for one graph, indexed by flat column name."""
    frame = feature_extraction(graph, classes)
    row = frame.iloc[0]
    row.index = [column_name(col) for col in frame.columns]
    return pd.to_numeric(row, errors="coerce").astype(float)


def compute_all():
    """Compute the table for every fixture; returns {fixture name: Series}."""
    collection, names = fixture_collection()
    classes = baseline_classes()
    return {name: compute_table(classes, graph) for name, graph in zip(names, collection.graphs)}


def save_tables(tables, path):
    arrays = {}
    for name, table in tables.items():
        arrays[f"names__{name}"] = np.array(table.index, dtype=str)
        arrays[f"values__{name}"] = table.to_numpy(dtype=float)
    np.savez_compressed(path, **arrays)


def load_tables(path):
    with np.load(path) as data:
        names = sorted(key[len("names__") :] for key in data.files if key.startswith("names__"))
        return {
            name: pd.Series(data[f"values__{name}"], index=data[f"names__{name}"])
            for name in names
        }


def _start_worker(hash_seed, out_path):
    env = dict(os.environ, PYTHONHASHSEED=str(hash_seed))
    return subprocess.Popen(
        [sys.executable, __file__, "--worker", str(out_path)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _compute_in_subprocesses(hash_seeds, tmp_dir):
    """Run one worker per hash seed concurrently; returns the tables of each run."""
    paths = [Path(tmp_dir) / f"run_{seed}.npz" for seed in hash_seeds]
    procs = [_start_worker(seed, path) for seed, path in zip(hash_seeds, paths)]
    for proc in procs:
        if proc.wait() != 0:
            raise RuntimeError("baseline worker failed")
    return [load_tables(path) for path in paths]


def differing_columns(tables_a, tables_b):
    """Columns whose values differ between two runs, over all fixtures."""
    differing = set()
    for name in tables_a:
        a, b = tables_a[name].align(tables_b[name], join="outer")
        same = np.isclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True)
        differing.update(a.index[~same])
    return differing


def main():
    DATA_DIR.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        runs = _compute_in_subprocesses(HASH_SEEDS, tmp)
    nondeterministic = set()
    for other in runs[1:]:
        nondeterministic |= {feature_stem(col) for col in differing_columns(runs[0], other)}

    save_tables(runs[0], BASELINE_FILE)
    NONDET_FILE.write_text("".join(f"{stem}\n" for stem in sorted(nondeterministic)))

    n_cols = {name: len(table) for name, table in runs[0].items()}
    print(f"wrote {BASELINE_FILE} with columns per fixture: {n_cols}")
    print(f"{len(nondeterministic)} nondeterministic feature stems listed in {NONDET_FILE}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        save_tables(compute_all(), sys.argv[2])
    else:
        main()
