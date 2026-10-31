"""Check that the feature table still matches the frozen baseline.

A failure here means a value changed. If the change is intended, regenerate the baseline with
tests/tools/generate_baseline.py and describe the change in the commit message; do not loosen
the tolerances to make the test pass.

The baseline was computed on macOS (Accelerate BLAS). Other BLAS/LAPACK builds and other
networkx versions reproduce it to about 1e-9 relative, so the tolerance is 1e-6: loose enough
for rounding, tight enough to catch any real change in a feature. A few summary statistics are
not well conditioned and are skipped: ``mode`` and ``mean_repeats`` count exact float repeats
(symmetric nodes agree only up to rounding), ``gmean``, ``hmean`` and ``entropy`` depend on the
sign of eigenvalues that are zero up to rounding, ``variation`` divides by a mean that is
identically zero for adjacency spectra, and ``kurtosis`` of a constant distribution is NaN or
rounding noise.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures import fixture_collection  # noqa: E402
from tools.generate_baseline import (  # noqa: E402
    BASELINE_FILE,
    NONDET_FILE,
    baseline_classes,
    compute_table,
    load_tables,
    nondeterministic_mask,
    statistic_suffix,
)

RTOL = 1e-6
ATOL = 1e-9
ILL_CONDITIONED_STATISTICS = {
    "_mode",
    "_mean_repeats",
    "_gmean",
    "_hmean",
    "_entropy",
    "_variation",
    "_kurtosis",
}


@pytest.fixture(scope="module")
def baseline():
    if not BASELINE_FILE.exists():
        pytest.skip("no baseline file; run tests/tools/generate_baseline.py")
    tables = load_tables(BASELINE_FILE)
    skipped = set(NONDET_FILE.read_text().splitlines()) if NONDET_FILE.exists() else set()
    return tables, skipped


@pytest.fixture(scope="module")
def classes():
    return baseline_classes()


@pytest.fixture(scope="module")
def graphs():
    collection, names = fixture_collection()
    return dict(zip(names, collection.graphs))


def _fixture_names():
    _, names = fixture_collection()
    return names


def _comparable_mask(columns, skipped_stems):
    """Columns that are compared to the baseline: deterministic and well conditioned."""
    nondeterministic = nondeterministic_mask(columns, skipped_stems)
    ill_conditioned = np.array(
        [statistic_suffix(col) in ILL_CONDITIONED_STATISTICS for col in columns]
    )
    return ~nondeterministic & ~ill_conditioned


@pytest.mark.parametrize("name", _fixture_names())
def test_fixture_matches_baseline(name, baseline, classes, graphs):
    tables, skipped = baseline
    expected = tables[name]
    actual = compute_table(classes, graphs[name])

    missing = sorted(set(expected.index) - set(actual.index))
    added = sorted(set(actual.index) - set(expected.index))
    assert not missing and not added, f"columns missing: {missing[:10]}; added: {added[:10]}"

    actual = actual.reindex(expected.index)
    keep = _comparable_mask(expected.index, skipped)
    exp = expected[keep].to_numpy()
    act = actual[keep].to_numpy()
    cols = expected.index[keep]

    nan_changed = cols[np.isnan(exp) != np.isnan(act)]
    assert len(nan_changed) == 0, f"NaN pattern changed in: {list(nan_changed[:10])}"

    drifted = cols[~np.isclose(exp, act, rtol=RTOL, atol=ATOL, equal_nan=True)]
    report = [f"{c}: {expected[c]!r} -> {actual[c]!r}" for c in drifted[:10]]
    assert len(drifted) == 0, f"{len(drifted)} values drifted, e.g. " + "; ".join(report)
