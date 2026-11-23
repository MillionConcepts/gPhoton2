"""
compact, tolerance-comparable summaries ("fingerprints") of pipeline output
products, used by the end-to-end regression tests. fingerprints are small
enough to commit to the repository as golden baselines.
"""

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
from astropy.io import fits
from pyarrow import parquet

# header keywords worth tracking in image / movie HDUs: WCS, exposure, time
# ranges, and photon count
HEADER_KEY_PATTERN = re.compile(
    r"^(NAXIS\d|CTYPE\d|CRPIX\d|CRVAL\d|CDELT\d|PC\d_\d|EXPT_\d+|EXPTIME"
    r"|T[01]_\d+|PHOTCNT|BAND|ECLIPSE|LEG)$"
)

# order in which product types are produced by the pipeline; used to report
# the earliest diverging stage
STAGE_ORDER = ("photonlist", "image", "movie", "photometry", "other")

GOLDEN_DIR = Path(__file__).parent / "golden"

# all products are bit-for-bit reproducible (RICE_1 quantization included,
# provided the dither seed is fixed), so tolerances only need to absorb
# floating-point differences between platforms and library versions.
RTOL = 1e-6
ATOL = 1e-6

# quantities that carry a large offset: absolute times and sky coordinates.
# a relative tolerance on these is far too loose (1e-6 of a GALEX time is
# ~870 s; of an RA near 320 deg, ~1.2 arcsec), so they are compared with
# absolute tolerances in their own units, and are never summed.
TIME_ATOL = 1e-4  # seconds
COORD_ATOL = 0.01 / 3600  # degrees
ABSOLUTE_TOLERANCES = (
    (re.compile(r"^(t|t[01]_\d+|T[01]_\d+)$"), TIME_ATOL),
    (re.compile(r"^(ra|dec|CRVAL\d)$"), COORD_ATOL),
)

# number of rows recorded verbatim from each table. aggregate statistics
# cannot see values that move between rows (e.g. photometry attached to the
# wrong source, or photon times misaligned with positions); these can.
SAMPLE_SIZE = 16


def absolute_tolerance(name: str) -> float | None:
    for pattern, atol in ABSOLUTE_TOLERANCES:
        if pattern.match(name):
            return atol
    return None


def sample_indices(n: int) -> list[int]:
    """evenly spaced row indices, including the first and last row"""
    if n == 0:
        return []
    return sorted(set(np.linspace(0, n - 1, SAMPLE_SIZE).round().astype(int)))


def product_stage(path: Path) -> str:
    name = path.name
    if "-photom" in name:
        return "photometry"
    if name.endswith(".parquet"):
        return "photonlist"
    if "-image" in name:
        return "image"
    if "-movie" in name:
        return "movie"
    return "other"


def _num(value):
    """convert numpy / pyarrow scalars to JSON-friendly python values"""
    if value is None:
        return None
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return None if math.isnan(value) else value
    return value


def fingerprint_photonlist(path: Path) -> dict:
    table = parquet.read_table(path)
    out = {"rows": table.num_rows, "columns": table.column_names}
    for name in table.column_names:
        col = table[name]
        stats = {"nulls": col.null_count}
        if name == "flags":
            counts = pc.value_counts(col).to_pylist()
            stats["counts"] = {
                str(c["values"]): c["counts"]
                for c in sorted(counts, key=lambda c: c["values"])
            }
        col = pc.cast(col, "float64")
        # NaN and null are both possible in float columns
        valid = pc.filter(col, pc.invert(pc.is_nan(col)))
        stats["nans"] = len(col) - col.null_count - len(valid)
        if len(valid) > 0:
            if absolute_tolerance(name) is None:
                stats["sum"] = _num(pc.sum(valid))
            stats |= {
                "mean": _num(pc.mean(valid)),
                "std": _num(pc.stddev(valid)),
                "min": _num(pc.min(valid)),
                "max": _num(pc.max(valid)),
            }
        out[name] = stats
    # photonlist row order is deterministic
    ix = sample_indices(table.num_rows)
    sample = table.take(ix)
    out["sample"] = {"index": [int(i) for i in ix]} | {
        name: [_num(v) for v in sample[name].to_pylist()]
        for name in table.column_names
    }
    return out


def _array_stats(data: np.ndarray) -> dict:
    data = np.asarray(data)
    finite = data[np.isfinite(data)] if data.dtype.kind == "f" else data
    stats = {
        "shape": list(data.shape),
        "dtype": data.dtype.str.lstrip("<>=|"),
        "sum": _num(finite.astype(np.float64).sum()),
        "nonzero": int(np.count_nonzero(finite)),
        "max": _num(finite.max()) if finite.size else None,
    }
    if data.ndim == 3:
        stats["frame_sums"] = [
            _num(np.nansum(frame.astype(np.float64))) for frame in data
        ]
    return stats


def fingerprint_fits(path: Path) -> dict:
    out = {}
    with fits.open(path) as hdul:
        for ix, hdu in enumerate(hdul):
            if hdu.data is None:
                continue
            name = hdu.name or str(ix)
            header = {
                k: _num(v) for k, v in hdu.header.items()
                if HEADER_KEY_PATTERN.match(k)
            }
            out[f"{ix}:{name}"] = _array_stats(hdu.data) | {"header": header}
    return out


def fingerprint_csv(path: Path) -> dict:
    table = pd.read_csv(path)
    out = {"rows": len(table), "columns": list(table.columns)}
    numeric = []
    for name in table.columns:
        col = table[name]
        if not pd.api.types.is_numeric_dtype(col):
            out[name] = {"nulls": int(col.isna().sum())}
            continue
        numeric.append(name)
        col = col.astype(np.float64)
        stats = {"nans": int(col.isna().sum())}
        if absolute_tolerance(name) is None:
            stats["sum"] = _num(col.sum())
        else:
            stats |= {"min": _num(col.min()), "max": _num(col.max())}
        out[name] = stats | {"median": _num(col.median())}
    # sample sources evenly spaced in declination, so that the sample is
    # tied to sky positions rather than to row order
    if {"ra", "dec"}.issubset(table.columns):
        order = table.sort_values(["dec", "ra"], kind="mergesort").index
    else:
        order = table.index
    picks = order[sample_indices(len(order))]
    out["sample"] = {"index": [int(i) for i in picks]} | {
        name: [_num(v) for v in table.loc[picks, name]] for name in numeric
    }
    return out


def fingerprint_product(path: Path) -> dict:
    name = path.name
    if ".fits" in name:
        return fingerprint_fits(path)
    if name.endswith(".csv"):
        return fingerprint_csv(path)
    if name.endswith(".parquet"):
        return fingerprint_photonlist(path)
    raise ValueError(f"don't know how to fingerprint {path}")


def fingerprint_products(paths) -> dict:
    """fingerprint a collection of products, keyed by stage, then filename"""
    out = {stage: {} for stage in STAGE_ORDER}
    for path in sorted(map(Path, paths)):
        out[product_stage(path)][path.name] = fingerprint_product(path)
    return {stage: v for stage, v in out.items() if v}


def _compare(
    expected, actual, where: str, diffs: list[str], atol: float | None = None
):
    """
    atol, when set, is the absolute tolerance of the quantity being compared
    (see ABSOLUTE_TOLERANCES); otherwise floats are compared with RTOL.
    """
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            if key not in actual:
                diffs.append(f"{where}/{key}: missing")
            elif key not in expected:
                diffs.append(f"{where}/{key}: unexpected")
            else:
                key_atol = absolute_tolerance(key)
                _compare(
                    expected[key],
                    actual[key],
                    f"{where}/{key}",
                    diffs,
                    atol if key_atol is None else key_atol,
                )
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            diffs.append(
                f"{where}: length {len(actual)} != expected {len(expected)}"
            )
            return
        for ix, (e, a) in enumerate(zip(expected, actual)):
            _compare(e, a, f"{where}[{ix}]", diffs, atol)
    elif isinstance(expected, float) or isinstance(actual, float):
        if atol is None:
            rel_tol, abs_tol = RTOL, ATOL
        else:
            rel_tol, abs_tol = 0, atol
        if expected is None or actual is None or not math.isclose(
            expected, actual, rel_tol=rel_tol, abs_tol=abs_tol
        ):
            diffs.append(f"{where}: {actual!r} != expected {expected!r}")
    elif expected != actual:
        diffs.append(f"{where}: {actual!r} != expected {expected!r}")


def compare_fingerprints(expected: dict, actual: dict) -> list[str]:
    """
    return a list of human-readable differences, earliest pipeline stage
    first. an empty list means the fingerprints match.
    """
    diffs = []
    for key in ("return_code", *STAGE_ORDER):
        if key in expected or key in actual:
            _compare(expected.get(key), actual.get(key), key, diffs)
    return diffs


def load_golden(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def write_golden(path: Path, fingerprint: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(fingerprint, f, indent=1, sort_keys=False)
        f.write("\n")
