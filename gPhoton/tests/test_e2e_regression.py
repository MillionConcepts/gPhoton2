"""
end-to-end regression tests. each test runs execute_pipeline on a real
eclipse, fingerprints every output product, and compares the fingerprints
against a committed baseline in gPhoton/tests/golden/. the scenarios are
chosen to exercise distinct critical branches of the pipeline; see
gPhoton/tests/README.md.

run with `pytest -m e2e`; regenerate baselines (after an intentional change
in output) with `pytest -m e2e --regen-golden`.
"""

from pathlib import Path

import pandas as pd
import pytest

from gPhoton.pipeline import execute_pipeline
from gPhoton.tests.fingerprint import (
    GOLDEN_DIR,
    compare_fingerprints,
    fingerprint_products,
    load_golden,
    write_golden,
)

pytestmark = pytest.mark.e2e

# RICE_1 quantization dithers with a clock-seeded random sequence by
# default; a fixed seed makes RICE-compressed products, and photometry
# measured from them, reproducible
RICE_KWARGS = {"dither_seed": 1}


def outputs(root: Path, eclipse: int, exclude=()) -> list[Path]:
    """pipeline products in root for eclipse, excluding raw6 inputs"""
    edir = root / f"e{eclipse:05}"
    return [
        p for p in edir.iterdir()
        if p.is_file() and "raw6" not in p.name and p.name not in exclude
    ]


def check_against_golden(
    name: str,
    return_code: str,
    products: list[Path],
    regen: bool,
):
    actual = {"return_code": return_code} | fingerprint_products(products)
    golden_path = GOLDEN_DIR / f"{name}.json"
    if regen:
        write_golden(golden_path, actual)
        return
    if not golden_path.exists():
        pytest.fail(
            f"no baseline at {golden_path}; run with --regen-golden to "
            f"create it"
        )
    diffs = compare_fingerprints(load_golden(golden_path), actual)
    assert not diffs, (
        f"{name}: {len(diffs)} difference(s) from baseline; first diverging "
        f"stage listed first:\n" + "\n".join(diffs[:50])
    )


@pytest.fixture(scope="module")
def e23456_nuv(stage_raw6):
    """E1: shared by the full-pipeline test and the photometry-only test"""
    root = stage_raw6(23456, "NUV", "e23456_nuv")
    return_code = execute_pipeline(
        23456,
        "NUV",
        depth=120,
        threads=4,
        local_root=str(root),
        recreate=True,
        aperture_sizes=[12.8],
        write={"movie": True, "image": True},
        compression="rice",
        hdu_constructor_kwargs=RICE_KWARGS,
        verbose=0,
    )
    return root, return_code, outputs(root, 23456)


def test_e23456_nuv(e23456_nuv, regen_golden):
    """
    E1: pre-CSP NUV, single leg. raw6 decoding, NUV calibration, aspect,
    flagging, image / movie generation, WCS, exposure time, coverage
    backplane, source finding, aperture photometry, RICE FITS output.
    """
    root, return_code, products = e23456_nuv
    check_against_golden("e23456_nuv", return_code, products, regen_golden)


def test_e23456_nuv_forced_photometry(e23456_nuv, regen_golden, tmp_path):
    """
    E2: photometry-only rerun on E1's images and movies with a forced source
    catalog. reading written FITS products back, catalog loading, forced
    photometry.
    """
    root, _, products = e23456_nuv
    photom = next(p for p in products if "-photom" in p.name)
    catalog = tmp_path / "catalog.csv"
    pd.read_csv(photom)[["ra", "dec"]].to_csv(catalog, index=False)
    return_code = execute_pipeline(
        23456,
        "NUV",
        depth=120,
        threads=4,
        local_root=str(root),
        aperture_sizes=[12.8],
        compression="rice",
        photometry_only=True,
        source_catalog_file=str(catalog),
        suffix="forced",
        verbose=0,
    )
    forced = [p for p in outputs(root, 23456) if "-forced" in p.name]
    check_against_golden("e23456_nuv_forced", return_code, forced, regen_golden)


def test_e23456_fuv(stage_raw6, regen_golden):
    """
    E3: pre-CSP FUV, single leg. FUV-specific calibration (detector
    temperature, FUV offset, FUV cal files).
    """
    root = stage_raw6(23456, "FUV", "e23456_fuv")
    return_code = execute_pipeline(
        23456,
        "FUV",
        depth=120,
        threads=4,
        local_root=str(root),
        recreate=True,
        aperture_sizes=[12.8],
        write={"movie": True, "image": True},
        compression="rice",
        hdu_constructor_kwargs=RICE_KWARGS,
        verbose=0,
    )
    check_against_golden(
        "e23456_fuv", return_code, outputs(root, 23456), regen_golden
    )


def test_e43337_nuv_post_csp_multileg(stage_raw6, regen_golden):
    """
    E4: post-CSP NUV AIS eclipse with 12 legs, images only. post-CSP
    calibration, leg splitting, per-leg boresight / coverage, gzip output.
    """
    root = stage_raw6(43337, "NUV", "e43337_nuv")
    return_code = execute_pipeline(
        43337,
        "NUV",
        depth=None,
        threads=4,
        local_root=str(root),
        recreate=True,
        aperture_sizes=[12.8],
        write={"movie": False, "image": True},
        compression="gzip",
        verbose=0,
    )
    check_against_golden(
        "e43337_nuv", return_code, outputs(root, 43337), regen_golden
    )
