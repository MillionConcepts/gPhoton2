"""
fast unit tests for contracts and logic that the end-to-end regression
tests do not reach (or reach only indirectly). these require no network
access except for the small metadata aspect table, which is downloaded
automatically on first need.
"""

from pathlib import Path

import pyarrow as pa
import pytest
from pyarrow import parquet

import gPhoton.aspect
from gPhoton.aspect import ASPECT_URL_BASE, aspect_table_path
from gPhoton.eclipse import eclipse_to_paths
from gPhoton.io.mast import get_raw_paths
from gPhoton.reference import check_eclipse


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (
            {},
            {
                "raw6": "e23456-nd-raw6.fits.gz",
                "photonfile": "e23456-nd-b00.parquet",
                "image": "e23456-nd-ffull-b00-image-g.fits.gz",
                "photomfile": "e23456-nd-ffull-b00-image-photom.csv",
            },
        ),
        (
            {"depth": 120, "compression": "rice", "aperture": 12.8},
            {
                "image": "e23456-nd-ffull-b00-image-r.fits",
                "movie": "e23456-nd-f0120-b00-movie-r.fits",
                "photomfile": "e23456-nd-f0120-b00-movie-photom-12_8.csv",
                "expfile": "e23456-nd-f0120-b00-movie-exptime.csv",
            },
        ),
        (
            {
                "band": "FUV", "leg": 3, "compression": "none",
                "aperture": 6.0, "suffix": "forced", "ftype": "parquet",
            },
            {
                "raw6": "e23456-fd-raw6.fits.gz",
                "photonfile": "e23456-fd-b03.parquet",
                "image": "e23456-fd-ffull-b03-image-n.fits",
                "photomfile":
                    "e23456-fd-ffull-b03-image-photom-6_0-forced.parquet",
            },
        ),
        (
            {"depth": 30, "start": 874361234.5},
            {
                "movie": "e23456-nd-f0030-b00-t8743612345-g.fits.gz",
                "photomfile": "e23456-nd-f0030-b00-t8743612345-photom.csv",
            },
        ),
    ],
)
def test_file_naming_contract(kwargs, expected):
    """
    output filenames are a contract with photometry_only runs, remote
    roots, and downstream users.
    """
    paths = eclipse_to_paths(23456, root="root", **kwargs)
    for key, name in expected.items():
        assert paths[key] == Path("root", "e23456", name)
    if kwargs.get("depth") is None:
        assert "movie" not in paths and "expfile" not in paths


def test_get_raw_paths(tmp_path):
    table = pa.table({
        "eclipse": [100, 100, 100, 200],
        "band": ["NUV ", "FUV ", "BOTH", "NUV "],
        "URL": ["n.fits.gz", "f.fits.gz", "scst.fits.gz", "n2.fits.gz"],
    })
    parquet.write_table(table, tmp_path / "raw_data_urls.parquet")
    assert get_raw_paths(100, aspect_dir=tmp_path) == {
        "NUV": "n.fits.gz", "FUV": "f.fits.gz", "scst": "scst.fits.gz",
    }
    assert get_raw_paths(200, aspect_dir=tmp_path) == {
        "NUV": "n2.fits.gz", "FUV": None, "scst": None,
    }
    assert get_raw_paths(300, aspect_dir=tmp_path) == {
        "NUV": None, "FUV": None, "scst": None,
    }


def test_aspect_table_auto_download(tmp_path, monkeypatch):
    calls = []

    def fake_download(url, destination, render_bar=True):
        calls.append(url)
        Path(destination).write_bytes(b"table")

    monkeypatch.setattr(gPhoton.aspect, "chunked_download", fake_download)
    # missing remote table is downloaded into aspect_dir
    path = aspect_table_path("boresight", tmp_path / "new")
    assert path == tmp_path / "new" / "boresight.parquet"
    assert path.read_bytes() == b"table"
    assert calls == [ASPECT_URL_BASE + "boresight.parquet"]
    # present table is not downloaded again
    aspect_table_path("boresight", tmp_path / "new")
    assert len(calls) == 1
    # tables with no remote copy are never fetched
    assert not aspect_table_path("aspect2", tmp_path).exists()
    assert len(calls) == 1

    # failed downloads leave nothing behind
    def failing_download(url, destination, render_bar=True):
        Path(destination).write_bytes(b"partial")
        raise ConnectionError("simulated failure")

    monkeypatch.setattr(gPhoton.aspect, "chunked_download", failing_download)
    with pytest.raises(ConnectionError):
        aspect_table_path("metadata", tmp_path)
    assert list(tmp_path.glob("metadata*")) == []
    assert list(tmp_path.glob("*.part")) == []


@pytest.mark.parametrize(
    "eclipse, has_error, post_csp",
    [
        (37422, False, False),  # last pre-CSP eclipse
        (37466, True, None),    # post-CSP, pre-TAC switch: refused
        (38150, False, True),   # first post-TAC-switch eclipse
    ],
)
def test_check_eclipse_gating(eclipse, has_error, post_csp):
    e_warn, e_error, is_post_csp = check_eclipse(eclipse)
    assert bool(e_error) is has_error
    if post_csp is not None:
        assert is_post_csp is post_csp


def test_check_eclipse_missing_metadata():
    # execute_pipeline unpacks 3 values, including when metadata is missing
    e_warn, e_error, post_csp = check_eclipse(1)
    assert e_error == ["No metadata found for e1."]
    assert post_csp is False


# ---- science-critical calculations ----------------------------------------
# the end-to-end tests detect changes in these, but cannot say which
# calculation changed or whether the new values are wrong.

import numpy as np
import pandas as pd
from astropy.wcs import WCS
from photutils.aperture import CircularAperture

from gPhoton.calibrate import compute_exptime, compute_shutter
from gPhoton.coords.gnomonic import gnomfwd_simple, gnomrev_simple
from gPhoton.lightcurve._steps import bitwise_aperture_photometry
from gPhoton.moviemaker._steps import make_frame
from gPhoton.tests.fingerprint import (
    compare_fingerprints, fingerprint_csv, fingerprint_photonlist
)

# detector-to-sky parameters as passed by photonpipe and aspect: xi / eta in
# 0.1-arcsecond units, boresight at the origin
ASPSCALE = 1 / 36000


@pytest.mark.parametrize(
    "ra0, dec0, roll",
    [(323.5, -2.0, 37.0), (0.05, 30.0, 200.0), (180.0, 85.0, -90.0)],
)
def test_gnomonic_matches_astropy_tan_and_round_trips(ra0, dec0, roll):
    rng = np.random.default_rng(0)
    n = 500
    # +/- 0.67 degrees: the full GALEX field of view
    xi, eta = rng.uniform(-24000, 24000, (2, n))
    ra0s, dec0s, crota = np.full(n, ra0), np.full(n, dec0), np.full(n, -roll)
    ra, dec = gnomrev_simple(xi, eta, ra0s, dec0s, crota, ASPSCALE, 0., 0.)
    # equivalent FITS TAN projection: RA increases to -xi, rotation by roll
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [ra0, dec0]
    wcs.wcs.crpix = [1, 1]
    wcs.wcs.cdelt = [-ASPSCALE, ASPSCALE]
    theta = np.radians(roll)
    wcs.wcs.pc = [
        [np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]
    ]
    ra_ref, dec_ref = wcs.wcs_pix2world(xi, eta, 0)
    dra = (ra - ra_ref + 180) % 360 - 180
    assert np.all((ra >= 0) & (ra < 360))
    assert np.abs(dra * np.cos(np.radians(dec))).max() * 3600 < 1e-4
    assert np.abs(dec - dec_ref).max() * 3600 < 1e-4
    xi_back, eta_back = gnomfwd_simple(ra, dec, ra0s, dec0s, crota, ASPSCALE, 0.)
    assert np.allclose(xi_back, xi, atol=1e-4)
    assert np.allclose(eta_back, eta, atol=1e-4)


def test_compute_shutter_counts_only_gaps_between_unflagged_events():
    t = np.array([0.0, 0.01, 0.02, 1.02, 1.03, 1.04])
    flags = np.zeros(6, dtype=int)
    # 1 s gap between events, plus 0.96 s between last event and trange end
    assert compute_shutter(t, flags, (0.0, 2.0)) == pytest.approx(1.96)
    # gaps shorter than shutgap (0.05 s) are not shutter
    assert compute_shutter(t[:3], flags[:3], (0.0, 0.04)) == 0
    # flagged events do not close a gap
    flags[3:5] = 7
    assert compute_shutter(t, flags, (0.0, 2.0)) == pytest.approx(1.98)


@pytest.mark.parametrize("band", ["NUV", "FUV"])
def test_compute_exptime_dead_time_model(band):
    # 10 s of unflagged events at 1000 cps, a 2 s shutter gap, and flagged
    # events that must not count toward the global count rate
    t = np.concatenate([np.arange(0, 5, 0.001), np.arange(7, 12, 0.001)])
    flags = np.zeros(len(t), dtype=int)
    t = np.concatenate([t, np.linspace(5.1, 6.9, 500)])
    flags = np.concatenate([flags, np.full(500, 7)])
    slope, intercept = {
        "NUV": (-0.000434730599193, 77.217817988),
        "FUV": (-0.000408075976406, 76.3000943221),
    }[band]
    # shutter: 4.999 -> 7.0 (11.999 -> 12.0 is shorter than shutgap)
    rawexpt = 12.0 - 2.001
    gcr = 10000 / rawexpt
    # dead time fraction = -slope * gcr / intercept
    expected = rawexpt * (1 + slope * gcr / intercept)
    exptime = compute_exptime(t, flags, band, (0.0, 12.0))
    assert exptime == pytest.approx(expected, rel=1e-9)
    # sanity: ~0.55% dead time at 1000 cps
    assert 0.994 * rawexpt < exptime < 0.995 * rawexpt


def test_compute_exptime_fully_shuttered_interval():
    t, flags = np.array([0.0, 10.0]), np.array([0, 0])
    assert compute_exptime(t, flags, "NUV", (0.0, 10.0)) == 0


def test_make_frame_bins_fits_pixel_coordinates():
    # foc columns are 1-based FITS (x, y); pixel n spans [n - 0.5, n + 0.5]
    foc = np.array([
        [3.0, 7.0],    # -> row 6, column 2
        [0.5, 0.5],    # lower edge of the first pixel
        [10.5, 10.5],  # upper edge of the last pixel
        [0.4, 5.0],    # off the image
        [5.0, 10.6],   # off the image
    ])
    weights = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    frame = make_frame(foc, weights, "sum", (10, 10))
    assert frame.dtype == np.float32
    assert frame[6, 2] == 1 and frame[0, 0] == 2 and frame[9, 9] == 4
    # weights are conserved for every photon on the image, and only those
    assert frame.sum() == 7


@pytest.mark.xfail(
    strict=True,
    reason="make_frame bounds both axes by min(imsz), dropping photons from "
           "the longer axis of non-square images (e.g. 56% of photons in "
           "e43337 leg 2, 4964 x 3331 pixels)",
)
def test_make_frame_non_square_image():
    # 10 rows x 20 columns; photon at x = 15, y = 3
    frame = make_frame(np.array([[15.0, 3.0]]), np.array([1.0]), "sum", (10, 20))
    assert frame[2, 14] == 1


def test_bitwise_aperture_photometry_flags_bits_inside_apertures():
    artifacts = np.zeros((50, 50), dtype=np.uint8)
    artifacts[10, 12] = 0b0001    # row 10, column 12: in aperture 0
    artifacts[30, 30] = 0b0100    # in aperture 1
    artifacts[31, 30] |= 0b1000   # also in aperture 1
    artifacts[45, 45] = 0b0010    # in no aperture
    apertures = CircularAperture([(12, 10), (30, 30), (10, 40)], r=3)
    flags = bitwise_aperture_photometry(artifacts, apertures)
    assert flags.dtype == np.uint8
    assert flags.tolist() == [0b0001, 0b1100, 0]


# ---- end-to-end comparison machinery ---------------------------------------


@pytest.fixture
def photonlist_and_photometry(tmp_path):
    rng = np.random.default_rng(1)
    n = 1000
    photons = pd.DataFrame({
        "t": np.sort(874361243.0 + rng.uniform(0, 1700, n)),
        "col": rng.uniform(0, 800, n).astype(np.float32),
        "ra": 323.5 + rng.uniform(-0.6, 0.6, n),
        "dec": -2.0 + rng.uniform(-0.6, 0.6, n),
        "flags": rng.choice([0, 7, 12], n).astype(np.int8),
    })
    sources = pd.DataFrame({
        "ra": 323.5 + rng.uniform(-0.6, 0.6, 50),
        "dec": -2.0 + rng.uniform(-0.6, 0.6, 50),
        "aperture_sum": rng.uniform(10, 1e4, 50),
        "t0_0": np.full(50, 874361243.0),
        "extended_source": rng.choice([True, False], 50),
    })
    photons.to_parquet(tmp_path / "photons.parquet")
    sources.to_csv(tmp_path / "photom.csv", index=False)
    return tmp_path, photons, sources


def _fingerprints(path):
    return {
        "photonlist": {"p": fingerprint_photonlist(path / "photons.parquet")},
        "photometry": {"p": fingerprint_csv(path / "photom.csv")},
    }


def _perturbed(path, photons, sources, which, column, func):
    photons, sources = photons.copy(), sources.copy()
    frame = photons if which == "photonlist" else sources
    frame[column] = func(frame[column].to_numpy())
    photons.to_parquet(path / "photons.parquet")
    sources.to_csv(path / "photom.csv", index=False)
    return _fingerprints(path)


@pytest.mark.parametrize(
    "which, column, func, detected",
    [
        # 0.01 s shift in absolute time: invisible to a 1e-6 relative check
        ("photonlist", "t", lambda t: t + 0.01, True),
        ("photonlist", "t", lambda t: t + 1e-6, False),
        # 0.1 arcsec shift in RA: likewise
        ("photonlist", "ra", lambda ra: ra + 0.1 / 3600, True),
        ("photonlist", "ra", lambda ra: ra + 1e-4 / 3600, False),
        # sorted columns from different rows: aggregates are unchanged
        ("photonlist", "col", lambda col: np.sort(col), True),
        ("photometry", "aperture_sum", lambda s: np.roll(s, 1), True),
        ("photometry", "t0_0", lambda t: t + 0.01, True),
        ("photometry", "aperture_sum", lambda s: s * (1 + 1e-9), False),
    ],
)
def test_fingerprint_sensitivity(
    photonlist_and_photometry, which, column, func, detected
):
    path, photons, sources = photonlist_and_photometry
    golden = _fingerprints(path)
    assert compare_fingerprints(golden, golden) == []
    diffs = compare_fingerprints(
        golden, _perturbed(path, photons, sources, which, column, func)
    )
    assert bool(diffs) is detected, diffs
    if detected:
        assert all(d.startswith(f"{which}/p/{column}") or
                   f"/sample/{column}" in d for d in diffs), diffs
