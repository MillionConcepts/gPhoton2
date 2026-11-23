"""fast, network-free unit tests"""
import numpy as np
import pandas as pd
import pytest

from gPhoton.analysis.utils import counts2mag, mag2counts
from gPhoton.aspect import ALL_ASPECT_TABLES, ASPECT_TABLE_TYPE
from gPhoton.eclipse import photomfile_path
from gPhoton.lightcurve._steps import load_source_catalog


def _catalog(tmp_path, fmt, with_eclipse=True):
    df = pd.DataFrame(
        {"ra": [1.0, 1.0, 2.0, 3.0], "dec": [4.0, 4.0, 5.0, 6.0],
         "eclipse": [100, 100, 100, 200]}
    )
    if not with_eclipse:
        df = df[["ra", "dec"]]
    path = tmp_path / f"cat.{fmt}"
    if fmt == "csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path)
    return path


@pytest.mark.parametrize("fmt", ["csv", "parquet"])
def test_catalog_filters_by_eclipse_and_dedupes(tmp_path, fmt):
    result = load_source_catalog(_catalog(tmp_path, fmt), 100)
    assert list(result.columns) == ["ra", "dec"]
    assert result["ra"].tolist() == [1.0, 2.0]


@pytest.mark.parametrize("fmt", ["csv", "parquet"])
def test_catalog_without_eclipse_column_uses_all_rows(tmp_path, fmt):
    result = load_source_catalog(_catalog(tmp_path, fmt, False), 999)
    assert result["ra"].tolist() == [1.0, 2.0, 3.0]


def test_catalog_bad_extension(tmp_path):
    with pytest.raises(ValueError):
        load_source_catalog(tmp_path / "cat.txt", 100)


@pytest.mark.parametrize("band", ["NUV", "FUV"])
def test_mag_counts_roundtrip(band):
    cps = np.array([0.1, 1.0, 50.0])
    assert np.allclose(mag2counts(counts2mag(cps, band), band), cps)


def test_aspect_table_names():
    assert "exposure_rgns" not in ALL_ASPECT_TABLES
    assert "leg-aperture" in ALL_ASPECT_TABLES
    assert "leg-aperture" in ASPECT_TABLE_TYPE.__args__


def test_photomfile_path():
    path = photomfile_path(
        23456, 0, "NUV", "direct", depth=30, aperture=12.8
    )
    assert path.as_posix() == "e23456/e23456-nd-f0030-b00-movie-photom-12_8.csv"
    assert photomfile_path(
        23456, 0, "NUV", "direct", depth=30, aperture=12.8, ftype="parquet"
    ).suffix == ".parquet"


def test_check_eclipse_unsupported_returns_three_values(monkeypatch):
    import gPhoton.aspect as aspect
    from gPhoton.reference import check_eclipse

    class _Empty(list):
        pass

    monkeypatch.setattr(aspect, "aspect_tables", lambda **kw: [[]])
    warns, errors, post_csp = check_eclipse(23456)
    assert len(errors) == 1 and "No metadata" in errors[0]
    assert post_csp is False
    assert check_eclipse(50000)[1][0].startswith("CAUSE")


def test_aspect_tables_without_filters(tmp_path):
    from gPhoton.aspect import aspect_tables

    pd.DataFrame({"eclipse": [1, 2]}).to_parquet(tmp_path / "metadata.parquet")
    (table,) = aspect_tables(None, tables="metadata", aspect_dir=tmp_path)
    assert table.num_rows == 2
    (table,) = aspect_tables(2, tables="metadata", aspect_dir=tmp_path)
    assert table.num_rows == 1
