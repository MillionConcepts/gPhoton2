import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]


def pytest_addoption(parser):
    parser.addoption(
        "--regen-golden",
        action="store_true",
        default=False,
        help="write end-to-end regression baselines instead of comparing "
             "against them",
    )


@pytest.fixture(scope="session")
def regen_golden(request) -> bool:
    return request.config.getoption("--regen-golden")


@pytest.fixture(scope="session")
def raw6_cache() -> Path:
    """
    directory in which raw6 inputs for end-to-end tests are cached, laid out
    like a pipeline data root (eNNNNN/eNNNNN-nd-raw6.fits.gz). set with the
    GPHOTON_TEST_DATA environment variable; defaults to <repo>/test_data.
    """
    return Path(os.environ.get("GPHOTON_TEST_DATA", REPO_ROOT / "test_data"))


@pytest.fixture(scope="session")
def stage_raw6(raw6_cache, tmp_path_factory):
    """
    return a function that makes a fresh pipeline data root containing only
    the raw6 file for (eclipse, band), downloading it to the cache first if
    necessary.
    """
    from gPhoton.eclipse import eclipse_to_paths
    from gPhoton.io.mast import retrieve_raw6

    def stage(eclipse: int, band: str, name: str) -> Path:
        cached = eclipse_to_paths(eclipse, band=band, root=raw6_cache)["raw6"]
        if not cached.exists():
            cached.parent.mkdir(parents=True, exist_ok=True)
            retrieve_raw6(eclipse, band, cached)
        root = tmp_path_factory.mktemp(name)
        staged = eclipse_to_paths(eclipse, band=band, root=root)["raw6"]
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.symlink_to(cached.resolve())
        return root

    return stage
