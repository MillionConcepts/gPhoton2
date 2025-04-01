"""
utilities for skimming and reading parquet-format files
"""

import numpy as np
from pyarrow import parquet

from collections.abc import Collection
from typing import Any
from gPhoton.types import Pathlike, NDArray

def get_parquet_stats(
    fn: Pathlike,
    columns: Collection[str],
    row_group: int = 0,
) -> dict[str, dict[str, Any]]:
    group = parquet.read_metadata(fn).row_group(row_group)
    statistics = {}
    for column in group.to_dict()["columns"]:
        path = column["path_in_schema"]
        if path in columns:
            statistics[path] = column["statistics"]
    return statistics


def parquet_to_ndarray(
    table: parquet.Table,
    columns: Collection[str] | None = None
) -> NDArray[Any]:
    if columns is None:
        columns = table.column_names
    return np.array([table[column].to_numpy() for column in columns]).T


def parquet_to_ndarrays(
    table: parquet.Table,
    columns: Collection[str] | None = None
) -> dict[str, NDArray[Any]]:
    if columns is None:
        columns = table.column_names
    return {column: table[column].to_numpy() for column in columns}
