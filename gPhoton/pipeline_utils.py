"""
temporary home for pipeline utilities to reduce need for expensive and maybe
incompatible imports.
"""

import numpy as np
from pyarrow import parquet


def get_parquet_stats(fn, columns, row_group=0):
    group = parquet.read_metadata(fn).row_group(row_group)
    statistics = {}
    for column in group.to_dict()["columns"]:
        if column["path_in_schema"] in columns:
            statistics[column["path_in_schema"]] = column["statistics"]
    return statistics


def table_values(table, columns):
    return np.array([table[column].to_numpy() for column in columns]).T
