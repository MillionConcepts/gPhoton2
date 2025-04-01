"""
.. module:: sharing
   :synopsis: Utilities for working with python shared memory structures.
"""

# TODO, maybe: consider moving this up to dustgoggles, or integrating
#  the `notetaking` code with this?

# TODO: You're supposed to call block.close() once for *every* process
# that accesses the block, but you're supposed to call block.unlink()
# once *only*.  The best way to make that happen is probably to introduce
# a SharedMemoryManager.

from itertools import pairwise
from multiprocessing.shared_memory import SharedMemory

import numpy as np

from collections.abc import Mapping
from gPhoton.types import NDArray
from typing import Any, NamedTuple


class SharedArrayInfo(NamedTuple):
    """
    Information required to reconstruct a numpy.ndarray from a
    byte buffer in shared memory.
    """
    #: The name of the shared buffer.
    name: str
    #: The dtype of the ndarray in the buffer.
    dtype: np.dtype[Any]
    #: The shape of the ndarray in the buffer.
    shape: tuple[int, ...]


def get_column_from_shared_memory(
    results: Mapping[str, Mapping[str, SharedArrayInfo]],
    column_name: str,
    unlink: bool = True
) -> NDArray[Any]:
    column_info = {
        chunk_ix: chunk[column_name]
        for chunk_ix, chunk in results.items()
    }
    blocks, column_slices = reference_shared_memory_arrays(column_info)
    column = np.hstack([slc for ix, slc in sorted(column_slices.items())])
    del column_slices
    if unlink:
        for block in blocks.values():
            block.close()
            block.unlink()
    return column


def unlink_nested_block_dict(
    cal_data: Mapping[Any, Mapping[Any, SharedArrayInfo] | None]
) -> None:
    for cal_info in cal_data.values():
        if cal_info is None:
            continue
        for info in cal_info.values():
            block = SharedMemory(name=info.name)
            block.close()
            block.unlink()


def reference_shared_memory_arrays(
    block_info: Mapping[str, SharedArrayInfo],
    fetch: bool = True,
) -> tuple[dict[str, SharedMemory], dict[str, NDArray[Any]]]:
    blocks: dict[str, SharedMemory] = {}
    chunk: dict[str, NDArray[Any]] = {}
    for variable, info in block_info.items():
        block = SharedMemory(name=info.name)
        blocks[variable] = block
        if fetch:
            chunk[variable] = np.ndarray(
                info.shape, info.dtype, block.buf
            )
    return blocks, chunk


def send_to_shared_memory(
    array_dict: Mapping[str, NDArray[Any]],
    slc: slice | None = None,
) -> dict[str, SharedArrayInfo]:
    block_info = {}
    if slc is None:
        slc = slice(None) # [:]
    for name, array in array_dict.items():
        sliced = array[slc] # no copy here
        block = SharedMemory(create=True, size=sliced.size * sliced.itemsize)
        shared_array: NDArray[Any] = np.ndarray(
            sliced.shape, dtype=sliced.dtype, buffer=block.buf
        )
        shared_array[:] = sliced # copy here
        block_info[name] = SharedArrayInfo(
            name = block.name,
            dtype = sliced.dtype,
            shape = sliced.shape,
        )
        block.close()
    return block_info


def send_mapping_to_shared_memory(
    mapping: Mapping[str, Mapping[str, NDArray[Any]]],
    slc: slice | None = None,
) -> dict[str, dict[str, SharedArrayInfo]]:
    return {
        key: send_to_shared_memory(value, slc)
        for key, value in mapping.items()
    }


def slice_into_shared_chunks(
    chunksz: int,
    data: Mapping[str, NDArray[Any]]
) -> dict[int, dict[str, SharedArrayInfo]]:
    data_len = len(next(iter(data.values())))
    chunk_slices = make_chunk_slices(chunksz, data_len)
    return {
        chunk_ix: send_to_shared_memory(data, slc)
        for chunk_ix, slc in enumerate(chunk_slices)
    }


def make_chunk_slices(chunksz: int, nphots: int) -> list[slice]:
    table_indices = [
        slice(a, b)
        for a, b in pairwise(range(0, nphots, chunksz))
    ]
    if table_indices:
        table_indices.append(slice(table_indices[-1].stop, None))
    else:
        table_indices.append(slice(None, None))
    return table_indices
