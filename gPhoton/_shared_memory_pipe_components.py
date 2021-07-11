from multiprocessing.shared_memory import SharedMemory

import numpy as np

from gPhoton.MCUtils import NestingDict


def get_arrays_from_children(chunk_indices, results):
    array_dicts = []
    data_blocks = []
    for chunk_ix in chunk_indices:
        blocks, chunk = get_arrays_from_shared_memory(results[chunk_ix])
        array_dicts.append(chunk)
        data_blocks.append(blocks)
    return data_blocks, array_dicts


def unlink_cal_blocks(cal_data):
    all_cal_blocks = []
    for cal_name, cal_info in cal_data.items():
        cal_blocks, _ = get_arrays_from_shared_memory(cal_info)
        all_cal_blocks += list(cal_blocks.values())
    for block in all_cal_blocks:
        block.close()
        block.unlink()


def get_arrays_from_shared_memory(block_info) -> tuple[dict, dict]:
    blocks = {
        variable: SharedMemory(name=info["name"])
        for variable, info in block_info.items()
    }
    chunk = {
        variable: np.ndarray(
            info["shape"], dtype=info["dtype"], buffer=blocks[variable].buf
        )
        for variable, info in block_info.items()
    }
    return blocks, chunk


def send_to_shared_memory(array_dict):
    block_info = NestingDict()
    for name, array in array_dict.items():
        block = SharedMemory(create=True, size=array.size * array.itemsize)
        shared_array = np.ndarray(
            array.shape, dtype=array.dtype, buffer=block.buf
        )
        shared_array[:] = array[:]
        block_info[name]["name"] = block.name
        block_info[name]["dtype"] = array.dtype
        block_info[name]["shape"] = array.shape
        block.close()
    return block_info


def slice_into_shared_chunks(chunksz, data, nphots):
    variable_names = [key for key in data.keys()]
    chunk_slices = make_chunk_slices(chunksz, nphots)
    total_chunks = len(chunk_slices)
    block_directory = {}
    for chunk_ix in range(total_chunks):
        block_directory = slice_chunk_into_memory(
            block_directory, chunk_ix, data, chunk_slices, variable_names
        )
    return block_directory


def make_chunk_slices(chunksz, nphots):
    table_indices = []
    total_chunks = range(int(nphots / chunksz) + 1)
    for chunk_ix in total_chunks:
        chunkbeg, chunkend = chunk_ix * chunksz, (chunk_ix + 1) * chunksz
        if chunkend > nphots:
            chunkend = None
        table_indices.append((chunkbeg, chunkend))
    return table_indices


def slice_chunk_into_memory(
    block_directory, chunk_ix, data, table_indices, variable_names
):
    arrays = [
        array[slice(*table_indices[chunk_ix])] for array in data.values()
    ]
    block_info = send_to_shared_memory(
        {
            variable_name: array
            for variable_name, array in zip(variable_names, arrays)
        }
    )
    block_directory[chunk_ix] = block_info
    return block_directory
