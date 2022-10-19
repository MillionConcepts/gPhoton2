"""
profiling utilities, not guaranteed to be stable or functional. We do not
recommend their use for any purpose whatsoever.
"""
import gc

# import objgraph
from time import time
from typing import Mapping

# from line_profiler import LineProfiler
from pympler import muppy
from pympler.asizeof import asizeof
from pympler.process import _ProcessMemoryInfoProc


# def make_big_ref_graph(how_many: int, callback: Callable = get_size_and_id):
#     biggest = get_biggest_vars(how_many)
#     objgraph.show_backrefs(biggest, extra_info=callback, filename="back.dot")
#     objgraph.show_refs(biggest, extra_info=callback, filename="ref.dot")


def get_biggest_vars(how_many: int):
    return muppy.sort(muppy.get_objects())[-how_many:]


def get_size_and_id(obj):
    return f"{asizeof(obj)/1024**2} {id(obj)}"


def pm(message=""):
    print(f"{message} {round(_ProcessMemoryInfoProc().rss/1024**3, 2)}")


# LP = LineProfiler()


def filter_ipython_history(item):
    if not isinstance(item, Mapping):
        return True
    if item.get("__name__") == "__main__":
        return False
    if "_i" not in item.keys():
        return True
    return False


def print_referents(obj, filter_literal=True, filter_ipython=True):
    return print_references(
        obj, gc.get_referents, filter_literal, filter_ipython
    )


def print_referrers(obj, filter_literal=True, filter_ipython=True):
    return print_references(
        obj, gc.get_referrers, filter_literal, filter_ipython
    )


def print_references(obj, method, filter_literal=True, filter_ipython=True):
    refs = method(obj)
    if filter_literal is True:
        refs = tuple(
            filter(lambda ref: not isinstance(ref, (float, str)), refs)
        )
    if filter_ipython is True:
        refs = tuple(filter(filter_ipython_history, refs))
    extra_printables = [
        None if not isinstance(ref, tuple) else ref[0] for ref in refs
    ]
    for ref, extra in zip(refs, extra_printables):
        print(id(ref), type(ref), id(extra), type(extra))
    return refs


def di(obj_id):
    import _ctypes

    return _ctypes.PyObj_FromPtr(obj_id)
