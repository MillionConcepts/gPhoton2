# import objgraph
from time import time

from line_profiler import LineProfiler
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


def pm():
    print(_ProcessMemoryInfoProc().rss/1024**3)


LP = LineProfiler()


