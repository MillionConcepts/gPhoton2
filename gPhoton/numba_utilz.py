"""
utilities for working with numba
"""

from collections.abc import Callable
from typing import Any, TypeVar, overload
from numba import jit as _jit

Clbl = TypeVar("Clbl", bound = Callable[..., Any])


@overload
def jit(f: Clbl, **kwargs: Any) -> Clbl: ...
@overload
def jit(**kwargs: Any) -> Callable[[Clbl], Clbl]: ...


def jit(f: Clbl | None = None, **kwargs: Any) -> Clbl | Callable[[Clbl], Clbl]:
    """
    Wrapper around numba.jit.  Forces nopython=True and forceobj=False
    (like njit) and cache=True.  Annotated to inform type checkers
    that the decorator preserves the signature of the decorated function.

    Does not support passing a numba type signature or list of
    signatures as the first positional argument.  Because numba
    itself doesn't have type annotations, it would be too difficult
    to write down the type of a numba type signature.

    (See https://github.com/numba/numba/issues/7424 re the need to
    inform type checkers that the decorator preserves signatures.)
    """
    if 'nopython' in kwargs:
        raise ValueError("numba_utilz.jit forces nopython=True")
    if 'forceobj' in kwargs:
        raise ValueError("numba_utilz.jit forces forceobj=False")
    if 'cache' in kwargs:
        raise ValueError("numba_utilz.jit forces cache=True")

    if f is None:
        return _jit(nopython=True, cache=True, **kwargs) # type: ignore
    else:
        return _jit(f, nopython=True, cache=True, **kwargs) # type: ignore
