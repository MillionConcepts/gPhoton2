"""
utilities for working with numba
"""

from collections.abc import Callable
from typing import Any, TypeVar, cast, overload
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

    If you want to specify a numba type signature, you have to pass
    it using a hidden 'signatures=' keyword argument (which is not
    available in the normal numba.jit decorator), not as the first
    positional argument.  This is to work around limitations in mypy's
    handling of @overload.  Also, because numba itself doesn't have
    type annotations, it is not practical to write down the type of
    that argument.

    (See https://github.com/numba/numba/issues/7424 re the need to
    inform type checkers that the decorator preserves signatures.)
    """
    if 'nopython' in kwargs:
        raise ValueError("numba_utilz.jit forces nopython=True")
    kwargs['nopython'] = True

    if 'forceobj' in kwargs:
        raise ValueError("numba_utilz.jit forces forceobj=False")
    kwargs['forceobj'] = False

    if 'cache' in kwargs:
        raise ValueError("numba_utilz.jit forces cache=True")
    kwargs['cache'] = True

    signatures: Any = kwargs.pop('signatures', None)

    match (f, signatures):
        case (None, None):
            return cast(Callable[[Clbl], Clbl], _jit(**kwargs))
        case (None, sigs):
            return cast(Callable[[Clbl], Clbl], _jit(sigs, **kwargs))
        case (fn, None):
            return cast(Clbl, _jit(**kwargs)(fn))
        case (fn, sigs):
            return cast(Clbl, _jit(sigs, **kwargs)(fn))

    raise AssertionError("unreachable")
