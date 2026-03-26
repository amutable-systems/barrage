# SPDX-License-Identifier: MIT

"""
Assertion helpers for barrage tests.

Provides the same assertion quality as :class:`AsyncTestCase` methods
but as free functions, suitable for standalone test functions::

    import barrage.assertions as Assert

    async def test_something() -> None:
        Assert.eq(1 + 1, 2)
        Assert.in_("hello", items)
        with Assert.raises(ValueError):
            int("not a number")

The :class:`SkipTest` exception also lives here so that both
class-based and function-based tests can access it without circular
imports.

All functions raise :class:`AssertionError` with descriptive messages
on failure.
"""

from collections.abc import Container
from types import TracebackType
from typing import Any, Protocol, Self, TypeGuard, overload

# ===================================================================== #
#  Ordering protocol helpers (used by comparison assertions)
# ===================================================================== #


class _SupportsDunderGT[T_contra](Protocol):
    def __gt__(self, other: T_contra, /) -> object: ...


class _SupportsDunderGE[T_contra](Protocol):
    def __ge__(self, other: T_contra, /) -> object: ...


class _SupportsDunderLT[T_contra](Protocol):
    def __lt__(self, other: T_contra, /) -> object: ...


class _SupportsDunderLE[T_contra](Protocol):
    def __le__(self, other: T_contra, /) -> object: ...


# ===================================================================== #
#  Skip
# ===================================================================== #


class SkipTest(Exception):
    """Raised to skip a test."""

    def __init__(self, reason: str = "") -> None:
        super().__init__(reason)
        self.reason = reason


# ===================================================================== #
#  Core
# ===================================================================== #


def fail(msg: str | None = None) -> None:
    """Fail immediately with an optional message."""
    raise AssertionError(msg or "test failed explicitly")


def true(expr: object, msg: str | None = None) -> None:
    """Assert that *expr* is truthy."""
    if not expr:
        raise AssertionError(msg or f"expected truthy, got {expr!r}")


def false(expr: object, msg: str | None = None) -> None:
    """Assert that *expr* is falsy."""
    if expr:
        raise AssertionError(msg or f"expected falsy, got {expr!r}")


# ===================================================================== #
#  Equality / identity
# ===================================================================== #


def eq(first: object, second: object, msg: str | None = None) -> None:
    """Assert ``first == second``."""
    if first != second:
        raise AssertionError(msg or f"{first!r} != {second!r}")


def ne(first: object, second: object, msg: str | None = None) -> None:
    """Assert ``first != second``."""
    if first == second:
        raise AssertionError(msg or f"{first!r} == {second!r}")


def is_(first: object, second: object, msg: str | None = None) -> None:
    """Assert ``first is second``."""
    if first is not second:
        raise AssertionError(msg or f"{first!r} is not {second!r}")


def is_not(first: object, second: object, msg: str | None = None) -> None:
    """Assert ``first is not second``."""
    if first is second:
        raise AssertionError(msg or f"{first!r} is {second!r}")


def none(expr: object, msg: str | None = None) -> None:
    """Assert ``expr is None``."""
    if expr is not None:
        raise AssertionError(msg or f"expected None, got {expr!r}")


def not_none(expr: object, msg: str | None = None) -> None:
    """Assert ``expr is not None``."""
    if expr is None:
        raise AssertionError(msg or "expected non-None value")


# ===================================================================== #
#  Membership
# ===================================================================== #


def in_(member: object, container: Container[object], msg: str | None = None) -> None:
    """Assert ``member in container``."""
    if member not in container:
        raise AssertionError(msg or f"{member!r} not in {container!r}")


def not_in(member: object, container: Container[object], msg: str | None = None) -> None:
    """Assert ``member not in container``."""
    if member in container:
        raise AssertionError(msg or f"{member!r} unexpectedly in {container!r}")


# ===================================================================== #
#  Type checks
# ===================================================================== #


@overload
def isinstance_[T](obj: object, cls: type[T], msg: str | None = None) -> TypeGuard[T]: ...
@overload
def isinstance_[T](obj: object, cls: tuple[type[T], ...], msg: str | None = None) -> TypeGuard[T]: ...
def isinstance_(obj: object, cls: type | tuple[type, ...], msg: str | None = None) -> bool:
    """Assert ``isinstance(obj, cls)``."""
    if not isinstance(obj, cls):
        raise AssertionError(msg or f"{obj!r} is not an instance of {cls!r}")
    return True


def not_isinstance(obj: object, cls: type | tuple[type, ...], msg: str | None = None) -> None:
    """Assert ``not isinstance(obj, cls)``."""
    if isinstance(obj, cls):
        raise AssertionError(msg or f"{obj!r} is unexpectedly an instance of {cls!r}")


# ===================================================================== #
#  Ordering
# ===================================================================== #


@overload
def gt[T](first: _SupportsDunderGT[T], second: T, msg: str | None = None) -> None: ...
@overload
def gt[T](first: T, second: _SupportsDunderLT[T], msg: str | None = None) -> None: ...
def gt(first: Any, second: Any, msg: str | None = None) -> None:
    """Assert ``first > second``."""
    if not first > second:
        raise AssertionError(msg or f"{first!r} is not greater than {second!r}")


@overload
def ge[T](first: _SupportsDunderGE[T], second: T, msg: str | None = None) -> None: ...
@overload
def ge[T](first: T, second: _SupportsDunderLE[T], msg: str | None = None) -> None: ...
def ge(first: Any, second: Any, msg: str | None = None) -> None:
    """Assert ``first >= second``."""
    if not first >= second:
        raise AssertionError(msg or f"{first!r} is not greater than or equal to {second!r}")


@overload
def lt[T](first: _SupportsDunderLT[T], second: T, msg: str | None = None) -> None: ...
@overload
def lt[T](first: T, second: _SupportsDunderGT[T], msg: str | None = None) -> None: ...
def lt(first: Any, second: Any, msg: str | None = None) -> None:
    """Assert ``first < second``."""
    if not first < second:
        raise AssertionError(msg or f"{first!r} is not less than {second!r}")


@overload
def le[T](first: _SupportsDunderLE[T], second: T, msg: str | None = None) -> None: ...
@overload
def le[T](first: T, second: _SupportsDunderGE[T], msg: str | None = None) -> None: ...
def le(first: Any, second: Any, msg: str | None = None) -> None:
    """Assert ``first <= second``."""
    if not first <= second:
        raise AssertionError(msg or f"{first!r} is not less than or equal to {second!r}")


def almost_eq(first: float, second: float, places: int = 7, msg: str | None = None) -> None:
    """Assert that *first* and *second* are equal to *places* decimal places."""
    if round(abs(second - first), places) != 0:
        raise AssertionError(msg or f"{first!r} != {second!r} within {places} places")


# ===================================================================== #
#  Exception checking
# ===================================================================== #


class _RaisesContext:
    """Context manager returned by :func:`raises`."""

    def __init__(self, exc_type: type[BaseException]) -> None:
        self.exc_type = exc_type
        self.exception: BaseException | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, tb: TracebackType | None
    ) -> bool:
        if exc_type is None:
            raise AssertionError(f"{self.exc_type.__name__} not raised")
        if not issubclass(exc_type, self.exc_type):
            return False
        self.exception = exc_val
        return True


def raises(exc_type: type[BaseException]) -> _RaisesContext:
    """Assert that the ``with`` block raises *exc_type*.

    Usage::

        with Assert.raises(ValueError) as ctx:
            int("bad")
        Assert.in_("invalid literal", str(ctx.exception))
    """
    return _RaisesContext(exc_type)


def skip(reason: str = "") -> None:
    """Skip the current test with an optional *reason*."""
    raise SkipTest(reason)
