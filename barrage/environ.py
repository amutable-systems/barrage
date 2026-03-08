# SPDX-License-Identifier: MIT
"""
Per-test environment variable isolation via ``contextvars``.

Provides a :class:`MutableMapping` wrapper around ``os.environ`` that
delegates to a per-task snapshot when one is active.  Each test gets
its own copy of the environment at test start, so mutations are fully
isolated from other concurrent tests.

The pattern mirrors the stdout/stderr capture in ``barrage.runner``:
a ref-counted context manager installs the wrapper, and a per-test
context manager sets a fresh snapshot on a ``ContextVar``.
"""

import collections.abc
import contextlib
import contextvars
import os

_env_snapshot: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "_env_snapshot", default=None
)


class _ContextEnviron(collections.abc.MutableMapping[str, str]):
    """Drop-in replacement for ``os.environ`` that honours per-task snapshots.

    When a snapshot is active (set via :func:`isolated_environ`), all
    reads and writes go to the snapshot dict.  Otherwise everything
    falls through to the real :data:`os.environ`.
    """

    def __init__(self, original: os._Environ[str]) -> None:
        self._original = original

    def _store(self) -> collections.abc.MutableMapping[str, str]:
        snap = _env_snapshot.get()
        return snap if snap is not None else self._original

    def __getitem__(self, key: str) -> str:
        return self._store()[key]

    def __setitem__(self, key: str, value: str) -> None:
        self._store()[key] = value

    def __delitem__(self, key: str) -> None:
        del self._store()[key]

    def __iter__(self) -> collections.abc.Iterator[str]:
        return iter(self._store())

    def __len__(self) -> int:
        return len(self._store())

    def __contains__(self, key: object) -> bool:
        return key in self._store()

    def __repr__(self) -> str:
        return repr(dict(self._store()))


class _EnvironContext:
    """Ref-counted context manager that installs :class:`_ContextEnviron`.

    Replaces ``os.environ`` with the wrapper on first enter and
    restores the original on last exit.  Safe for nested runs
    (meta-tests).
    """

    _refcount: int = 0
    _saved: os._Environ[str] | None = None

    def __enter__(self) -> None:
        cls = type(self)
        cls._refcount += 1
        if cls._refcount == 1:
            original = os.environ
            cls._saved = original  # type: ignore[assignment]
            os.environ = _ContextEnviron(original)  # type: ignore[assignment]  # noqa: B003

    def __exit__(self, *args: object) -> None:
        cls = type(self)
        cls._refcount -= 1
        if cls._refcount == 0:
            if cls._saved is not None:
                os.environ = cls._saved  # type: ignore[assignment]  # noqa: B003
            cls._saved = None


@contextlib.contextmanager
def isolated_environ() -> collections.abc.Generator[None]:
    """Context manager that gives the current task its own environment copy."""
    snapshot = dict(os.environ)
    token = _env_snapshot.set(snapshot)
    try:
        yield
    finally:
        _env_snapshot.reset(token)
