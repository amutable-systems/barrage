# SPDX-License-Identifier: MIT
import importlib
import importlib.util
import inspect
import os
import sys
import types
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any, Never, cast

from barrage.case import AsyncTestCase
from barrage.colorize import ANSI, should_colorize, style
from barrage.runner import AsyncTestSuite, _collect_test_methods


def die(message: str, exitcode: int = 2) -> Never:
    """Print *message* to stderr and raise ``SystemExit(exitcode)``."""
    if should_colorize(sys.stderr):
        message = style(message, ANSI.RED)
    print(message, file=sys.stderr)
    raise SystemExit(exitcode)


def resolve_tests(
    path_specs: list[str],
    top_level_dir: Path,
    pattern: str = "test_*.py",
) -> AsyncTestSuite:
    """
    Build an :class:`AsyncTestSuite` from a list of path specifications.

    Each *path_spec* is either a **path-based** or **name-based**
    selector.

    **Path-based** (the path part must be an existing file or
    directory)::

        path                         # directory or file
        path::ClassName              # specific class
        path::ClassName::method      # specific method
        path::function_name          # specific function

    **Name-based** (no path prefix — discovers from *top_level_dir*
    or the current directory and filters by name)::

        ClassName                    # all methods of that class
        ClassName::method            # specific method
        test_function_name           # standalone function
        test_method_name             # method (must be unique)

    Parameters
    ----------
    path_specs:
        One or more path specifications as described above.
    pattern:
        Glob pattern for test file names used during directory
        discovery (default ``test_*.py``).
    top_level_dir:
        Top-level directory of the project, used as the import root
        for discovered modules and as the search root for name-based
        specs.

    Returns
    -------
    AsyncTestSuite
        A suite containing the resolved test classes, methods, and
        functions.

    Raises
    ------
    SystemExit
        If a path does not exist, a class is not found, or a method
        is not found, an error message is printed to stderr and
        ``SystemExit(2)`` is raised.
    """
    suite = AsyncTestSuite()

    # Separate path-based specs from name-based specs.  A spec is
    # name-based only if the path part contains no slashes (i.e. it
    # looks like a bare identifier, not a file path).
    path_based: list[str] = []
    name_based: list[str] = []

    for spec in path_specs:
        path_str = spec.split("::")[0]
        if os.sep in path_str:
            path_based.append(spec)
        else:
            # Resolve against the top-level directory.
            p = top_level_dir / path_str
            if p.is_dir() or p.is_file():
                path_based.append(spec)
            else:
                name_based.append(spec)

    # ── process path-based specs ─────────────────────────────────
    for spec in path_based:
        parts = spec.split("::")
        path_str = parts[0]
        class_name: str | None = parts[1] if len(parts) > 1 else None
        method_name: str | None = parts[2] if len(parts) > 2 else None

        if len(parts) > 3:
            die(f"Error: invalid test spec {spec!r} – expected path[::Class[::method]]")

        p = (top_level_dir / path_str).resolve()

        if not p.exists():
            die(f"Error: path does not exist: {path_str!r}")

        resolved_top = top_level_dir.resolve()
        try:
            p.relative_to(resolved_top)
        except ValueError:
            die(f"Error: path {path_str!r} is not inside top-level directory {str(resolved_top)!r}")

        if p.is_dir():
            if class_name is not None:
                die(f"Error: cannot use ::ClassName filter on a directory: {spec!r}")
            _discover_directory(p, suite, pattern=pattern, top_level_dir=top_level_dir)
        else:
            _discover_file(
                p,
                suite,
                top_level_dir=top_level_dir,
                class_name=class_name,
                method_name=method_name,
            )

    # ── process name-based specs ─────────────────────────────────
    if name_based:
        full = AsyncTestSuite()
        _discover_directory(top_level_dir.resolve(), full, pattern=pattern, top_level_dir=top_level_dir)
        _resolve_name_specs(name_based, full, suite)

    return suite


# ===================================================================== #
#  Directory discovery
# ===================================================================== #


def discover(
    start_dir: Path,
    top_level_dir: Path,
    pattern: str = "test_*.py",
) -> AsyncTestSuite:
    """
    Discover ``AsyncTestCase`` subclasses by walking *start_dir* and
    importing every Python file whose name matches *pattern*.

    Parameters
    ----------
    start_dir:
        Directory to start searching from.
    top_level_dir:
        The top-level directory of the project, inserted into
        ``sys.path`` so that test modules can be imported by their
        dotted package path.
    pattern:
        Glob pattern for test file names (default ``test_*.py``).

    Returns
    -------
    AsyncTestSuite
        A suite containing every discovered test class and its methods.
    """
    suite = AsyncTestSuite()
    _discover_directory(
        start_dir,
        suite,
        pattern=pattern,
        top_level_dir=top_level_dir,
    )
    return suite


def _discover_directory(
    start: Path,
    suite: AsyncTestSuite,
    top_level_dir: Path,
    pattern: str = "test_*.py",
) -> None:
    """Walk *start* and add every matching test class to *suite*."""
    start = start.resolve()
    top = top_level_dir.resolve() if top_level_dir else start

    if str(top) not in sys.path:
        sys.path.insert(0, str(top))

    for root, _dirs, files in os.walk(start):
        root_path = Path(root)
        for filename in sorted(files):
            if not _matches_pattern(filename, pattern):
                continue
            filepath = root_path / filename
            try:
                module = _import_path(filepath, top)
            except Exception:
                continue
            for cls in _find_test_classes(module):
                methods = _collect_test_methods(cls)
                if methods:
                    suite.add_class(cls, methods)
            for func in _find_test_functions(module):
                suite.add_function(func)


# ===================================================================== #
#  Single-file discovery
# ===================================================================== #


def _discover_file(
    filepath: Path,
    suite: AsyncTestSuite,
    top_level_dir: Path,
    class_name: str | None = None,
    method_name: str | None = None,
) -> None:
    """Import *filepath* and add matching test classes/methods to *suite*.

    Parameters
    ----------
    filepath:
        Path to a ``.py`` test file.
    suite:
        The suite to add discovered entries to.
    class_name:
        If given, only the class with this name is collected.
    method_name:
        If given (requires *class_name*), only this method is collected.
    top_level_dir:
        Top-level directory used as the import root.  When ``None``,
        the current working directory is used.
    """
    filepath = filepath.resolve()

    if str(top_level_dir) not in sys.path:
        sys.path.insert(0, str(top_level_dir))

    try:
        module = _import_path(filepath, top_level_dir)
    except Exception as e:
        die(f"Error: failed to import {filepath}: {e}")

    if class_name is not None:
        # Look for a specific class.
        cls = _find_named_class(module, class_name)
        if cls is None:
            # Not a class — try as a standalone function name.
            # Only valid when no method_name is given (functions don't
            # have sub-components).
            if method_name is None:
                func = _find_named_function(module, class_name)
                if func is not None:
                    suite.add_function(func)
                    return

            die(f"Error: class or function {class_name!r} not found in {filepath}")

        if method_name is not None:
            # Validate that the method exists and is an async test.
            attr = getattr(cls, method_name, None)
            if attr is None or not inspect.iscoroutinefunction(attr):
                die(f"Error: async test method {method_name!r} not found on {class_name!r} in {filepath}")
            suite.add_class(cls, [method_name])
        else:
            methods = _collect_test_methods(cls)
            if methods:
                suite.add_class(cls, methods)
    else:
        # Collect all test classes and standalone functions from the file.
        for cls in _find_test_classes(module):
            methods = _collect_test_methods(cls)
            if methods:
                suite.add_class(cls, methods)
        for func in _find_test_functions(module):
            suite.add_function(func)


# ===================================================================== #
#  Module-level discovery helper (public API)
# ===================================================================== #


def discover_module(module: types.ModuleType) -> AsyncTestSuite:
    """
    Build a suite from all ``AsyncTestCase`` subclasses and standalone
    ``async def test_*`` functions found in an already-imported module.
    """
    suite = AsyncTestSuite()
    for cls in _find_test_classes(module):
        methods = _collect_test_methods(cls)
        if methods:
            suite.add_class(cls, methods)
    for func in _find_test_functions(module):
        suite.add_function(func)
    return suite


# ===================================================================== #
#  Name-based test selection
# ===================================================================== #


def _resolve_name_specs(
    specs: list[str],
    source: AsyncTestSuite,
    dest: AsyncTestSuite,
) -> None:
    """Resolve name-based specs against a fully discovered *source* suite.

    Each spec is one of:

    * ``ClassName`` — all methods of that class
    * ``ClassName::method`` — a single method
    * ``test_function`` — a standalone function
    * ``test_name`` — a method or function name; must be unique

    Matched entries are added to *dest*.  Raises ``SystemExit(2)`` if
    a spec matches nothing or matches ambiguously.
    """
    entries = source.entries
    functions = source.functions

    for spec in specs:
        parts = spec.split("::")

        if len(parts) > 2:
            die(f"Error: invalid name spec {spec!r} – expected name or Class::method")

        if len(parts) == 2:
            # Class::method — look for the class by name, then the
            # method within it.
            cls_name, method_name = parts
            _resolve_class_method(cls_name, method_name, entries, dest)
            continue

        name = parts[0]

        # Try matching as a class name first.
        class_matches = [(cls, methods) for cls, methods in entries if cls.__name__ == name]
        # Try matching as a function name.
        func_matches = [f for f in functions if getattr(f, "__name__", None) == name]
        # Try matching as a method name across all classes.
        method_matches: list[tuple[type[AsyncTestCase], str]] = []
        if not class_matches and not func_matches:
            for cls, methods in entries:
                if name in methods:
                    method_matches.append((cls, name))

        total = len(class_matches) + len(func_matches) + len(method_matches)

        if total == 0:
            die(f"Error: no test matching {name!r} found")

        if total > 1:
            candidates: list[str] = []
            for cls, _methods in class_matches:
                candidates.append(f"  class {cls.__module__}.{cls.__qualname__}")
            for f in func_matches:
                qname = getattr(f, "__qualname__", repr(f))
                mod = getattr(f, "__module__", "<unknown>")
                candidates.append(f"  function {mod}.{qname}")
            for cls, meth in method_matches:
                candidates.append(f"  method {cls.__qualname__}.{meth}")
            die(f"Error: {name!r} is ambiguous — matches {total} tests:\n" + "\n".join(candidates))

        if class_matches:
            cls, methods = class_matches[0]
            dest.add_class(cls, methods)
        elif func_matches:
            dest.add_function(func_matches[0])
        else:
            cls, meth = method_matches[0]
            dest.add_class(cls, [meth])


def _resolve_class_method(
    cls_name: str,
    method_name: str,
    entries: list[tuple[type[AsyncTestCase], list[str]]],
    dest: AsyncTestSuite,
) -> None:
    """Resolve ``ClassName::method_name`` against discovered entries."""
    matches = [(cls, methods) for cls, methods in entries if cls.__name__ == cls_name]
    if not matches:
        die(f"Error: no test class matching {cls_name!r} found")
    if len(matches) > 1:
        candidates = [f"  {cls.__module__}.{cls.__qualname__}" for cls, _ in matches]
        die(f"Error: {cls_name!r} is ambiguous — matches {len(matches)} classes:\n" + "\n".join(candidates))
    cls, methods = matches[0]
    if method_name not in methods:
        die(f"Error: {cls_name!r} has no test method {method_name!r}")
    dest.add_class(cls, [method_name])


# ===================================================================== #
#  Internal helpers
# ===================================================================== #


def _matches_pattern(filename: str, pattern: str) -> bool:
    """Simple glob-style matching for ``test_*.py`` patterns."""
    # Support basic prefix/suffix wildcards that cover the common cases.
    if not filename.endswith(".py"):
        return False
    # Convert a simple ``prefix*suffix`` pattern into a check.
    parts = pattern.split("*")
    if len(parts) == 2:
        return filename.startswith(parts[0]) and filename.endswith(parts[1])
    # Fallback: use fnmatch for anything more complex.
    import fnmatch

    return fnmatch.fnmatch(filename, pattern)


def _import_path(filepath: Path, top_level: Path) -> types.ModuleType:
    """Import a Python file and return the module.

    Raises
    ------
    ImportError
        If the file cannot be loaded (missing spec, missing loader, or
        an exception during module execution).
    """
    try:
        relative = filepath.relative_to(top_level)
    except ValueError:
        relative = filepath

    # Build a dotted module name from the relative path.
    parts = list(relative.with_suffix("").parts)
    module_name = ".".join(parts)

    # Don't re-import if already loaded.
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, filepath)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot find module spec for {filepath}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    return module


def _find_test_classes(module: types.ModuleType) -> list[type[AsyncTestCase]]:
    """
    Return all concrete ``AsyncTestCase`` subclasses defined in *module*.

    Classes with ``__test__ = False`` are excluded (this follows the
    convention used by pytest and the existing code in this project).
    """
    classes: list[type[AsyncTestCase]] = []
    for _name, obj in inspect.getmembers(module, inspect.isclass):
        if not issubclass(obj, AsyncTestCase):
            continue
        if obj is AsyncTestCase:
            continue
        # Skip classes not defined in this module (imported bases, etc.)
        if obj.__module__ != module.__name__:
            continue
        # Honour the ``__test__ = False`` convention.
        if getattr(obj, "__test__", True) is False:
            continue
        classes.append(obj)
    return classes


def _find_named_class(
    module: types.ModuleType,
    class_name: str,
) -> type[AsyncTestCase] | None:
    """Find a specific ``AsyncTestCase`` subclass by name in *module*."""
    obj = getattr(module, class_name, None)
    if obj is None:
        return None
    if not isinstance(obj, type) or not issubclass(obj, AsyncTestCase):
        return None
    if obj is AsyncTestCase:
        return None
    return obj


def _find_test_functions(
    module: types.ModuleType,
) -> list[Callable[..., Coroutine[Any, Any, None]]]:
    """Return all module-level ``async def test_*`` coroutine functions.

    Functions with ``__test__ = False`` are excluded, following the
    same convention as :func:`_find_test_classes`.
    """
    functions: list[Callable[..., Coroutine[Any, Any, None]]] = []
    for name, obj in inspect.getmembers(module, inspect.iscoroutinefunction):
        if not name.startswith("test_"):
            continue
        # Skip functions not defined in this module.
        if getattr(obj, "__module__", None) != module.__name__:
            continue
        # Honour the ``__test__ = False`` convention.
        if getattr(obj, "__test__", True) is False:
            continue
        functions.append(obj)
    return sorted(functions, key=lambda f: f.__name__)


def _find_named_function(
    module: types.ModuleType,
    name: str,
) -> Callable[..., Coroutine[Any, Any, None]] | None:
    """Find a specific ``async def test_*`` function by name in *module*."""
    obj = getattr(module, name, None)
    if obj is None:
        return None
    if not inspect.iscoroutinefunction(obj):
        return None
    if not name.startswith("test_"):
        return None
    return cast(Callable[..., Coroutine[Any, Any, None]], obj)
