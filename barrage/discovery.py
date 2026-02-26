# SPDX-License-Identifier: MIT
import importlib
import importlib.util
import inspect
import os
import sys
import types
from pathlib import Path

from barrage.case import AsyncTestCase
from barrage.runner import AsyncTestSuite, _collect_test_methods


def resolve_tests(
    path_specs: list[str],
    pattern: str = "test_*.py",
) -> AsyncTestSuite:
    """
    Build an :class:`AsyncTestSuite` from a list of path specifications.

    Each *path_spec* is a string of the form::

        path[::ClassName[::method_name]]

    Where *path* may be a directory or a ``.py`` file.

    * **Directory** – recursively discover all test files matching
      *pattern* (``::`` suffixes are not allowed here).
    * **File** – import the file and collect all
      :class:`AsyncTestCase` subclasses.
    * **File::ClassName** – import the file and collect only the
      named class.
    * **File::ClassName::method_name** – import the file and collect
      only the named method from the named class.

    Parameters
    ----------
    path_specs:
        One or more path specifications as described above.
    pattern:
        Glob pattern for test file names used during directory
        discovery (default ``test_*.py``).

    Returns
    -------
    AsyncTestSuite
        A suite containing the resolved test classes and methods.

    Raises
    ------
    SystemExit
        If a path does not exist, a class is not found, or a method
        is not found, an error message is printed to stderr and
        ``SystemExit(2)`` is raised.
    """
    suite = AsyncTestSuite()

    for spec in path_specs:
        parts = spec.split("::")
        path_str = parts[0]
        class_name: str | None = parts[1] if len(parts) > 1 else None
        method_name: str | None = parts[2] if len(parts) > 2 else None

        if len(parts) > 3:
            print(
                f"Error: invalid test spec {spec!r} – expected path[::Class[::method]]",
                file=sys.stderr,
            )
            raise SystemExit(2)

        p = Path(path_str)

        if p.is_dir():
            if class_name is not None:
                print(
                    f"Error: cannot use ::ClassName filter on a directory: {spec!r}",
                    file=sys.stderr,
                )
                raise SystemExit(2)
            _discover_directory(p, suite, pattern=pattern)

        elif p.is_file():
            _discover_file(
                p,
                suite,
                class_name=class_name,
                method_name=method_name,
            )

        else:
            print(
                f"Error: path does not exist: {path_str!r}",
                file=sys.stderr,
            )
            raise SystemExit(2)

    return suite


# ===================================================================== #
#  Directory discovery
# ===================================================================== #


def discover(
    start_dir: str | Path,
    pattern: str = "test_*.py",
    top_level_dir: str | Path | None = None,
) -> AsyncTestSuite:
    """
    Discover ``AsyncTestCase`` subclasses by walking *start_dir* and
    importing every Python file whose name matches *pattern*.

    Parameters
    ----------
    start_dir:
        Directory to start searching from.
    pattern:
        Glob pattern for test file names (default ``test_*.py``).
    top_level_dir:
        The top-level directory of the project.  If given, it is inserted
        into ``sys.path`` so that test modules can be imported by their
        dotted package path.  When ``None``, *start_dir* is used.

    Returns
    -------
    AsyncTestSuite
        A suite containing every discovered test class and its methods.
    """
    suite = AsyncTestSuite()
    _discover_directory(
        Path(start_dir),
        suite,
        pattern=pattern,
        top_level_dir=Path(top_level_dir) if top_level_dir else None,
    )
    return suite


def _discover_directory(
    start: Path,
    suite: AsyncTestSuite,
    pattern: str = "test_*.py",
    top_level_dir: Path | None = None,
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


# ===================================================================== #
#  Single-file discovery
# ===================================================================== #


def _discover_file(
    filepath: Path,
    suite: AsyncTestSuite,
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
    """
    filepath = filepath.resolve()

    # Use the current working directory as the top-level import root so
    # that ``test/test_foo.py`` imports as ``test.test_foo``.
    top = Path.cwd().resolve()
    if str(top) not in sys.path:
        sys.path.insert(0, str(top))

    try:
        module = _import_path(filepath, top)
    except Exception as e:
        print(
            f"Error: failed to import {filepath}: {e}",
            file=sys.stderr,
        )
        raise SystemExit(2) from e

    if class_name is not None:
        # Look for a specific class.
        cls = _find_named_class(module, class_name)
        if cls is None:
            print(
                f"Error: class {class_name!r} not found in {filepath}",
                file=sys.stderr,
            )
            raise SystemExit(2)

        if method_name is not None:
            # Validate that the method exists and is an async test.
            attr = getattr(cls, method_name, None)
            if attr is None or not inspect.iscoroutinefunction(attr):
                print(
                    f"Error: async test method {method_name!r} not found on {class_name!r} in {filepath}",
                    file=sys.stderr,
                )
                raise SystemExit(2)
            suite.add_class(cls, [method_name])
        else:
            methods = _collect_test_methods(cls)
            if methods:
                suite.add_class(cls, methods)
    else:
        # Collect all test classes from the file.
        for cls in _find_test_classes(module):
            methods = _collect_test_methods(cls)
            if methods:
                suite.add_class(cls, methods)


# ===================================================================== #
#  Module-level discovery helper (public API)
# ===================================================================== #


def discover_module(module: types.ModuleType) -> AsyncTestSuite:
    """
    Build a suite from all ``AsyncTestCase`` subclasses found in an
    already-imported module.
    """
    suite = AsyncTestSuite()
    for cls in _find_test_classes(module):
        methods = _collect_test_methods(cls)
        if methods:
            suite.add_class(cls, methods)
    return suite


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
