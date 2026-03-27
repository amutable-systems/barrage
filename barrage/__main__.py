# SPDX-License-Identifier: MIT
"""
CLI entry point for the async test framework.

Usage::

    # Discover and run all tests in the current directory
    python3 -m barrage

    # Run all tests in a directory
    python3 -m barrage test/

    # Run all tests in a specific file
    python3 -m barrage test/test_example.py

    # Run all tests in a specific class
    python3 -m barrage test/test_example.py::MyTestClass

    # Run a single test method
    python3 -m barrage test/test_example.py::MyTestClass::test_method

    # Run a single standalone test function
    python3 -m barrage test/test_example.py::test_function_name

    # Run by name (discovers from current directory)
    python3 -m barrage TestMyClass
    python3 -m barrage TestMyClass::test_method
    python3 -m barrage test_some_function

    # Multiple paths at once
    python3 -m barrage test/test_foo.py test/test_bar.py::SomeClass

    # With options
    python3 -m barrage -v test/test_example.py
    python3 -m barrage --max-concurrency 4 test/
    python3 -m barrage -i test/test_example.py::MyTestClass::test_method
    python3 -m barrage -x test/

    # Set a test directory (used as default search dir and import root)
    python3 -m barrage -t tests/
    python3 -m barrage --top-level-directory tests/ test_foo.py::MyClass::test_method
"""

import argparse
import sys
from pathlib import Path

from barrage.colorize import should_colorize
from barrage.discovery import resolve_tests
from barrage.runner import AsyncTestRunner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m barrage",
        description="Async concurrent test framework",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        const=2,
        default=1,
        dest="verbosity",
        help="Verbose output (per-test lines)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_const",
        const=0,
        dest="verbosity",
        help="Quiet output (summary only)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of tests to run concurrently (default: unlimited)",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        default=False,
        help=(
            "Interactive mode: run all tests sequentially, send output "
            "directly to the terminal, and print live per-test status. "
            "Useful for debugging."
        ),
    )
    parser.add_argument(
        "--show-output",
        action="store_true",
        default=False,
        help=(
            "Show captured stdout/stderr for all tests in the final "
            "report, including passing tests.  By default only failing "
            "and erroring tests have their output shown."
        ),
    )
    parser.add_argument(
        "-x",
        "--failfast",
        action="store_true",
        default=False,
        help="Stop on the first test failure or error.",
    )
    parser.add_argument(
        "-p",
        "--pattern",
        default="test_*.py",
        help="Pattern to match test files when discovering in directories (default: test_*.py)",
    )
    parser.add_argument(
        "-t",
        "--top-level-directory",
        type=Path,
        default=Path.cwd(),
        dest="top_level_directory",
        metavar="DIR",
        help=(
            "Top-level directory for test discovery and imports.  "
            "When no positional paths are given, tests are discovered "
            "in this directory instead of the current directory.  "
            "When paths are given, relative paths are resolved against "
            "this directory and it is used as the import root."
        ),
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["."],
        metavar="path",
        help=(
            "Paths to test files or directories.  A path may be suffixed "
            "with ::ClassName to select a single test class, "
            "::ClassName::test_method to select a single test, or "
            "::function_name to select a standalone test function.  "
            "A bare name (no path) discovers tests from the current "
            "directory and filters by class, function, or method name.  "
            "When no paths are given, discovers tests in the current "
            "directory (or -t if set)."
        ),
    )

    args = parser.parse_args(argv)
    suite = resolve_tests(args.paths, args.top_level_directory, pattern=args.pattern)

    if not suite.entries and not suite.functions:
        print("No tests discovered.", file=sys.stderr)
        return 2

    runner = AsyncTestRunner(
        max_concurrency=args.max_concurrency,
        verbosity=args.verbosity,
        interactive=args.interactive,
        show_output=args.show_output,
        failfast=args.failfast,
    )

    try:
        runner.run_suite(suite)
    except KeyboardInterrupt:
        pass

    # When per-test lines were already streamed live (either via
    # interactive mode or via the progress display), cap the report
    # verbosity so they are not printed a second time.
    if args.interactive:
        report_verbosity = 0
    elif runner.streamed_results:
        report_verbosity = min(args.verbosity, 1)
    else:
        report_verbosity = args.verbosity
    report = runner.result.format_report(
        verbosity=report_verbosity,
        show_output=args.show_output,
        color=should_colorize(sys.stdout),
    )
    print(report, end="")

    return 0 if runner.result.was_successful else 1


if __name__ == "__main__":
    sys.exit(main())
