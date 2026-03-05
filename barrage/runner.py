# SPDX-License-Identifier: MIT
import asyncio
import contextlib
import contextvars
import inspect
import io
import os
import sys
import time
from collections.abc import Generator, Iterable
from typing import TextIO

from barrage.case import AsyncTestCase, SkipTest
from barrage.colorize import (
    ANSI,
    capture_excepthook,
    colored_captured_header,
    colored_duration,
    colored_overview,
    colored_result_line,
    colored_spinner_line,
    should_colorize,
    style,
)
from barrage.result import AsyncTestResult, Outcome, TestOutcome
from barrage.singleton import SingletonManager, discover_singletons


def _collect_test_methods(cls: type[AsyncTestCase]) -> list[str]:
    """Return sorted list of ``async def test_*`` method names from *cls*."""
    methods: list[str] = []
    for name in sorted(dir(cls)):
        if not name.startswith("test_"):
            continue
        attr = getattr(cls, name, None)
        if attr is not None and inspect.iscoroutinefunction(attr):
            methods.append(name)
    return methods


# ===================================================================== #
#  Output capture via contextvars
# ===================================================================== #

_capture_stdout: contextvars.ContextVar[io.StringIO | None] = contextvars.ContextVar(
    "_capture_stdout", default=None
)
_capture_stderr: contextvars.ContextVar[io.StringIO | None] = contextvars.ContextVar(
    "_capture_stderr", default=None
)


class _CapturingStream:
    """
    A stream wrapper that redirects writes into a per-task
    ``contextvars.ContextVar`` buffer when one is set, and falls
    through to the original stream otherwise.

    This allows concurrent async tests to each capture their own
    output without interfering with one another.
    """

    def __init__(self, original: TextIO, var: contextvars.ContextVar[io.StringIO | None]) -> None:
        self._original = original
        self._var = var

    # ── Core write / flush ────────────────────────────────────────────

    def write(self, s: str) -> int:
        buf = self._var.get(None)
        if buf is not None:
            return buf.write(s)
        return self._original.write(s)

    def flush(self) -> None:
        buf = self._var.get(None)
        if buf is not None:
            buf.flush()
        else:
            self._original.flush()

    def writelines(self, lines: Iterable[str]) -> None:
        for line in lines:
            self.write(line)

    # ── Delegate everything else to the original stream ───────────────

    def __getattr__(self, name: str) -> object:
        return getattr(self._original, name)


class _CaptureContext:
    """
    Installs :class:`_CapturingStream` wrappers on ``sys.stdout`` and
    ``sys.stderr`` for the duration of the suite run.  The wrappers are
    reference-counted so that nested runs (meta-tests) are safe.
    """

    _refcount: int = 0
    _saved_stdout: TextIO | None = None
    _saved_stderr: TextIO | None = None

    def __enter__(self) -> None:
        type(self)._refcount += 1
        if type(self)._refcount == 1:
            saved_stdout = type(self)._saved_stdout = sys.stdout
            saved_stderr = type(self)._saved_stderr = sys.stderr
            sys.stdout = _CapturingStream(saved_stdout, _capture_stdout)
            sys.stderr = _CapturingStream(saved_stderr, _capture_stderr)

    def __exit__(self, *args: object) -> None:
        type(self)._refcount -= 1
        if type(self)._refcount == 0:
            if type(self)._saved_stdout is not None:
                sys.stdout = type(self)._saved_stdout
            if type(self)._saved_stderr is not None:
                sys.stderr = type(self)._saved_stderr
            type(self)._saved_stdout = None
            type(self)._saved_stderr = None


class _CapturedOutput:
    """Holds the captured stdout/stderr buffers for a single test."""

    def __init__(self, out_buf: io.StringIO, err_buf: io.StringIO) -> None:
        self._out_buf = out_buf
        self._err_buf = err_buf

    @property
    def stdout(self) -> str:
        return self._out_buf.getvalue()

    @property
    def stderr(self) -> str:
        return self._err_buf.getvalue()


@contextlib.contextmanager
def _capture_output() -> Generator[_CapturedOutput]:
    """Context manager that captures stdout/stderr for the current task context."""
    out_buf = io.StringIO()
    err_buf = io.StringIO()
    out_token = _capture_stdout.set(out_buf)
    err_token = _capture_stderr.set(err_buf)
    try:
        yield _CapturedOutput(out_buf, err_buf)
    finally:
        _capture_stdout.reset(out_token)
        _capture_stderr.reset(err_token)


# ===================================================================== #
#  Suite
# ===================================================================== #


class AsyncTestSuite:
    """
    A collection of test classes (and optionally specific method names)
    to execute together.

    Usage::

        suite = AsyncTestSuite()
        suite.add_class(MyTests)
        suite.add_class(OtherTests, methods=["test_specific"])
    """

    def __init__(self) -> None:
        self._entries: list[tuple[type[AsyncTestCase], list[str] | None]] = []

    def add_class(
        self,
        cls: type[AsyncTestCase],
        methods: list[str] | None = None,
    ) -> None:
        """
        Add a test class.  If *methods* is ``None`` every ``test_*``
        coroutine on the class is collected automatically.
        """
        self._entries.append((cls, methods))

    @property
    def entries(self) -> list[tuple[type[AsyncTestCase], list[str]]]:
        resolved: list[tuple[type[AsyncTestCase], list[str]]] = []
        for cls, methods in self._entries:
            if methods is None:
                methods = _collect_test_methods(cls)
            resolved.append((cls, methods))
        return resolved


# ===================================================================== #
#  Live progress display
# ===================================================================== #


class _ProgressDisplay:
    """
    Streams per-test results as they complete and, when connected to a
    TTY, shows a live status line at the bottom of the output that
    cycles through the currently-running tests every second.

    The status line format is::

        <running>/<total> running — <test name>

    It is erased (via ANSI ``\\r\\033[2K``) before each result line is
    printed and after the last test finishes.

    All writes are serialised with an :class:`asyncio.Lock` so that
    concurrent test tasks never interleave partial lines.
    """

    def __init__(
        self,
        stream: TextIO,
        total: int,
        *,
        is_tty: bool = False,
        verbosity: int = 1,
        show_output: bool = False,
        color: bool = False,
    ) -> None:
        self._stream = stream
        self._total = total
        self._is_tty = is_tty
        self._verbosity = verbosity
        self._show_output = show_output
        self._color = color
        self._lock = asyncio.Lock()
        # Ordered dict would be nice but plain dict preserves insertion
        # order since Python 3.7.
        self._running: dict[str, str] = {}
        self._completed = 0
        self._progress_visible = False
        self._ticker_task: asyncio.Task[None] | None = None
        self._cycle_index = 0
        self._spinner_index = 0
        self._spinner_frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        # Classes currently running setUpClass / tearDownClass.
        self._setup_classes: dict[str, str] = {}

    # ── lifecycle ─────────────────────────────────────────────────────

    async def __aenter__(self) -> "_ProgressDisplay":
        if self._is_tty:
            self._ticker_task = asyncio.create_task(self._ticker())
        return self

    async def __aexit__(self, *args: object) -> None:
        if self._ticker_task is not None:
            self._ticker_task.cancel()
            try:
                await self._ticker_task
            except asyncio.CancelledError:
                pass
            self._ticker_task = None
        async with self._lock:
            self._clear_progress()

    # ── public notifications ──────────────────────────────────────────

    async def class_started(self, class_name: str) -> None:
        """Called before ``setUpClass`` runs."""
        async with self._lock:
            self._setup_classes[class_name] = f"{class_name} (setting up)"
            self._redraw_progress()

    async def class_setup_finished(self, class_name: str) -> None:
        """Called after ``setUpClass`` completes (success or failure)."""
        async with self._lock:
            self._setup_classes.pop(class_name, None)
            self._redraw_progress()

    async def test_started(self, test_id: str, display_name: str) -> None:
        """Called when a test begins executing (inside the semaphore)."""
        async with self._lock:
            self._running[test_id] = display_name
            self._redraw_progress()

    async def test_finished(self, outcome: TestOutcome) -> None:
        """Called when a test finishes.  Prints the result line."""
        async with self._lock:
            self._running.pop(outcome.test_id, None)
            self._completed += 1
            self._clear_progress()
            if self._verbosity >= 1:
                self._write_result(outcome)

    # ── background ticker ─────────────────────────────────────────────

    async def _ticker(self) -> None:
        try:
            ticks = 0
            while True:
                await asyncio.sleep(0.2)
                async with self._lock:
                    self._spinner_index += 1
                    ticks += 1
                    # Rotate the displayed test name every ~1 s (5 ticks).
                    if ticks % 5 == 0:
                        self._cycle_index += 1
                    self._redraw_progress()
        except asyncio.CancelledError:
            pass

    # ── low-level output helpers (caller must hold ``_lock``) ─────────

    def _clear_progress(self) -> None:
        if self._progress_visible:
            self._stream.write("\r\033[2K")
            self._stream.flush()
            self._progress_visible = False

    def _redraw_progress(self) -> None:
        if not self._is_tty:
            return
        # Combine actual running tests with class-setup entries for
        # the spinner display.  Setup entries are shown when no real
        # tests are running yet (e.g. during a long setUpClass).
        all_names: list[str] = list(self._running.values()) + list(self._setup_classes.values())
        if not all_names:
            return
        # Always clear the current line first so that back-to-back
        # redraws (e.g. from rapid ``test_started`` calls) overwrite
        # instead of concatenating.
        self._stream.write("\r\033[2K")
        idx = self._cycle_index % len(all_names)
        name = all_names[idx]
        n_running = len(self._running)
        frame = self._spinner_frames[self._spinner_index % len(self._spinner_frames)]
        if n_running:
            # Normal test-running spinner.
            # Truncate the test name *before* applying colors so we never
            # split in the middle of an ANSI escape sequence.
            # The prefix "  F N/T running — " is fixed-width (sans color).
            prefix_len = len(f"  {frame} {n_running}/{self._total} running \u2014 ")
            try:
                import shutil

                cols = shutil.get_terminal_size().columns
                max_name = cols - prefix_len - 1
                if max_name > 0 and len(name) > max_name:
                    name = name[: max_name - 1] + "\u2026"
            except Exception:
                pass
            if self._color:
                line = colored_spinner_line(frame, n_running, self._total, name)
            else:
                line = f"  {frame} {n_running}/{self._total} running \u2014 {name}"
        else:
            # Only class-setup entries – show a "setting up" spinner.
            try:
                import shutil

                cols = shutil.get_terminal_size().columns
                prefix_len = len(f"  {frame} ")
                max_name = cols - prefix_len - 1
                if max_name > 0 and len(name) > max_name:
                    name = name[: max_name - 1] + "\u2026"
            except Exception:
                pass
            if self._color:
                line = f"  {style(frame, ANSI.CYAN)} {style(name, ANSI.CYAN)}"
            else:
                line = f"  {frame} {name}"
        self._stream.write(line)
        self._stream.flush()
        self._progress_visible = True

    def _write_result(self, r: TestOutcome) -> None:
        line = colored_result_line(
            r.outcome.name,
            r.test_str,
            r.duration,
            color=self._color,
        )
        self._stream.write(line + "\n")
        if self._show_output and (r.stdout or r.stderr):
            _write_captured_output_to_stream(
                self._stream,
                r,
                indent="      ",
                color=self._color,
            )
        self._stream.flush()


class _OutputDetector:
    """Thin pass-through stream wrapper that records whether any write occurred.

    Installed on ``sys.stdout`` / ``sys.stderr`` while a test is running
    in interactive mode so that :func:`_interactive_line` can decide
    whether the test produced visible output.  All calls are forwarded
    to the wrapped stream unchanged.
    """

    def __init__(self, original: TextIO) -> None:
        self._original = original
        self.written = False

    def write(self, s: str) -> int:
        if s:
            self.written = True
        return self._original.write(s)

    def flush(self) -> None:
        self._original.flush()

    def writelines(self, lines: Iterable[str]) -> None:
        for line in lines:
            self.write(line)

    def __getattr__(self, name: str) -> object:
        return getattr(self._original, name)


class _OutputDetectors:
    """Pair of output detectors for stdout and stderr."""

    def __init__(self, stdout: _OutputDetector, stderr: _OutputDetector) -> None:
        self.stdout = stdout
        self.stderr = stderr

    @property
    def had_output(self) -> bool:
        return self.stdout.written or self.stderr.written


@contextlib.contextmanager
def _replace_stdin() -> Generator[None]:
    """Replace stdin with /dev/null, restoring it on exit."""
    saved = sys.stdin
    devnull = open(os.devnull)
    sys.stdin = devnull
    try:
        yield
    finally:
        sys.stdin = saved
        devnull.close()


@contextlib.contextmanager
def _detect_output() -> Generator[_OutputDetectors]:
    """Context manager that installs output detectors on stdout/stderr."""
    stdout_detector = _OutputDetector(sys.stdout)
    stderr_detector = _OutputDetector(sys.stderr)
    sys.stdout = stdout_detector  # type: ignore[assignment]
    sys.stderr = stderr_detector  # type: ignore[assignment]
    try:
        yield _OutputDetectors(stdout_detector, stderr_detector)
    finally:
        sys.stdout = stdout_detector._original
        sys.stderr = stderr_detector._original


def _write_captured_output_to_stream(
    stream: TextIO,
    r: TestOutcome,
    indent: str,
    color: bool = False,
) -> None:
    """Write captured stdout/stderr for *r* to *stream*."""
    if r.stdout:
        header = colored_captured_header("Captured stdout:", color=color)
        stream.write(f"{indent}{header}\n")
        for line in r.stdout.splitlines(keepends=True):
            stream.write(f"{indent}  {line}")
        if not r.stdout.endswith("\n"):
            stream.write("\n")
    if r.stderr:
        header = colored_captured_header("Captured stderr:", color=color)
        stream.write(f"{indent}{header}\n")
        for line in r.stderr.splitlines(keepends=True):
            stream.write(f"{indent}  {line}")
        if not r.stderr.endswith("\n"):
            stream.write("\n")


# ===================================================================== #
#  Core execution helpers
# ===================================================================== #


class _FailFast(Exception):
    """Raised when failfast is enabled and a test fails or errors."""


async def _run_single_test(
    cls: type[AsyncTestCase],
    method_name: str,
    result: AsyncTestResult,
    semaphore: asyncio.Semaphore,
    *,
    capture: bool = True,
    colorize: bool = False,
    interactive_stream: TextIO | None = None,
    progress: _ProgressDisplay | None = None,
    failfast: bool = False,
) -> None:
    """Run setUp → test → tearDown for one test method, recording the outcome."""

    instance = cls(method_name)
    method = getattr(instance, method_name)

    async with contextlib.AsyncExitStack() as stack:
        await stack.enter_async_context(semaphore)

        # Notify progress that this test is now actively running.
        if progress is not None:
            await progress.test_started(instance.id(), str(instance))

        # ── optional output capture ──────────────────────────────
        captured = stack.enter_context(_capture_output()) if capture else None

        # ── interactive prefix + output detection ────────────────
        if interactive_stream is not None:
            _interactive_pre(
                interactive_stream,
                instance,
                colorize=colorize,
            )
        detectors = stack.enter_context(_detect_output()) if interactive_stream is not None else None

        t0 = time.monotonic()

        # ── setUp ────────────────────────────────────────────────
        try:
            await instance.setUp()
        except SkipTest as exc:
            recorded = await result.add_skip(instance, exc.reason)
            if interactive_stream is not None:
                _interactive_line(
                    interactive_stream,
                    instance,
                    Outcome.SKIPPED,
                    0.0,
                    exc.reason,
                    colorize=colorize,
                    had_output=detectors.had_output if detectors else False,
                )
            if progress is not None:
                await progress.test_finished(recorded)
            return
        except asyncio.CancelledError:
            recorded = await result.add_interrupted(instance)
            if interactive_stream is not None:
                _interactive_line(
                    interactive_stream,
                    instance,
                    Outcome.INTERRUPTED,
                    0.0,
                    colorize=colorize,
                    had_output=detectors.had_output if detectors else False,
                )
            if progress is not None:
                await progress.test_finished(recorded)
            raise
        except BaseException as exc:
            duration = time.monotonic() - t0
            stdout = captured.stdout if captured else ""
            stderr = captured.stderr if captured else ""
            tb = capture_excepthook(type(exc), exc, exc.__traceback__, colorize=colorize)
            recorded = await result.add_error(
                instance, exc, duration, stdout=stdout, stderr=stderr, traceback_str=tb
            )
            if interactive_stream is not None:
                _interactive_line(
                    interactive_stream,
                    instance,
                    Outcome.ERRORED,
                    duration,
                    colorize=colorize,
                    had_output=detectors.had_output if detectors else False,
                )
                _interactive_traceback(interactive_stream, tb, stdout, stderr, colorize=colorize)
            if progress is not None:
                await progress.test_finished(recorded)
            if failfast:
                raise _FailFast() from None
            return

        # ── test method ──────────────────────────────────────────
        test_exc: BaseException | None = None
        test_is_skip = False
        test_is_failure = False
        try:
            await method()
        except SkipTest as exc:
            test_exc = exc
            test_is_skip = True
        except AssertionError as exc:
            test_exc = exc
            test_is_failure = True
        except Exception as exc:
            test_exc = exc

        # ── tearDown (always attempted) ──────────────────────────
        teardown_exc: Exception | None = None
        try:
            await instance.tearDown()
        except Exception as exc:
            teardown_exc = exc

        duration = time.monotonic() - t0
        stdout = captured.stdout if captured else ""
        stderr = captured.stderr if captured else ""
        had_output = detectors.had_output if detectors else False

        # ── record outcome ───────────────────────────────────────
        if cancelled_exc is not None:
            recorded = await result.add_interrupted(instance)
            if interactive_stream is not None:
                _interactive_line(
                    interactive_stream,
                    instance,
                    Outcome.INTERRUPTED,
                    duration,
                    colorize=colorize,
                    had_output=had_output,
                )
            if progress is not None:
                await progress.test_finished(recorded)
            raise cancelled_exc
        elif test_is_skip:
            assert isinstance(test_exc, SkipTest)
            recorded = await result.add_skip(instance, test_exc.reason)
            if interactive_stream is not None:
                _interactive_line(
                    interactive_stream,
                    instance,
                    Outcome.SKIPPED,
                    duration,
                    test_exc.reason,
                    colorize=colorize,
                    had_output=had_output,
                )
        elif test_is_failure:
            assert test_exc is not None
            tb = capture_excepthook(type(test_exc), test_exc, test_exc.__traceback__, colorize=colorize)
            recorded = await result.add_failure(
                instance, test_exc, duration, stdout=stdout, stderr=stderr, traceback_str=tb
            )
            if interactive_stream is not None:
                _interactive_line(
                    interactive_stream,
                    instance,
                    Outcome.FAILED,
                    duration,
                    colorize=colorize,
                    had_output=had_output,
                )
                _interactive_traceback(interactive_stream, tb, stdout, stderr, colorize=colorize)
        elif test_exc is not None:
            tb = capture_excepthook(type(test_exc), test_exc, test_exc.__traceback__, colorize=colorize)
            recorded = await result.add_error(
                instance, test_exc, duration, stdout=stdout, stderr=stderr, traceback_str=tb
            )
            if interactive_stream is not None:
                _interactive_line(
                    interactive_stream,
                    instance,
                    Outcome.ERRORED,
                    duration,
                    colorize=colorize,
                    had_output=had_output,
                )
                _interactive_traceback(interactive_stream, tb, stdout, stderr, colorize=colorize)
        elif teardown_exc is not None:
            tb = capture_excepthook(
                type(teardown_exc), teardown_exc, teardown_exc.__traceback__, colorize=colorize
            )
            recorded = await result.add_error(
                instance, teardown_exc, duration, stdout=stdout, stderr=stderr, traceback_str=tb
            )
            if interactive_stream is not None:
                _interactive_line(
                    interactive_stream,
                    instance,
                    Outcome.ERRORED,
                    duration,
                    colorize=colorize,
                    had_output=had_output,
                )
                _interactive_traceback(interactive_stream, tb, stdout, stderr, colorize=colorize)
        else:
            recorded = await result.add_success(instance, duration, stdout=stdout, stderr=stderr)
            if interactive_stream is not None:
                _interactive_line(
                    interactive_stream,
                    instance,
                    Outcome.PASSED,
                    duration,
                    colorize=colorize,
                    had_output=had_output,
                )

        if progress is not None:
            await progress.test_finished(recorded)

        if failfast and recorded.outcome in (Outcome.FAILED, Outcome.ERRORED):
            raise _FailFast()


async def _run_class(
    cls: type[AsyncTestCase],
    method_names: list[str],
    result: AsyncTestResult,
    semaphore: asyncio.Semaphore,
    *,
    capture: bool = True,
    colorize: bool = False,
    interactive_stream: TextIO | None = None,
    progress: _ProgressDisplay | None = None,
    failfast: bool = False,
) -> None:
    """
    Run all selected tests from a single class.

    * Calls ``setUpClass`` before any test and ``tearDownClass`` after all.
    * Respects ``cls.__concurrent__`` to decide whether the tests within
      the class run concurrently or sequentially.
    * When *interactive_stream* is not ``None``, forces sequential
      execution and prints live results to that stream.
    """

    if interactive_stream is not None:
        cls_name = cls.__qualname__
        if colorize:
            cls_name = style(cls_name, ANSI.BOLD, ANSI.UNDERLINE)
        interactive_stream.write(f"\n{cls_name}\n")
        interactive_stream.flush()
    elif progress is not None:
        await progress.class_started(cls.__qualname__)

    # setUpClass – capture stdout/stderr so that background tasks
    # spawned here (e.g. MonitoredTestCase.monitor_async_context) inherit
    # the capture context-var and don't leak output to the terminal.
    with _capture_output() if capture else contextlib.nullcontext() as setup_captured:
        try:
            await cls.setUpClass()
        except SkipTest as exc:
            if progress is not None:
                await progress.class_setup_finished(cls.__qualname__)
            for name in method_names:
                instance = cls(name)
                recorded = await result.add_skip(instance, exc.reason)
                if interactive_stream is not None:
                    _interactive_line(
                        interactive_stream, instance, Outcome.SKIPPED, 0.0, exc.reason, colorize=colorize
                    )
                if progress is not None:
                    await progress.test_finished(recorded)
            return
        except Exception as exc:
            if progress is not None:
                await progress.class_setup_finished(cls.__qualname__)
            stdout = setup_captured.stdout if setup_captured else ""
            stderr = setup_captured.stderr if setup_captured else ""
            # If setUpClass fails, record an error for each test and skip them.
            tb = capture_excepthook(type(exc), exc, exc.__traceback__, colorize=colorize)
            for name in method_names:
                instance = cls(name)
                recorded = await result.add_error(
                    instance, exc, 0.0, stdout=stdout, stderr=stderr, traceback_str=tb
                )
                if interactive_stream is not None:
                    _interactive_line(interactive_stream, instance, Outcome.ERRORED, 0.0, colorize=colorize)
                    _interactive_traceback(interactive_stream, tb, stdout, stderr, colorize=colorize)
                if progress is not None:
                    await progress.test_finished(recorded)
            return

        if progress is not None:
            await progress.class_setup_finished(cls.__qualname__)
        # setUpClass succeeded – stop capturing its direct output but
        # leave the buffers alive: background tasks that were spawned
        # during setUpClass already copied the context-var pointing at
        # these buffers, so they will keep writing there instead of to
        # the real stream.

    try:
        concurrent = cls.__concurrent__ and interactive_stream is None
        if concurrent:
            # Run all test methods concurrently
            async with asyncio.TaskGroup() as tg:
                for name in method_names:
                    tg.create_task(
                        _run_single_test(
                            cls,
                            name,
                            result,
                            semaphore,
                            capture=capture,
                            colorize=colorize,
                            interactive_stream=interactive_stream,
                            progress=progress,
                            failfast=failfast,
                        ),
                        name=f"{cls.__qualname__}.{name}",
                    )
        else:
            # Run test methods sequentially (in sorted order)
            for name in method_names:
                await _run_single_test(
                    cls,
                    name,
                    result,
                    semaphore,
                    capture=capture,
                    colorize=colorize,
                    interactive_stream=interactive_stream,
                    progress=progress,
                    failfast=failfast,
                )
    finally:
        # tearDownClass – always attempt it, with output capture.
        with _capture_output() if capture else contextlib.nullcontext() as teardown_captured:
            try:
                await cls.tearDownClass()
            except Exception as exc:
                stdout = teardown_captured.stdout if teardown_captured else ""
                stderr = teardown_captured.stderr if teardown_captured else ""
                # Record the tearDownClass failure against a synthetic test id
                tb = capture_excepthook(type(exc), exc, exc.__traceback__, colorize=colorize)
                instance = cls("tearDownClass")
                instance._test_method_name = "tearDownClass"
                recorded = await result.add_error(
                    instance, exc, 0.0, stdout=stdout, stderr=stderr, traceback_str=tb
                )
                if interactive_stream is not None:
                    _interactive_line(interactive_stream, instance, Outcome.ERRORED, 0.0, colorize=colorize)
                    _interactive_traceback(interactive_stream, tb, stdout, stderr, colorize=colorize)
                if progress is not None:
                    await progress.test_finished(recorded)


# ===================================================================== #
#  Interactive-mode helpers
# ===================================================================== #

_OUTCOME_LABELS: dict[Outcome, str] = {
    Outcome.PASSED: "ok",
    Outcome.FAILED: "FAIL",
    Outcome.ERRORED: "ERROR",
    Outcome.SKIPPED: "SKIPPED",
    Outcome.INTERRUPTED: "INTERRUPTED",
}

_OUTCOME_LABEL_STYLES: dict[Outcome, tuple[str, ...]] = {
    Outcome.PASSED: (ANSI.BOLD_GREEN,),
    Outcome.FAILED: (ANSI.BOLD_RED,),
    Outcome.ERRORED: (ANSI.BOLD_RED,),
    Outcome.SKIPPED: (ANSI.YELLOW,),
    Outcome.INTERRUPTED: (ANSI.BOLD_RED,),
}


def _interactive_pre(
    stream: TextIO,
    instance: AsyncTestCase,
    colorize: bool = False,
) -> None:
    """Print the ``test_name (Class) ...`` line before a test runs.

    A trailing newline is emitted so that any output the test produces
    appears on its own line rather than being appended to the status
    prefix.  :func:`_interactive_line` uses the output-detection flag
    to decide whether to move back up and append or to print a fresh
    result line.
    """
    method = instance._test_method_name
    cls = type(instance).__qualname__
    if colorize:
        display = f"  {method} ({style(cls, ANSI.DIM)}) ... "
    else:
        display = f"  {method} ({cls}) ... "
    stream.write(display + "\n")
    stream.flush()


def _interactive_line(
    stream: TextIO,
    instance: AsyncTestCase,
    outcome: Outcome,
    duration: float,
    reason: str = "",
    colorize: bool = False,
    had_output: bool = False,
) -> None:
    """Print the result of a single test in interactive mode.

    When *had_output* is ``False`` the test produced no visible output,
    so we move the cursor back up and append the result to the existing
    prefix line.  Otherwise a fresh ``test_name (Class) ... result``
    line is emitted so that the outcome is clearly associated with the
    test even when test output appeared on preceding lines.

    The cursor-up trick requires a TTY; on non-TTY streams a full line
    is always written.
    """
    method = instance._test_method_name
    cls_name = type(instance).__qualname__
    label = _OUTCOME_LABELS[outcome]
    extra = f" ({reason})" if reason else ""
    dur = colored_duration(duration, color=colorize)

    is_tty = False
    try:
        is_tty = stream.isatty()
    except (AttributeError, ValueError):
        pass

    if not had_output and is_tty:
        # No output was produced — go back up to the prefix line and
        # append the result there.
        if colorize:
            label_styled = style(label, *_OUTCOME_LABEL_STYLES.get(outcome, ()))
            name_styled = style(cls_name, ANSI.DIM)
            stream.write(f"\033[A\r\033[2K  {method} ({name_styled}) ... {label_styled}{extra} ({dur})\n")
        else:
            stream.write(f"\033[A\r\033[2K  {method} ({cls_name}) ... {label}{extra} ({duration:.3f}s)\n")
    else:
        # Output was produced (or no TTY) — print a full result line.
        if colorize:
            prefix = f"  {method} ({style(cls_name, ANSI.DIM)}) ... "
            label_styled = style(label, *_OUTCOME_LABEL_STYLES.get(outcome, ()))
            stream.write(f"{prefix}{label_styled}{extra} ({dur})\n")
        else:
            prefix = f"  {method} ({cls_name}) ... "
            stream.write(f"{prefix}{label}{extra} ({duration:.3f}s)\n")
    stream.flush()


def _interactive_traceback(
    stream: TextIO,
    tb_str: str,
    stdout: str,
    stderr: str,
    colorize: bool = False,
) -> None:
    """Print traceback + captured output immediately in interactive mode."""
    for line in tb_str.splitlines(keepends=True):
        stream.write(f"    {line}")
    if stdout:
        header = colored_captured_header("Captured stdout:", color=colorize)
        stream.write(f"    {header}\n")
        for line in stdout.splitlines(keepends=True):
            stream.write(f"      {line}")
        if not stdout.endswith("\n"):
            stream.write("\n")
    if stderr:
        header = colored_captured_header("Captured stderr:", color=colorize)
        stream.write(f"    {header}\n")
        for line in stderr.splitlines(keepends=True):
            stream.write(f"      {line}")
        if not stderr.endswith("\n"):
            stream.write("\n")
    stream.flush()


# ===================================================================== #
#  Runner
# ===================================================================== #


class AsyncTestRunner:
    """
    Discovers and runs :class:`AsyncTestCase` subclasses concurrently.

    Parameters
    ----------
    max_concurrency:
        Upper bound on the number of test methods that may execute at the
        same time across *all* classes.  ``None`` means unlimited.
        Ignored (forced to ``1``) in interactive mode.
    verbosity:
        Controls output detail.  ``0`` = summary only, ``1`` = default
        (failures + summary), ``2`` = per-test lines.
    interactive:
        When ``True``, tests run **sequentially** (classes and methods),
        output is **not** captured (it flows directly to the terminal),
        and per-test status is printed live.  Useful for debugging.
    show_output:
        When ``True``, captured stdout/stderr is included in the report
        even for *passing* tests.  By default only failing / erroring
        tests have their output shown.
    """

    def __init__(
        self,
        max_concurrency: int | None = None,
        verbosity: int = 1,
        interactive: bool = False,
        show_output: bool = False,
        interactive_stream: TextIO | None = None,
        failfast: bool = False,
        debug: bool = False,
    ) -> None:
        self.max_concurrency = max_concurrency
        self.verbosity = verbosity
        self.interactive = interactive
        self.show_output = show_output
        self._interactive_stream = interactive_stream
        self.failfast = failfast
        self.debug = debug
        # Set after a run completes – True when per-test result lines
        # were already streamed to the output, so ``format_report``
        # should not duplicate them.
        self.streamed_results: bool = False

    def run_suite(self, suite: AsyncTestSuite) -> AsyncTestResult:
        """Run the suite synchronously (creates its own event loop)."""
        return asyncio.run(self.run_suite_async(suite), debug=self.debug)

    async def run_suite_async(self, suite: AsyncTestSuite) -> AsyncTestResult:
        """Run the suite inside an already-running event loop."""
        result = self.result = AsyncTestResult()
        entries = suite.entries

        # ── singleton injection ───────────────────────────────────────
        has_singletons = any(discover_singletons(cls) for cls, _methods in entries)

        interactive_stream: TextIO | None = None
        capture: bool
        if self.interactive:
            # Interactive: sequential, no capture, live output.
            # Use the explicitly provided stream, fall back to stderr,
            # or stay None when verbosity=0 (suppresses status lines
            # while keeping sequential-no-capture semantics).
            if self._interactive_stream is not None:
                interactive_stream = self._interactive_stream
            elif self.verbosity >= 1:
                interactive_stream = sys.stderr
            capture = False
            limit = 1
        else:
            capture = True
            limit = self.max_concurrency or 2**31

        # Decide whether tracebacks should contain ANSI colors.
        # In interactive mode the target stream is stderr; otherwise
        # the report will eventually be printed to stdout.
        color_stream = interactive_stream if interactive_stream is not None else sys.stdout
        colorize = should_colorize(color_stream)

        semaphore = asyncio.Semaphore(limit)

        # Grab a reference to the real stdout *before* capture replaces
        # it with _CapturingStream.  The progress display writes here so
        # its output is never captured into a test's buffer.
        real_stdout = sys.stdout

        async with contextlib.AsyncExitStack() as suite_stack:
            if capture:
                suite_stack.enter_context(_CaptureContext())

            # ── replace stdin with /dev/null (non-interactive only) ──
            # When not in interactive mode, tests must not block waiting
            # for terminal input.  Point sys.stdin at /dev/null so any
            # accidental read returns EOF immediately.
            if not self.interactive:
                suite_stack.enter_context(_replace_stdin())

            # ── set up live progress display (non-interactive only) ──
            total_tests = sum(len(methods) for _, methods in entries)
            progress: _ProgressDisplay | None = None
            if not self.interactive and self.verbosity >= 1:
                is_tty = False
                try:
                    is_tty = real_stdout.isatty()
                except (AttributeError, ValueError):
                    pass
                progress = await suite_stack.enter_async_context(
                    _ProgressDisplay(
                        stream=real_stdout,
                        total=total_tests,
                        is_tty=is_tty,
                        verbosity=self.verbosity,
                        show_output=self.show_output,
                        color=colorize,
                    )
                )
                self.streamed_results = self.verbosity >= 1
            else:
                self.streamed_results = False

            # ── print pre-run overview ───────────────────────────────
            if self.verbosity >= 1:
                overview_entries = [(cls.__qualname__, len(methods)) for cls, methods in entries]
                overview = colored_overview(overview_entries, total_tests, color=colorize)
                overview_stream = interactive_stream if interactive_stream is not None else real_stdout
                overview_stream.write(overview)
                overview_stream.flush()

            result.start_time = time.monotonic()

            async def run_tests() -> None:
                """Run all test classes, with optional singleton injection."""
                if has_singletons:
                    async with SingletonManager() as sm:
                        await sm.inject(entries)
                        await _run_all_classes()
                else:
                    await _run_all_classes()

            async def _run_all_classes() -> None:
                if self.interactive:
                    # In interactive mode run classes sequentially so that
                    # their output is not interleaved.
                    for cls, methods in entries:
                        await _run_class(
                            cls,
                            methods,
                            result,
                            semaphore,
                            capture=capture,
                            colorize=colorize,
                            interactive_stream=interactive_stream,
                            progress=progress,
                            failfast=self.failfast,
                        )
                else:
                    # Each class gets its own task so that different classes
                    # run concurrently.
                    async with asyncio.TaskGroup() as tg:
                        for cls, methods in entries:
                            tg.create_task(
                                _run_class(
                                    cls,
                                    methods,
                                    result,
                                    semaphore,
                                    capture=capture,
                                    colorize=colorize,
                                    interactive_stream=interactive_stream,
                                    progress=progress,
                                    failfast=self.failfast,
                                ),
                                name=f"class:{cls.__qualname__}",
                            )

            try:
                await run_tests()
            except* _FailFast:
                pass

        result.end_time = time.monotonic()

        return result

    def run_classes(self, *classes: type[AsyncTestCase]) -> AsyncTestResult:
        """Convenience: build a suite from one or more classes and run it."""
        suite = AsyncTestSuite()
        for cls in classes:
            suite.add_class(cls)
        return self.run_suite(suite)

    async def run_classes_async(self, *classes: type[AsyncTestCase]) -> AsyncTestResult:
        """Async version of :meth:`run_classes`."""
        suite = AsyncTestSuite()
        for cls in classes:
            suite.add_class(cls)
        return await self.run_suite_async(suite)
