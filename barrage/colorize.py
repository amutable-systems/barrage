# SPDX-License-Identifier: MIT
"""
Capture ``sys.excepthook()`` output for display in test reports.

When outputting to a terminal, Python 3.13+ emits colorized tracebacks.
This module captures that exact output by routing ``sys.excepthook()``
through a pty (pseudo-terminal), so the result contains ANSI color
escapes when Python would normally produce them.

The key entry point is :func:`capture_excepthook` which returns the
formatted traceback string exactly as Python's exception machinery
would render it.

:func:`should_colorize` decides whether colors are appropriate for a
given stream, respecting ``PYTHON_COLORS``, ``NO_COLOR``, and
``FORCE_COLOR`` environment variables as well as TTY detection.

:func:`strip_ansi` removes ANSI escape sequences from a string for
contexts where colors are unwanted (e.g. piping to a file).

Color helpers
-------------
:data:`ANSI` provides named ANSI SGR escape codes.

:func:`style` wraps text in an ANSI style sequence (with automatic reset).

:func:`colored_symbol` returns a colored outcome symbol (✓/✗/E/S).

:func:`colored_duration` returns a duration string with color coding
based on how long the test took (bold-red > 5 s, yellow > 1 s, dim
otherwise).

:func:`colored_result_line` formats a complete per-test result line
with colors.
"""

import io
import os
import re
import sys
from types import TracebackType
from typing import TextIO

# ===================================================================== #
#  ANSI helpers
# ===================================================================== #

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    """Remove all ANSI SGR escape sequences from *text*."""
    return _ANSI_RE.sub("", text)


# ===================================================================== #
#  ANSI color constants & helpers
# ===================================================================== #


class ANSI:
    """Named ANSI SGR escape sequences."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"

    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    BOLD_RED = "\033[1;31m"
    BOLD_GREEN = "\033[1;32m"
    BOLD_YELLOW = "\033[1;33m"
    BOLD_BLUE = "\033[1;34m"
    BOLD_MAGENTA = "\033[1;35m"
    BOLD_CYAN = "\033[1;36m"

    DIM_WHITE = "\033[2;37m"


def style(text: str, *codes: str) -> str:
    """Wrap *text* in ANSI escape *codes* with an automatic reset.

    >>> style("hello", ANSI.BOLD, ANSI.GREEN)
    '\\033[1m\\033[32mhello\\033[0m'

    When no codes are given, returns *text* unchanged.
    """
    if not codes:
        return text
    return "".join(codes) + text + ANSI.RESET


# ── Outcome formatting helpers ────────────────────────────────────────

_OUTCOME_STYLES: dict[str, tuple[str, ...]] = {
    "PASSED": (ANSI.GREEN,),
    "FAILED": (ANSI.BOLD_RED,),
    "ERRORED": (ANSI.BOLD_RED,),
    "SKIPPED": (ANSI.YELLOW,),
}

_OUTCOME_PLAIN_SYMBOLS: dict[str, str] = {
    "PASSED": "✓",
    "FAILED": "✗",
    "ERRORED": "E",
    "SKIPPED": "S",
}


def colored_symbol(outcome_name: str, color: bool = True) -> str:
    """Return the symbol for *outcome_name* optionally wrapped in color.

    *outcome_name* should be one of ``"PASSED"``, ``"FAILED"``,
    ``"ERRORED"``, or ``"SKIPPED"`` (matching ``Outcome`` enum names).
    """
    sym = _OUTCOME_PLAIN_SYMBOLS.get(outcome_name, "?")
    if not color:
        return sym
    codes = _OUTCOME_STYLES.get(outcome_name, ())
    return style(sym, *codes)


# Slow-test duration thresholds (seconds).
_SLOW_THRESHOLD = 1.0
_VERY_SLOW_THRESHOLD = 5.0


def colored_duration(seconds: float, color: bool = True) -> str:
    """Format a duration value with color hints for slow tests.

    * > 5 s  → bold red
    * > 1 s  → yellow
    * else   → dim
    """
    text = f"{seconds:.3f}s"
    if not color:
        return text
    if seconds >= _VERY_SLOW_THRESHOLD:
        return style(text, ANSI.BOLD_RED)
    if seconds >= _SLOW_THRESHOLD:
        return style(text, ANSI.YELLOW)
    return style(text, ANSI.DIM)


def colored_result_line(
    outcome_name: str,
    test_str: str,
    duration: float,
    color: bool = True,
) -> str:
    """Build a complete per-test result line with optional color.

    Returns something like ``  ✓ test_foo (0.003s)`` with ANSI escapes
    when *color* is ``True``.
    """
    sym = colored_symbol(outcome_name, color=color)
    dur = colored_duration(duration, color=color)
    if color:
        name = test_str
        # Dim the test name for passing tests so failures pop.
        if outcome_name == "PASSED":
            name = style(test_str, ANSI.DIM)
        elif outcome_name in ("FAILED", "ERRORED"):
            name = style(test_str, ANSI.BOLD_RED)
        elif outcome_name == "SKIPPED":
            name = style(test_str, ANSI.YELLOW)
        return f"  {sym} {name} ({dur})"
    return f"  {sym} {test_str} ({dur})"


def colored_section_header(label: str, test_str: str, color: bool = True) -> str:
    """Format a ``FAIL: test_name`` or ``ERROR: test_name`` section header."""
    if not color:
        return f"{label}: {test_str}"
    return f"{style(label, ANSI.BOLD_RED)}: {style(test_str, ANSI.BOLD)}"


def colored_separator(char: str = "=", width: int = 70, color: bool = True) -> str:
    """Return a separator line, dimmed when color is enabled."""
    line = char * width
    if color:
        return style(line, ANSI.DIM)
    return line


def colored_summary(
    was_successful: bool,
    tests_run: int,
    total_duration: float,
    n_failures: int = 0,
    n_errors: int = 0,
    n_skipped: int = 0,
    color: bool = True,
) -> str:
    """Build the final summary block (separator + counts + OK/FAILED)."""
    lines: list[str] = []
    lines.append(colored_separator("-", 70, color=color))
    ran_line = f"Ran {tests_run} test(s) in {total_duration:.3f}s"
    if color:
        ran_line = style(ran_line, ANSI.BOLD)
    lines.append(ran_line)
    lines.append("")

    if was_successful:
        parts: list[str] = []
        if n_skipped:
            parts.append(f"skipped={n_skipped}")
        extra = f" ({', '.join(parts)})" if parts else ""
        ok_text = f"OK{extra}"
        if color:
            ok_text = style(ok_text, ANSI.BOLD_GREEN)
        lines.append(ok_text)
    else:
        parts = []
        if n_failures:
            parts.append(f"failures={n_failures}")
        if n_errors:
            parts.append(f"errors={n_errors}")
        if n_skipped:
            parts.append(f"skipped={n_skipped}")
        fail_text = f"FAILED ({', '.join(parts)})"
        if color:
            fail_text = style(fail_text, ANSI.BOLD_RED)
        lines.append(fail_text)

    return "\n".join(lines) + "\n"


def colored_captured_header(label: str, color: bool = True) -> str:
    """Format a 'Captured stdout:' / 'Captured stderr:' header."""
    if color:
        return style(label, ANSI.DIM, ANSI.ITALIC)
    return label


def colored_overview(
    entries: list[tuple[str, int]],
    total_tests: int,
    color: bool = True,
) -> str:
    """Format the pre-run overview showing how many tests will be run.

    *entries* is a list of ``(class_name, n_methods)`` pairs.
    *total_tests* is the total number of test methods across all classes.

    Returns a multi-line string like::

        Collected 5 test(s) from 2 class(es)

          SharedVMSmokeTests (2 tests)
          OtherTests (3 tests)
    """
    n_classes = len(entries)
    buf: list[str] = []

    header = f"Collected {total_tests} test(s) from {n_classes} class(es)"
    if color:
        header = style(header, ANSI.BOLD)
    buf.append(header)
    buf.append("")

    for cls_name, n_methods in entries:
        label = "test" if n_methods == 1 else "tests"
        if color:
            line = f"  {style(cls_name, ANSI.CYAN)} ({n_methods} {label})"
        else:
            line = f"  {cls_name} ({n_methods} {label})"
        buf.append(line)

    buf.append("")
    return "\n".join(buf) + "\n"


def colored_spinner_line(
    frame: str,
    n_running: int,
    total: int,
    name: str,
) -> str:
    """Format the live progress spinner line with color."""
    return (
        f"  {style(frame, ANSI.CYAN)} "
        f"{style(str(n_running), ANSI.BOLD_CYAN)}"
        f"{style('/', ANSI.DIM)}"
        f"{style(str(total), ANSI.DIM)}"
        f"{style(' running', ANSI.DIM)} "
        f"{style('—', ANSI.DIM)} "
        f"{style(name, ANSI.CYAN)}"
    )


# ===================================================================== #
#  TTY / environment detection
# ===================================================================== #


def should_colorize(stream: TextIO | None = None) -> bool:
    """
    Decide whether colored output is appropriate for *stream*.

    The check order mirrors Python 3.13's ``_colorize.can_colorize()``:

    1. ``PYTHON_COLORS=0`` → **no** color
    2. ``PYTHON_COLORS=1`` → color
    3. ``NO_COLOR`` set    → **no** color  (https://no-color.org/)
    4. ``FORCE_COLOR`` set → color
    5. *stream*.isatty()   → color if True

    When *stream* is ``None``, ``sys.stderr`` is used.
    """
    py_colors = os.environ.get("PYTHON_COLORS")
    if py_colors is not None:
        return py_colors != "0"

    if os.environ.get("NO_COLOR") is not None:
        return False

    if os.environ.get("FORCE_COLOR") is not None:
        return True

    if stream is None:
        stream = sys.stderr
    try:
        return stream.isatty()
    except (AttributeError, ValueError):
        return False


# ===================================================================== #
#  sys.excepthook() capture
# ===================================================================== #


def _capture_excepthook_plain(
    exc_type: type[BaseException],
    exc_val: BaseException,
    exc_tb: TracebackType | None,
) -> str:
    """Capture ``sys.excepthook()`` output via a :class:`io.StringIO`.

    Because the target is not a terminal, Python will **not** add ANSI
    color codes – the result is always plain text.
    """
    buf = io.StringIO()
    saved = sys.stderr
    sys.stderr = buf
    try:
        sys.excepthook(exc_type, exc_val, exc_tb)
    finally:
        sys.stderr = saved
    return buf.getvalue()


def _capture_excepthook_pty(
    exc_type: type[BaseException],
    exc_val: BaseException,
    exc_tb: TracebackType | None,
) -> str:
    """Capture ``sys.excepthook()`` output via a *pty*.

    The slave end of the pty looks like a real terminal to Python's
    traceback machinery, so on Python 3.13+ this produces output with
    ANSI color escapes (subject to ``NO_COLOR`` / ``PYTHON_COLORS``).
    """
    import termios

    master_fd, slave_fd = os.openpty()
    try:
        # Disable output post-processing so the terminal driver does
        # not convert ``\n`` into ``\r\n``.
        attrs = termios.tcgetattr(slave_fd)
        attrs[1] = attrs[1] & ~termios.OPOST
        termios.tcsetattr(slave_fd, termios.TCSANOW, attrs)

        slave_file = os.fdopen(slave_fd, "w")
        # slave_fd is now owned by slave_file – prevent double-close.
        slave_fd = -1

        saved = sys.stderr
        sys.stderr = slave_file
        try:
            sys.excepthook(exc_type, exc_val, exc_tb)
        finally:
            sys.stderr = saved
            slave_file.flush()
            slave_file.close()

        # Read everything the hook wrote from the master end.
        chunks: list[bytes] = []
        while True:
            try:
                chunk = os.read(master_fd, 16384)
                if not chunk:
                    break
                chunks.append(chunk)
            except OSError:
                # EIO on Linux once the slave is closed and drained.
                break

        return b"".join(chunks).decode("utf-8", errors="replace")
    finally:
        os.close(master_fd)
        if slave_fd != -1:
            os.close(slave_fd)


def capture_excepthook(
    exc_type: type[BaseException],
    exc_val: BaseException,
    exc_tb: TracebackType | None,
    *,
    colorize: bool = False,
) -> str:
    """Capture the output of ``sys.excepthook()`` and return it as a string.

    Parameters
    ----------
    exc_type, exc_val, exc_tb:
        The exception triple (as returned by ``sys.exc_info()``).
    colorize:
        When ``True``, a *pty* is used so that Python's traceback
        machinery believes it is writing to a terminal and emits ANSI
        color escapes (on Python 3.13+).  When ``False``, a plain
        :class:`~io.StringIO` is used and the output contains no
        escape sequences.

    Returns
    -------
    str
        The formatted traceback, exactly as ``sys.excepthook()`` would
        have printed it to ``sys.stderr``.
    """
    if colorize:
        try:
            return _capture_excepthook_pty(exc_type, exc_val, exc_tb)
        except (OSError, ImportError):
            # pty not available (unlikely on Linux but be safe).
            pass

    return _capture_excepthook_plain(exc_type, exc_val, exc_tb)
