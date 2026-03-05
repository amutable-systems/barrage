# SPDX-License-Identifier: MIT

"""
Subprocess helpers for barrage.

Provides :func:`spawn` and :func:`run` for launching subprocesses whose
output is automatically relayed through ``sys.stdout`` / ``sys.stderr``
so that barrage's output-capturing machinery can intercept it.

When *stdout* or *stderr* is left as ``None`` (the default), the
corresponding file descriptor is connected to a PTY or a plain pipe
and a background :class:`asyncio.Task` relays every chunk of output to
the current ``sys.stdout`` / ``sys.stderr``.  A PTY is used when the
real standard stream (``sys.__stdout__`` / ``sys.__stderr__``) is a TTY,
so the subprocess sees ``isatty() == True`` and preserves colours,
line-buffering, etc.  When the real stream is not a TTY (e.g. output is
piped or redirected), a plain pipe is used instead.

Because barrage replaces those streams with context-var-aware capturing
wrappers, the subprocess output ends up in the correct per-test capture
buffer automatically.

:func:`spawn` is an async context manager that guarantees the child
process is killed and all relay tasks are drained on exit::

    async with spawn(["my-server", "--port", "8080"]) as proc:
        ...  # interact with the server
    # proc is killed & cleaned up here

:func:`run` is a thin convenience wrapper that waits for the subprocess
to finish::

    result = await run(["ls", "-la"])

Both reuse the standard-library types :class:`asyncio.subprocess.Process`,
:class:`subprocess.CompletedProcess`, and :class:`subprocess.CalledProcessError`
rather than inventing custom replacements.
"""

import asyncio
import errno
import os
import pty
import subprocess
import sys
from collections.abc import AsyncIterator, Sequence
from contextlib import ExitStack, asynccontextmanager
from io import TextIOWrapper
from typing import IO, TextIO

# Re-export subprocess constants for convenience so callers do not need
# to also import :mod:`subprocess`.
DEVNULL = subprocess.DEVNULL
PIPE = subprocess.PIPE
STDOUT = subprocess.STDOUT

# Re-export stdlib types so callers can ``from barrage.subprocess import …``
# without a separate ``import subprocess``.
CompletedProcess = subprocess.CompletedProcess
CalledProcessError = subprocess.CalledProcessError

# The type for the command and its arguments – a sequence of strings
# or path-like objects, matching what :func:`asyncio.create_subprocess_exec`
# accepts.
_Args = Sequence[str | os.PathLike[str]]

# The type accepted for *stdin*, *stdout*, and *stderr* parameters –
# mirrors what :func:`asyncio.create_subprocess_exec` accepts.
_Redirect = int | IO[bytes] | None


# ------------------------------------------------------------------- #
#  Internal relay helpers
# ------------------------------------------------------------------- #


async def _relay_fd(fd: int, target: TextIO) -> None:
    """Read from *fd* (set to non-blocking) and write to *target* until EOF.

    *target* is typically ``sys.stdout`` or ``sys.stderr`` (possibly
    wrapped by barrage's ``_CapturingStream``).  The file descriptor is
    closed when the relay finishes.

    This intentionally mirrors the pattern from the user-facing example:
    use ``loop.add_reader`` so we can ``await`` readability without
    burning CPU, then ``os.read`` raw bytes and decode them for the
    Python text stream.
    """
    loop = asyncio.get_running_loop()
    os.set_blocking(fd, False)
    try:
        while True:
            waiter: asyncio.Future[None] = loop.create_future()

            def _on_readable(fut: asyncio.Future[None] = waiter) -> None:
                if not fut.done():
                    fut.set_result(None)

            loop.add_reader(fd, _on_readable)
            try:
                await waiter
            finally:
                loop.remove_reader(fd)

            try:
                data = os.read(fd, 4096)
            except OSError as e:
                # On Linux, reading the master side of a PTY after the
                # slave has been closed raises ``EIO``.
                if e.errno != errno.EIO:
                    raise

                break
            if not data:
                break

            target.write(data.decode(errors="replace"))
            target.flush()
    finally:
        os.close(fd)


def _open_relay(use_pty: bool) -> tuple[int, int]:
    """Return ``(reader_fd, writer_fd)`` for output relaying.

    If *use_pty* is ``True``, a PTY pair is created so the subprocess
    sees ``isatty() == True`` (preserving colours, line-buffering, etc.).
    Otherwise a plain :func:`os.pipe` is used.
    """
    if use_pty:
        master, slave = pty.openpty()
        return master, slave
    r, w = os.pipe()
    return r, w


def _real_stream_is_tty(stream: TextIOWrapper | None) -> bool:
    """Return ``True`` if *stream* refers to a real TTY.

    *stream* should be one of ``sys.__stdout__`` or ``sys.__stderr__``.
    Returns ``False`` when the stream is ``None``, closed, or not a TTY.
    """
    if stream is None:
        return False
    try:
        return stream.isatty()
    except (AttributeError, ValueError):
        return False


# ------------------------------------------------------------------- #
#  Public API
# ------------------------------------------------------------------- #


@asynccontextmanager
async def spawn(
    args: _Args,
    *,
    stdin: _Redirect = None,
    stdout: _Redirect = None,
    stderr: _Redirect = None,
    env: dict[str, str] | None = None,
    cwd: str | os.PathLike[str] | None = None,
    check: bool = True,
) -> AsyncIterator[asyncio.subprocess.Process]:
    """Start a subprocess, yield it, and guarantee cleanup on exit.

    This is an **async context manager**.  The child process is killed
    (if still running) and all relay background tasks are drained when
    the ``async with`` block exits – whether normally or via an
    exception.

    Parameters
    ----------
    args:
        The command and its arguments as a sequence,
        e.g. ``["ls", "-la"]``.
    stdin, stdout, stderr:
        Stream redirections.  When ``None`` (the default for *stdout*
        and *stderr*), the stream is intercepted via a PTY (or pipe)
        and relayed to the corresponding ``sys`` stream so that
        barrage's capture machinery can record it.  Pass an explicit
        value such as :data:`PIPE`, :data:`DEVNULL`, or a file
        descriptor / file object to disable relaying for that stream.

        When relaying, a pseudo-terminal is used automatically if the
        corresponding real standard stream (``sys.__stdout__`` /
        ``sys.__stderr__``) is a TTY, so the subprocess sees
        ``isatty() == True`` (preserving colours, line-buffering, etc.).
        When the real stream is not a TTY, a plain pipe is used instead.
    env:
        Environment variables for the child process.
    cwd:
        Working directory for the child process.
    check:
        When ``True``, raise :class:`subprocess.CalledProcessError`
        after the context manager exits if the process returned a
        non-zero exit code.
    """
    # Pairs of (reader_fd, target_text_stream) that need relay tasks.
    relay_fds: list[tuple[int, TextIO]] = []
    # Writer fds that must be closed in the parent after the subprocess
    # has been created (the child inherits its own copy).
    close_after_fork: list[int] = []

    real_stdout: _Redirect = stdout
    real_stderr: _Redirect = stderr

    # The ExitStack registers os.close() for every fd we open.  If
    # anything fails (a second _open_relay, create_subprocess_exec, …)
    # the stack unwinds and closes them all.  On the success path we
    # pop_all() so that writer fds and reader fds can be handed off to
    # their new owners (the parent close-after-fork loop and the relay
    # tasks respectively).
    with ExitStack() as stack:
        if stdout is None:
            reader_fd, writer_fd = _open_relay(_real_stream_is_tty(sys.__stdout__))
            stack.callback(os.close, writer_fd)
            stack.callback(os.close, reader_fd)
            real_stdout = writer_fd
            close_after_fork.append(writer_fd)
            relay_fds.append((reader_fd, sys.stdout))

        if stderr is None:
            reader_fd, writer_fd = _open_relay(_real_stream_is_tty(sys.__stderr__))
            stack.callback(os.close, writer_fd)
            stack.callback(os.close, reader_fd)
            real_stderr = writer_fd
            close_after_fork.append(writer_fd)
            relay_fds.append((reader_fd, sys.stderr))

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=stdin,
            stdout=real_stdout,
            stderr=real_stderr,
            env=env,
            cwd=cwd,
        )

        # Subprocess created – take over fd management from the stack.
        stack.pop_all()

    # The subprocess has inherited the writer fds; the parent must
    # close its copies so that reads on the reader fds will see EOF
    # when the subprocess exits.
    for fd in close_after_fork:
        os.close(fd)

    # Start relay tasks up-front.  They are never cancelled: once the
    # subprocess terminates, the master side of each PTY/pipe sees EOF
    # and the relay exits on its own.  This avoids losing buffered
    # output that would be discarded if the task were cancelled.
    relay_tasks: list[asyncio.Task[None]] = []
    for rfd, target in relay_fds:
        relay_tasks.append(
            asyncio.create_task(
                _relay_fd(rfd, target),
                name=f"barrage-relay-fd{rfd}",
            )
        )

    try:
        yield proc
        # Wait for the process on the normal-exit path.  If this gets
        # cancelled, it falls through to the except clause which
        # terminates the process.
        await proc.wait()
    except BaseException:
        try:
            proc.terminate()
        except ProcessLookupError:
            pass
        raise
    finally:
        # Wait for the subprocess to exit, then drain every relay task.
        # If we get cancelled again during cleanup, escalate to SIGKILL
        # and keep waiting — the process *will* exit after SIGKILL so
        # this always terminates.
        while proc.returncode is None:
            try:
                await proc.wait()
            except asyncio.CancelledError:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
        for task in relay_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass

    assert proc.returncode is not None
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode,
            list(args),
        )


async def run(
    args: _Args,
    *,
    stdin: _Redirect = None,
    stdout: _Redirect = None,
    stderr: _Redirect = None,
    input: bytes | None = None,
    env: dict[str, str] | None = None,
    cwd: str | os.PathLike[str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    """Run a subprocess to completion.

    This is a convenience wrapper around :func:`spawn` that waits for
    the process to exit and returns a :class:`subprocess.CompletedProcess`.

    Parameters
    ----------
    input:
        Data to send to the subprocess's stdin.  When provided, *stdin*
        is automatically set to :data:`PIPE` if not already specified.
    check:
        When ``True``, raise :class:`subprocess.CalledProcessError` if
        the process returns a non-zero exit code.

    All other parameters are forwarded to :func:`spawn`.
    """
    if input is not None and stdin is None:
        stdin = PIPE

    stdout_data: bytes | None = None
    stderr_data: bytes | None = None
    try:
        async with spawn(
            args,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            env=env,
            cwd=cwd,
            check=check,
        ) as proc:
            stdout_data, stderr_data = await proc.communicate(input=input)
    except subprocess.CalledProcessError as e:
        # Enrich the error raised by spawn() with any captured output
        # from proc.communicate().
        e.output = stdout_data
        e.stderr = stderr_data
        raise

    assert proc.returncode is not None

    return subprocess.CompletedProcess(
        args=list(args),
        returncode=proc.returncode,
        stdout=stdout_data,
        stderr=stderr_data,
    )
