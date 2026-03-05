# SPDX-License-Identifier: MIT

"""
Tests for :mod:`barrage.subprocess` – the ``spawn()`` and ``run()`` helpers.

These tests exercise:
  - Basic subprocess execution via ``run()`` and ``spawn()``.
  - Output relaying to ``sys.stdout`` / ``sys.stderr`` (and therefore
    into barrage's per-test capture buffers).
  - Explicit redirections (``PIPE``, ``DEVNULL``, ``STDOUT``).
  - The ``check`` parameter of ``spawn()`` and ``run()``.
  - Error handling (non-existent command, fd cleanup).
  - That relayed output is properly captured by the framework.
  - That ``spawn()`` as an async context manager always cleans up.
"""

import asyncio
import signal
import subprocess

from barrage.case import AsyncTestCase
from barrage.result import AsyncTestResult
from barrage.runner import AsyncTestRunner
from barrage.subprocess import (
    DEVNULL,
    PIPE,
    STDOUT,
    run,
    spawn,
)

# ===================================================================== #
#  Helpers
# ===================================================================== #


async def _run_inner(
    *classes: type[AsyncTestCase],
) -> AsyncTestResult:
    """Run inner test classes through the runner and return the result."""
    runner = AsyncTestRunner(verbosity=0)
    return await runner.run_classes_async(*classes)


# ===================================================================== #
#  run() basics
# ===================================================================== #


class TestRunBasic(AsyncTestCase):
    async def test_run_returns_completed_process(self) -> None:
        result = await run(["true"])
        self.assertIsInstance(result, subprocess.CompletedProcess)
        self.assertEqual(result.returncode, 0)

    async def test_run_captures_nonzero_exit(self) -> None:
        result = await run(["false"], check=False)
        self.assertNotEqual(result.returncode, 0)

    async def test_run_check_success(self) -> None:
        result = await run(["true"], check=True)
        self.assertEqual(result.returncode, 0)

    async def test_run_check_raises_on_failure(self) -> None:
        with self.assertRaises(subprocess.CalledProcessError) as ctx:
            await run(["false"], check=True)
        assert isinstance(ctx.exception, subprocess.CalledProcessError)
        self.assertNotEqual(ctx.exception.returncode, 0)
        self.assertIn("false", str(ctx.exception))

    async def test_run_args_stored(self) -> None:
        result = await run(["echo", "hello"])
        self.assertIn("echo", [str(a) for a in result.args])

    async def test_run_with_env(self) -> None:
        result = await run(
            ["sh", "-c", 'test "$MY_VAR" = "hello"'],
            env={"MY_VAR": "hello", "PATH": "/usr/bin:/bin"},
            check=True,
        )
        self.assertEqual(result.returncode, 0)

    async def test_run_with_cwd(self) -> None:
        result = await run(
            ["sh", "-c", 'test "$(pwd)" = "/"'],
            cwd="/",
            check=True,
        )
        self.assertEqual(result.returncode, 0)

    async def test_run_stdout_pipe(self) -> None:
        result = await run(["echo", "hello-stdout"], stdout=PIPE)
        self.assertIn(b"hello-stdout", result.stdout)

    async def test_run_stderr_pipe(self) -> None:
        result = await run(["sh", "-c", "echo hello-stderr >&2"], stderr=PIPE)
        self.assertIn(b"hello-stderr", result.stderr)

    async def test_run_check_error_carries_output(self) -> None:
        with self.assertRaises(subprocess.CalledProcessError) as ctx:
            await run(["sh", "-c", "echo oops; exit 1"], stdout=PIPE, check=True)
        assert isinstance(ctx.exception, subprocess.CalledProcessError)
        self.assertIn(b"oops", ctx.exception.output)


# ===================================================================== #
#  spawn() basics
# ===================================================================== #


class TestSpawnBasic(AsyncTestCase):
    async def test_spawn_yields_process(self) -> None:
        async with spawn(["true"]) as proc:
            self.assertIsInstance(proc, asyncio.subprocess.Process)
            await proc.wait()
        self.assertEqual(proc.returncode, 0)

    async def test_spawn_pid(self) -> None:
        async with spawn(["sleep", "10"], check=False) as proc:
            self.assertIsNotNone(proc.pid)
            self.assertGreater(proc.pid, 0)
            proc.kill()
            await proc.wait()

    async def test_spawn_terminate(self) -> None:
        async with spawn(["sleep", "60"], check=False) as proc:
            proc.terminate()
            returncode = await proc.wait()
        # SIGTERM typically gives -15 on Linux.
        self.assertNotEqual(returncode, 0)

    async def test_spawn_kill(self) -> None:
        async with spawn(["sleep", "60"], check=False) as proc:
            proc.kill()
            returncode = await proc.wait()
        # SIGKILL gives -9 on Linux.
        self.assertNotEqual(returncode, 0)

    async def test_spawn_returncode_none_while_running(self) -> None:
        async with spawn(["sleep", "60"], check=False) as proc:
            self.assertIsNone(proc.returncode)
            proc.kill()
            await proc.wait()
        self.assertIsNotNone(proc.returncode)

    async def test_spawn_check_success(self) -> None:
        async with spawn(["true"], check=True) as proc:
            await proc.wait()
        self.assertEqual(proc.returncode, 0)

    async def test_spawn_check_raises_on_failure(self) -> None:
        with self.assertRaises(subprocess.CalledProcessError) as ctx:
            async with spawn(["false"], check=True) as proc:
                await proc.wait()
        assert isinstance(ctx.exception, subprocess.CalledProcessError)
        self.assertNotEqual(ctx.exception.returncode, 0)
        self.assertIn("false", str(ctx.exception))


# ===================================================================== #
#  spawn() cleanup guarantees
# ===================================================================== #


class TestSpawnCleanup(AsyncTestCase):
    async def test_cleanup_waits_for_process(self) -> None:
        """Exiting the context manager waits for the process. The caller
        is responsible for killing it first if needed."""
        async with spawn(["sleep", "60"], check=False) as proc:
            proc.kill()
            # Don't await proc.wait() — __aexit__ handles the wait.
        self.assertIsNotNone(proc.returncode)

    async def test_cleanup_on_exception(self) -> None:
        """If user code raises, __aexit__ still waits for the process."""
        proc_ref: asyncio.subprocess.Process | None = None
        try:
            async with spawn(["sleep", "60"]) as proc:
                proc_ref = proc
                proc.kill()
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert proc_ref is not None
        self.assertIsNotNone(proc_ref.returncode)

    async def test_cleanup_after_normal_exit(self) -> None:
        """If the process exited before __aexit__, no error should occur."""
        async with spawn(["true"]) as proc:
            await proc.wait()
        self.assertEqual(proc.returncode, 0)

    async def test_cleanup_on_cancellation(self) -> None:
        """When a sibling task raises inside a TaskGroup, tasks with
        active subprocesses are cancelled.  The subprocess must be
        killed and fully cleaned up without errors."""
        proc_ref: asyncio.subprocess.Process | None = None

        async def slow_subprocess() -> None:
            nonlocal proc_ref
            async with spawn(["sleep", "60"], check=False) as proc:
                proc_ref = proc
                await proc.wait()

        async def fail_fast() -> None:
            raise RuntimeError("boom")

        with self.assertRaises(ExceptionGroup):
            async with asyncio.TaskGroup() as tg:
                tg.create_task(slow_subprocess())
                # Yield so the subprocess task starts and enters spawn().
                await asyncio.sleep(0.05)
                tg.create_task(fail_fast())

        assert proc_ref is not None
        self.assertIsNotNone(proc_ref.returncode)

    async def test_cleanup_survives_repeated_cancellation(self) -> None:
        """If a task with an active subprocess is cancelled multiple
        times (e.g. by an impatient caller), spawn() must still finish
        cleanup without errors or leaked processes."""
        proc_ref: asyncio.subprocess.Process | None = None
        entered = asyncio.Event()

        async def slow_subprocess() -> None:
            nonlocal proc_ref
            async with spawn(["sleep", "60"], check=False) as proc:
                proc_ref = proc
                entered.set()
                await proc.wait()

        task = asyncio.create_task(slow_subprocess())
        await entered.wait()

        # Cancel the task multiple times in quick succession to
        # exercise the while-loop in spawn()'s finally block.
        for _ in range(5):
            task.cancel()
            await asyncio.sleep(0.01)

        # The task must finish despite the repeated cancellations.
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert proc_ref is not None
        self.assertIsNotNone(proc_ref.returncode)


# ===================================================================== #
#  Output relaying (auto-detects PTY vs pipe based on real stream)
# ===================================================================== #


class TestRelay(AsyncTestCase):
    async def test_stdout_relayed(self) -> None:
        """Subprocess stdout should appear in sys.stdout (captured by the
        framework) via the automatic relay."""

        class _Inner(AsyncTestCase):
            __test__ = False

            async def test_echo(self) -> None:
                await run(["echo", "hello-stdout"])

        result = await _run_inner(_Inner)
        self.assertTrue(result.was_successful)
        self.assertEqual(len(result.passed), 1)
        self.assertIn("hello-stdout", result.passed[0].stdout)

    async def test_stderr_relayed(self) -> None:
        class _Inner(AsyncTestCase):
            __test__ = False

            async def test_echo_err(self) -> None:
                await run(["sh", "-c", "echo hello-stderr >&2"])

        result = await _run_inner(_Inner)
        self.assertTrue(result.was_successful)
        self.assertEqual(len(result.passed), 1)
        self.assertIn("hello-stderr", result.passed[0].stderr)

    async def test_both_streams_relayed(self) -> None:
        class _Inner(AsyncTestCase):
            __test__ = False

            async def test_both(self) -> None:
                await run(["sh", "-c", "echo out-hello; echo err-hello >&2"])

        result = await _run_inner(_Inner)
        self.assertTrue(result.was_successful)
        self.assertEqual(len(result.passed), 1)
        self.assertIn("out-hello", result.passed[0].stdout)
        self.assertIn("err-hello", result.passed[0].stderr)


# ===================================================================== #
#  Explicit redirections bypass relaying
# ===================================================================== #


class TestExplicitRedirects(AsyncTestCase):
    async def test_stdout_pipe(self) -> None:
        """When stdout=PIPE, process.stdout should be readable and no
        relay task should be created for stdout."""
        async with spawn(["echo", "captured-via-pipe"], stdout=PIPE) as proc:
            assert proc.stdout is not None
            data = await proc.stdout.read()
            await proc.wait()
        self.assertIn(b"captured-via-pipe", data)

    async def test_stderr_pipe(self) -> None:
        async with spawn(
            ["sh", "-c", "echo err-captured >&2"],
            stderr=PIPE,
        ) as proc:
            assert proc.stderr is not None
            data = await proc.stderr.read()
            await proc.wait()
        self.assertIn(b"err-captured", data)

    async def test_stdout_devnull(self) -> None:
        """DEVNULL should suppress output without error."""
        result = await run(["echo", "gone"], stdout=DEVNULL)
        self.assertEqual(result.returncode, 0)

    async def test_stderr_devnull(self) -> None:
        result = await run(
            ["sh", "-c", "echo gone >&2"],
            stderr=DEVNULL,
        )
        self.assertEqual(result.returncode, 0)

    async def test_stdout_pipe_stderr_relayed(self) -> None:
        """When only stdout is redirected, stderr should still be relayed."""

        class _Inner(AsyncTestCase):
            __test__ = False

            async def test_mixed(self) -> None:
                async with spawn(
                    ["sh", "-c", "echo out-piped; echo err-relayed >&2"],
                    stdout=PIPE,
                ) as proc:
                    assert proc.stdout is not None
                    out = await proc.stdout.read()
                    await proc.wait()
                assert b"out-piped" in out

        result = await _run_inner(_Inner)
        self.assertTrue(result.was_successful)
        self.assertEqual(len(result.passed), 1)
        # stderr was relayed, so it should be captured.
        self.assertIn("err-relayed", result.passed[0].stderr)

    async def test_stderr_stdout_merges(self) -> None:
        """stderr=STDOUT should merge stderr into stdout."""
        async with spawn(
            ["sh", "-c", "echo from-stdout; echo from-stderr >&2"],
            stdout=PIPE,
            stderr=STDOUT,
        ) as proc:
            assert proc.stdout is not None
            data = await proc.stdout.read()
            await proc.wait()
        text = data.decode()
        self.assertIn("from-stdout", text)
        self.assertIn("from-stderr", text)


# ===================================================================== #
#  spawn() with background processes
# ===================================================================== #


class TestSpawnBackground(AsyncTestCase):
    async def test_spawn_background_output_relayed(self) -> None:
        """A background process's output should be relayed while it runs."""

        class _Inner(AsyncTestCase):
            __test__ = False

            async def test_bg(self) -> None:
                async with spawn(
                    ["sh", "-c", "echo background-out; sleep 0.05; echo background-done"],
                ) as proc:
                    await proc.wait()

        result = await _run_inner(_Inner)
        self.assertTrue(result.was_successful)
        self.assertIn("background-out", result.passed[0].stdout)
        self.assertIn("background-done", result.passed[0].stdout)

    async def test_spawn_multiple_processes(self) -> None:
        """Multiple spawned processes should all relay correctly."""

        class _Inner(AsyncTestCase):
            __test__ = False

            async def test_multi(self) -> None:
                async with spawn(["echo", "proc-one"]) as p1:
                    await p1.wait()
                async with spawn(["echo", "proc-two"]) as p2:
                    await p2.wait()

        result = await _run_inner(_Inner)
        self.assertTrue(result.was_successful)
        stdout = result.passed[0].stdout
        self.assertIn("proc-one", stdout)
        self.assertIn("proc-two", stdout)


# ===================================================================== #
#  Concurrent test isolation
# ===================================================================== #


class TestConcurrentIsolation(AsyncTestCase, concurrent=True):
    async def test_concurrent_relay_isolation(self) -> None:
        """Output from subprocesses in concurrent tests must not leak
        across capture boundaries."""

        class _Inner(AsyncTestCase, concurrent=True):
            __test__ = False

            async def test_a(self) -> None:
                await run(["sh", "-c", "echo isolated-aaa"])
                # Yield so both tests overlap.
                await asyncio.sleep(0)

            async def test_b(self) -> None:
                await run(["sh", "-c", "echo isolated-bbb"])
                await asyncio.sleep(0)

        result = await _run_inner(_Inner)
        self.assertTrue(result.was_successful)
        self.assertEqual(result.tests_run, 2)

        for r in result.passed:
            if "test_a" in r.test_str:
                self.assertIn("isolated-aaa", r.stdout)
                self.assertNotIn("isolated-bbb", r.stdout)
            else:
                self.assertIn("isolated-bbb", r.stdout)
                self.assertNotIn("isolated-aaa", r.stdout)


# ===================================================================== #
#  CalledProcessError (stdlib)
# ===================================================================== #


class TestCalledProcessError(AsyncTestCase):
    async def test_attributes(self) -> None:
        err = subprocess.CalledProcessError(42, ["my-cmd", "--flag"])
        self.assertEqual(err.returncode, 42)
        self.assertEqual(err.cmd, ["my-cmd", "--flag"])
        self.assertIn("42", str(err))

    async def test_run_check_error_attributes(self) -> None:
        with self.assertRaises(subprocess.CalledProcessError) as ctx:
            await run(["sh", "-c", "exit 7"], check=True)
        assert isinstance(ctx.exception, subprocess.CalledProcessError)
        self.assertEqual(ctx.exception.returncode, 7)


# ===================================================================== #
#  CompletedProcess (stdlib)
# ===================================================================== #


class TestCompletedProcess(AsyncTestCase):
    async def test_fields(self) -> None:
        cp: subprocess.CompletedProcess[str] = subprocess.CompletedProcess(args=["a", "b"], returncode=3)
        self.assertEqual(cp.args, ["a", "b"])
        self.assertEqual(cp.returncode, 3)


# ===================================================================== #
#  Error handling
# ===================================================================== #


class TestErrorHandling(AsyncTestCase):
    async def test_spawn_nonexistent_command(self) -> None:
        """Spawning a command that doesn't exist should raise promptly
        and not leak file descriptors."""
        with self.assertRaises(OSError):
            async with spawn(["/nonexistent/command/xxxxx"]):
                pass

    async def test_run_nonexistent_command(self) -> None:
        with self.assertRaises(OSError):
            await run(["/nonexistent/command/xxxxx"])


# ===================================================================== #
#  Large output
# ===================================================================== #


class TestLargeOutput(AsyncTestCase):
    async def test_large_stdout_relayed(self) -> None:
        """Ensure large output doesn't deadlock the relay."""

        class _Inner(AsyncTestCase):
            __test__ = False

            async def test_big(self) -> None:
                # Generate ~100KB of output.
                await run(
                    [
                        "sh",
                        "-c",
                        "for i in $(seq 1 1000); do echo "
                        "'line-of-output-padding-to-make-it-longer-"
                        "xxxxxxxxxxxxxxxxxxxx'; done",
                    ],
                )

        result = await _run_inner(_Inner)
        self.assertTrue(result.was_successful)
        # Check a decent amount got through.
        self.assertGreater(len(result.passed[0].stdout), 1000)


# ===================================================================== #
#  stdin passthrough
# ===================================================================== #


class TestStdin(AsyncTestCase):
    async def test_stdin_pipe(self) -> None:
        async with spawn(["cat"], stdin=PIPE, stdout=PIPE) as proc:
            assert proc.stdin is not None
            proc.stdin.write(b"hello from stdin\n")
            proc.stdin.close()
            assert proc.stdout is not None
            data = await proc.stdout.read()
            await proc.wait()
        self.assertEqual(data.strip(), b"hello from stdin")

    async def test_stdin_devnull(self) -> None:
        async with spawn(["cat"], stdin=DEVNULL, stdout=PIPE) as proc:
            assert proc.stdout is not None
            data = await proc.stdout.read()
            await proc.wait()
        self.assertEqual(data, b"")


# ===================================================================== #
#  communicate()
# ===================================================================== #


class TestCommunicate(AsyncTestCase):
    async def test_communicate_with_input(self) -> None:
        async with spawn(["cat"], stdin=PIPE, stdout=PIPE) as proc:
            stdout_data, stderr_data = await proc.communicate(b"comm-input\n")
        self.assertEqual(stdout_data, b"comm-input\n")
        self.assertEqual(proc.returncode, 0)

    async def test_communicate_without_input(self) -> None:
        async with spawn(["echo", "comm-out"], stdout=PIPE) as proc:
            stdout_data, stderr_data = await proc.communicate()
        self.assertIn(b"comm-out", stdout_data)


# ===================================================================== #
#  send_signal()
# ===================================================================== #


class TestSendSignal(AsyncTestCase):
    async def test_send_signal(self) -> None:
        async with spawn(["sleep", "60"], check=False) as proc:
            proc.send_signal(signal.SIGTERM)
            returncode = await proc.wait()
        self.assertEqual(returncode, -signal.SIGTERM)


# ===================================================================== #
#  Relay with failing test (output should still be captured)
# ===================================================================== #


class TestRelayOnFailure(AsyncTestCase):
    async def test_subprocess_output_captured_on_test_failure(self) -> None:
        """Even when a test fails, subprocess output that was relayed
        before the failure should be in the captured output."""

        class _Inner(AsyncTestCase):
            __test__ = False

            async def test_fail_after_subprocess(self) -> None:
                await run(["echo", "before-failure"])
                raise AssertionError("deliberate failure")

        result = await _run_inner(_Inner)
        self.assertFalse(result.was_successful)
        self.assertEqual(len(result.failures), 1)
        self.assertIn("before-failure", result.failures[0].stdout)
