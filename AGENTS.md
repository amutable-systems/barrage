# AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All commands must be run via `mkosi box`, e.g. `mkosi box -- just check`.

- **Check (format, lint, spell, type-check):** `mkosi box -- just check`
- **Auto-format:** `mkosi box -- just fmt`
- **Run all tests:** `mkosi box -- just test`
- **Run specific tests:** `mkosi box -- just test tests/test_framework.py` or `mkosi box -- just test tests/test_framework.py::TestClassName::test_method`
- **Run mypy only:** `mkosi box -- mypy ./barrage ./tests`

## Architecture

Barrage is a concurrent async test framework for Python 3.12+ with zero external dependencies. It uses itself to run its own tests.

### Core modules (`barrage/`)

- **`runner.py`** — Execution engine. Runs test classes concurrently as asyncio tasks. Uses `contextvars.ContextVar` for per-task stdout/stderr capture so concurrent tests get isolated output.
- **`case.py`** — `AsyncTestCase` (base class, all hooks/tests are async) and `MonitoredTestCase` (adds background task crash monitoring with auto-cancel/skip).
- **`singleton.py`** — Singleton lifecycle with dependency injection. `Singleton` base class (async context manager), `singleton[T]` descriptor for declaring dependencies on both test classes and singleton classes, `SingletonManager` resolves dependencies recursively and tears down in reverse creation order.
- **`discovery.py`** — Test discovery. Supports directories, files, and `File::Class::method` selectors.
- **`result.py`** — `TestOutcome` dataclass and `AsyncTestResult` collector (asyncio.Lock-protected).
- **`subprocess.py`** — Async `spawn()` (context manager for long-lived processes) and `run()` helpers with PTY-aware output relaying through the capture system.
- **`taskgroups.py`** — `TaskGroup` extending `asyncio.TaskGroup` with `monitor_async_context()`.
- **`colorize.py`** — ANSI output formatting, respects `NO_COLOR`/`FORCE_COLOR` env vars.

## Python guidelines

- Don't use `TypeVar`, use the new generics syntax added in python 3.12.
- When setting up something that needs to be cleaned up, always use context managers to ensure proper cleanup. Use ExitStack or AsyncExitStack where needed to manage context managers.

## Commit guidelines

- Always add `Co-developed-by: Claude <claude@anthropic.com>` to any commits.
- Always do Signed off by for commits
