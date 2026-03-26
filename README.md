# barrage – concurrent async test framework

`barrage` is a test framework for Python built on `asyncio`. Unlike
traditional frameworks that run tests sequentially, `barrage` runs
tests **concurrently** — both across classes and, optionally, within
a single class. Tests within a class run sequentially by default;
pass `concurrent=True` to opt in to intra-class concurrency.

## Quick start

```python
from barrage import AsyncTestCase

class MyTests(AsyncTestCase):
    async def setUp(self) -> None:
        self.value = 42

    async def test_example(self) -> None:
        self.assertEqual(self.value, 42)

    async def test_other(self) -> None:
        self.assertIn("oo", "foobar")
```

All test methods must be `async def test_*`. The lifecycle hooks
(`setUp`, `tearDown`, `setUpClass`, `tearDownClass`) are all async too.

Run the tests:

```console
$ python3 -m barrage test/
```

## CLI reference

```
python3 -m barrage [options] [path ...]
```

| Option | Description |
|---|---|
| `-v`, `--verbose` | Per-test output lines |
| `-q`, `--quiet` | Summary only |
| `-p`, `--pattern PATTERN` | File name glob for directory discovery (default: `test_*.py`) |
| `--max-concurrency N` | Cap on concurrent test methods |
| `-i`, `--interactive` | Run tests sequentially with no output capture and live per-test status. Useful for debugging. |
| `--show-output` | Show captured stdout/stderr for all tests, including passing tests. |
| `-x`, `--failfast` | Stop on the first test failure or error. |

Paths may be suffixed with `::ClassName` to select a single test
class, or `::ClassName::test_method` to select a single test. When
no paths are given, tests are discovered in the current directory.

```console
# Discover and run all tests in the current directory
$ python3 -m barrage

# Run all tests in a directory
$ python3 -m barrage test/

# Run all tests in a specific file
$ python3 -m barrage test/test_example.py

# Run all tests in a specific class
$ python3 -m barrage test/test_example.py::MyTestClass

# Run a single test method
$ python3 -m barrage test/test_example.py::MyTestClass::test_method

# Multiple paths at once
$ python3 -m barrage test/test_foo.py test/test_bar.py::SomeClass

# Run a single test interactively for debugging
$ python3 -m barrage -i test/my_tests.py::MyTests::test_example
```

## Writing tests

### `AsyncTestCase`

The base class for tests. Each `test_*` method gets its own instance
(just like `unittest`), so instance attributes set in `setUp` are
isolated between tests.

**Lifecycle hooks** (all `async`, all optional):

| Hook | When |
|---|---|
| `setUpClass` | Once before any test in the class |
| `setUp` | Before each test method |
| `tearDown` | After each test method (even on failure) |
| `tearDownClass` | Once after all tests in the class |

**Assertion helpers** — same names as `unittest`:

`assertEqual`, `assertNotEqual`, `assertTrue`, `assertFalse`,
`assertIs`, `assertIsNot`, `assertIsNone`, `assertIsNotNone`,
`assertIn`, `assertNotIn`, `assertIsInstance`, `assertIsNotInstance`,
`assertGreater`, `assertGreaterEqual`, `assertLess`,
`assertLessEqual`, `assertAlmostEqual`, `assertRaises`, `fail`

**Skip support** — call `self.skipTest("reason")` to skip a test.

### Concurrency model

| Scope | Behaviour |
|---|---|
| **Across classes** | Always concurrent – every test class gets its own `asyncio.Task`. |
| **Within a class** | Sequential by default. Pass `concurrent=True` to opt in. |

```python
class Sequential(AsyncTestCase):
    ...

class Fast(AsyncTestCase, concurrent=True):
    ...
```

The child class inherits its parent's setting, but can override it:

```python
class Base(AsyncTestCase):
    ...

class Child(Base, concurrent=True):
    ...
```

#### Global concurrency limit

```console
$ python3 -m barrage --max-concurrency 8 test/
```

### `MonitoredTestCase`

Extends `AsyncTestCase` with background-task crash monitoring. Register
long-lived coroutines via `create_task()` or async context managers via
`monitor_async_context()`. If any background task fails unexpectedly, all
currently running tests are cancelled and remaining tests are skipped.

This is useful for integration tests that depend on an external
component (e.g. a VM or helper service) staying alive for the duration
of the test class.

```python
from barrage import MonitoredTestCase

class VMTests(MonitoredTestCase):
    @classmethod
    async def setUpClass(cls) -> None:
        await super().setUpClass()
        cls.vm, _ = await cls.monitor_async_context(start_vm())

    async def test_something(self) -> None:
        result = await self.vm.run("echo hello")
        assert result == "hello\n"
```

## Subprocess helpers

The `barrage.subprocess` module provides `spawn` and `run` for
launching subprocesses whose output is automatically captured by
barrage's per-test output capture.

When stdout/stderr are left as `None` (the default), output is
relayed through a PTY (preserving colours and line-buffering) when
the real standard stream is a TTY, or through a plain pipe otherwise.

```python
from barrage.subprocess import spawn, run, PIPE, DEVNULL

# Run a command to completion
result = await run(["ls", "-la"])

# Capture stdout
result = await run(["echo", "hello"], stdout=PIPE)

# Long-lived process with guaranteed cleanup
async with spawn(["my-server", "--port", "8080"]) as proc:
    ...  # interact with the server
# proc is killed & cleaned up here
```

Both functions raise `subprocess.CalledProcessError` on non-zero
exit codes by default (pass `check=False` to disable).

### Monitored subprocesses

Combine `spawn()` with `MonitoredTestCase.monitor_async_context()` to
run a long-lived subprocess that is monitored for the lifetime of a
test class. If the process exits unexpectedly, all running tests in
the class are cancelled and remaining tests are skipped.

```python
from barrage import MonitoredTestCase
from barrage.subprocess import spawn

class ServerTests(MonitoredTestCase):
    @classmethod
    async def setUpClass(cls) -> None:
        await super().setUpClass()
        cls.server, _ = await cls.monitor_async_context(
            spawn(["my-server", "--port", "8080"])
        )

    async def test_health(self) -> None:
        ...  # talk to self.server
```

Because `spawn()` is an async context manager, `monitor_async_context()`
enters it in a background task and monitors that task. The subprocess is
killed and cleaned up when the test class finishes or when the background
task is cancelled due to a crash.

## Singletons

A *singleton* is a resource that is expensive to create and tear down
(e.g. a pool of virtual machines, a database connection pool) and
should be shared across many test classes.

Singletons are declared as classes that inherit from `Singleton` and
implement the async context manager protocol. They are attached to
test classes using the `singleton` descriptor. The framework creates
each singleton once, injects it before `setUpClass`, and tears
everything down in reverse order when the test session ends.

```python
from typing import Self

from barrage import AsyncTestCase, Singleton, singleton

class VMManager(Singleton):
    async def __aenter__(self) -> Self:
        self.pool = await create_vm_pool()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.pool.shutdown()

class MyTests(AsyncTestCase):
    manager = singleton(VMManager)

    async def test_something(self) -> None:
        vm = await self.manager.acquire()
        result = await vm.run("uname -r")
        self.assertIn("6.", result)
```

### Default `__aexit__` behaviour

The base `Singleton` class provides a default `__aexit__` that blocks
forever (via an unresolved `asyncio.Future`), keeping the singleton's
background task alive until the framework cancels it during teardown.
Override `__aexit__` when you need custom cleanup logic.

### Dependency injection

Dependencies between singletons are declared using the same
`singleton` descriptor. If singleton A has a `singleton(B)` descriptor,
B is created and injected first. Circular dependencies are detected
and raise `RuntimeError`.

```python
class ResourceMonitor(Singleton):
    async def __aenter__(self) -> Self:
        ...

class VMManager(Singleton):
    monitor = singleton(ResourceMonitor)

    async def __aenter__(self) -> Self:
        ...

class MyTests(AsyncTestCase):
    monitor = singleton(ResourceMonitor)
    manager = singleton(VMManager)  # ResourceMonitor is created first
```

Multiple test classes that reference the same singleton class share a
single instance.

### Parameterised singletons

When a singleton needs configuration, define `__init_subclass__` to
accept keyword arguments. Configuration is captured at class
definition time; heavy initialization happens later in `__aenter__`.

```python
class Database(Singleton):
    url: str

    def __init_subclass__(cls, *, url: str, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        cls.url = url

    async def __aenter__(self) -> Self:
        self.conn = await connect(self.url)
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.conn.close()
```

Create named configurations using the class keyword syntax (fully
statically typed):

```python
class TestDB(Database, url="postgres://localhost/test"):
    pass

class StagingDB(Database, url="postgres://staging/app"):
    pass

class SchemaTests(AsyncTestCase):
    db = singleton(TestDB)

class MigrationTests(AsyncTestCase):
    test_db = singleton(TestDB)       # same instance as SchemaTests.db
    staging_db = singleton(StagingDB) # separate instance
```

Or pass keyword arguments directly to `singleton()` for a more
compact inline syntax:

```python
class SchemaTests(AsyncTestCase):
    db = singleton(Database, url="postgres://localhost/test")

class MigrationTests(AsyncTestCase):
    test_db = singleton(Database, url="postgres://localhost/test")  # same instance
    staging_db = singleton(Database, url="postgres://staging/app")  # separate
```

The inline form creates a subclass under the hood via
`__init_subclass__`. Identical `(cls, kwargs)` combinations produce
the same subclass, so deduplication works as usual. All values must
be hashable. The generated type gets a readable name like
`Database[url='postgres://localhost/test']`.
