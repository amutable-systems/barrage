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
