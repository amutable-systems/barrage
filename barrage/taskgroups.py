# SPDX-License-Identifier: MIT
import asyncio
from contextlib import AbstractAsyncContextManager


class TaskGroup(asyncio.TaskGroup):
    async def monitor_async_context[T](
        self, cm: AbstractAsyncContextManager[T], name: str | None = None
    ) -> tuple[T, asyncio.Task[None]]:
        """Enter an async context manager and monitor it for failures.

        The context manager's ``__aenter__`` and ``__aexit__`` both run
        inside a single monitored background task. This is important
        because async context managers that use ``asyncio.TaskGroup``
        internally record the *current task* as the parent task during
        ``__aenter__``. By running both methods in the same background
        task, structured concurrency is preserved: if a child task in
        the TaskGroup fails, the TaskGroup cancels its parent (the
        background task), which triggers ``__aexit__`` and propagates
        the error through ``_on_task_done``, aborting the task group.

        The context manager's ``__aexit__`` must block until all of its
        background work is finished (e.g. by awaiting a
        ``TaskGroup``).  This is what keeps the monitored background
        task alive for the lifetime of the component.  When the
        background task is cancelled during teardown, ``__aexit__``
        receives the cancellation and can clean up its resources.
        """
        assert self._loop
        ready: asyncio.Future[T] = self._loop.create_future()

        async def run_cm() -> None:
            try:
                async with cm as value:
                    ready.set_result(value)
            except BaseException as ex:
                if not ready.done():
                    ready.set_exception(ex)
                    return
                raise

        task = self.create_task(run_cm(), name=name)
        return await ready, task
