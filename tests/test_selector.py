# SPDX-License-Identifier: MIT

"""
Tests for the PriSelector custom epoll selector.

Run with::

    python3 -m barrage tests/test_selector.py
"""

import asyncio
import os
import selectors

from barrage.case import AsyncTestCase
from barrage.selector import PriSelector


class TestPriSelector(AsyncTestCase, concurrent=True):
    async def test_normal_fd_gets_epollin(self) -> None:
        """Fds not in pri_fds are registered with EPOLLIN."""
        r, w = os.pipe()
        try:
            sel = PriSelector()
            try:
                key = sel.register(r, selectors.EVENT_READ)
                self.assertEqual(key.fd, r)

                # Verify the fd is watchable: writing makes it
                # readable, so select should return it.
                os.write(w, b"x")
                ready = sel.select(timeout=0.1)
                fds = [k.fd for k, _ev in ready]
                self.assertIn(r, fds)
            finally:
                sel.close()
        finally:
            os.close(r)
            os.close(w)

    async def test_pri_fd_gets_epollpri(self) -> None:
        """Fds in pri_fds are registered with EPOLLPRI instead."""
        r, w = os.pipe()
        try:
            PriSelector.pri_fds.add(r)
            sel = PriSelector()
            try:
                key = sel.register(r, selectors.EVENT_READ)
                self.assertEqual(key.fd, r)

                # Verify EPOLLPRI was set by checking via a dup'd
                # epoll handle.  Writing to the pipe produces
                # EPOLLIN, not EPOLLPRI, so select should NOT
                # return the fd.
                os.write(w, b"x")
                ready = sel.select(timeout=0.1)
                fds = [k.fd for k, _ev in ready]
                self.assertNotIn(r, fds)
            finally:
                sel.close()
        finally:
            PriSelector.pri_fds.discard(r)
            os.close(r)
            os.close(w)

    async def test_pri_fds_is_class_level(self) -> None:
        """pri_fds is shared across all PriSelector instances."""
        r, _ = os.pipe()
        try:
            PriSelector.pri_fds.add(r)
            sel1 = PriSelector()
            sel2 = PriSelector()
            self.assertIn(r, sel1.pri_fds)
            self.assertIn(r, sel2.pri_fds)
            sel1.close()
            sel2.close()
        finally:
            PriSelector.pri_fds.discard(r)
            os.close(r)
            os.close(_)

    async def test_unregister_restores_state(self) -> None:
        """Unregistering a pri fd removes it from epoll cleanly."""
        r, w = os.pipe()
        try:
            PriSelector.pri_fds.add(r)
            sel = PriSelector()
            try:
                sel.register(r, selectors.EVENT_READ)
                sel.unregister(r)

                # Re-register without pri — should work normally.
                PriSelector.pri_fds.discard(r)
                sel.register(r, selectors.EVENT_READ)
                os.write(w, b"x")
                ready = sel.select(timeout=0.1)
                fds = [k.fd for k, _ev in ready]
                self.assertIn(r, fds)
            finally:
                sel.close()
        finally:
            PriSelector.pri_fds.discard(r)
            os.close(r)
            os.close(w)

    async def test_loop_uses_pri_selector(self) -> None:
        """The running event loop uses PriSelector as its selector."""
        loop = asyncio.get_running_loop()
        self.assertIsInstance(
            loop._selector,  # type: ignore[attr-defined]
            PriSelector,
        )

    async def test_add_reader_triggers_callback(self) -> None:
        """add_reader fires the callback when a normal fd becomes readable."""
        r, w = os.pipe()
        try:
            loop = asyncio.get_running_loop()
            ready: asyncio.Future[bool] = loop.create_future()

            def on_readable() -> None:
                loop.remove_reader(r)
                ready.set_result(True)

            loop.add_reader(r, on_readable)
            os.write(w, b"x")
            result = await asyncio.wait_for(ready, timeout=1.0)
            self.assertTrue(result)
        finally:
            os.close(r)
            os.close(w)

    async def test_add_reader_pri_fd_uses_epollpri(self) -> None:
        """add_reader with a pri fd uses EPOLLPRI via the loop."""
        r, w = os.pipe()
        try:
            loop = asyncio.get_running_loop()
            PriSelector.pri_fds.add(r)
            try:
                triggered = loop.create_future()

                def on_event() -> None:
                    loop.remove_reader(r)
                    triggered.set_result(True)

                loop.add_reader(r, on_event)

                # Writing to a pipe produces EPOLLIN, not
                # EPOLLPRI, so the callback should NOT fire.
                os.write(w, b"x")
                await asyncio.sleep(0.1)
                self.assertFalse(triggered.done())

                loop.remove_reader(r)
            finally:
                PriSelector.pri_fds.discard(r)
        finally:
            os.close(r)
            os.close(w)
