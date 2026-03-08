# SPDX-License-Identifier: MIT

"""
Custom selector that supports ``EPOLLPRI`` for PSI trigger files.

Linux PSI (Pressure Stall Information) trigger files notify via
``POLLPRI``/``EPOLLPRI``, but asyncio's ``add_reader()`` only
registers ``EPOLLIN``.  :class:`PriSelector` overrides registration
so that file descriptors listed in the class-level :attr:`pri_fds`
set are watched for ``EPOLLPRI`` instead of ``EPOLLIN``.
"""

import os
import select
import selectors
from typing import Any, ClassVar


class PriSelector(selectors.EpollSelector):
    """EpollSelector that watches marked fds for ``EPOLLPRI``.

    File descriptors added to the class-level :attr:`pri_fds` set
    *before* being registered via ``loop.add_reader()`` will be
    watched for ``EPOLLPRI`` instead of ``EPOLLIN``.

    Uses ``fileno()`` and ``select.epoll.fromfd()`` to access the
    underlying epoll instance without touching private attributes.
    """

    pri_fds: ClassVar[set[int]] = set()

    def register(
        self,
        fileobj: int | Any,
        events: int,
        data: Any = None,
    ) -> selectors.SelectorKey:
        key = super().register(fileobj, events, data)
        if key.fd in self.pri_fds:
            # Get a handle to the underlying epoll via a dup'd fd
            # so closing it doesn't affect the selector.
            dup_fd = os.dup(self.fileno())
            try:
                ep = select.epoll.fromfd(dup_fd)
                try:
                    ep.modify(key.fd, select.EPOLLPRI)
                finally:
                    ep.close()
            except BaseException:
                os.close(dup_fd)
                raise
        return key
