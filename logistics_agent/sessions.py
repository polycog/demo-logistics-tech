"""
In-memory session storage
"""

import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field

# ===


@dataclass
class SessionData[T]:
    """
    Data from an individual session
    """

    seen: float = field(default=0, init=False)
    """
    time of last access
    """

    lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    """
    individual data lock
    """

    state: T
    """
    associated data
    """


# ===


class SessionStore[T]:
    """
    Hacktastic in-memory state storage
    """

    def __init__(
        self,
        entry_factory: Callable[[], T],
        ttl_seconds: int = 3600,
        max_sessions: int = 1000,
    ):
        self._ttl = ttl_seconds
        self._max_sessions = max_sessions

        self._sessions: OrderedDict[str, SessionData[T]] = OrderedDict()
        self._lock = threading.Lock()

        self._entry_factory = entry_factory

    def _evict(self, now: float) -> None:
        """
        Drop expired sessions; if still too many, drop the oldest.
        """

        # sessions ordered in increasing recency
        expired: list[str] = []
        for k, v in self._sessions.items():
            if now - v.seen > self._ttl:
                expired.append(k)
            else:
                # so once we hit something within
                # range, stop looking
                break

        for k in expired:
            del self._sessions[k]

        # ===

        overflow = max(len(self._sessions) - self._max_sessions, 0)

        for _ in range(overflow):
            # and if there are more to remove
            # start from the front
            self._sessions.popitem(last=False)

    @contextmanager
    def session(self, session_id: str) -> Generator[tuple[T, bool]]:
        """
        Yield (state, is_new) while holding that session's lock.
        """

        with self._lock:
            now = time.time()

            entry: SessionData[T] | None = self._sessions.get(session_id)
            is_new = entry is None
            if is_new:
                entry = SessionData(self._entry_factory())
                self._sessions[session_id] = entry

            assert entry is not None

            # timestamp entry
            entry.seen = now

            # by defn it's most recent, so back of the line
            self._sessions.move_to_end(session_id, last=True)

            # with entry update, clean up as necessary
            self._evict(now)

        with entry.lock:
            yield entry.state, is_new

    def reset(self, session_id: str) -> None:
        """
        Removes session_id from the store
        if it exists
        """

        with self._lock:
            self._sessions.pop(session_id, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)
