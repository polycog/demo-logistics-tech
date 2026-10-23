"""
Chat app backend
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

# ===


class ChatAppBackend[T](Protocol):
    """
    Chat backend, with
    associated session type
    """

    @property
    def title(self) -> str:
        """app title"""

    @property
    def greeting(self) -> str:
        """app greeting"""

    @property
    def allow_unsafe_output(self) -> bool:
        """
        if True, greeting and replies allow HTML;
        importantly, this means NOT to include
        raw user msg in response
        """

    def session_init(self) -> T:
        """produces initial session state"""

    def respond(self, utterance: str, state: T) -> str:
        """produces chat responses"""


@dataclass(frozen=True)
class SimpleChatAppBackend[T]:
    """
    Simple data class impl of
    ChatAppBackend
    """

    title: str
    """app title"""

    greeting: str
    """app greeting"""

    session_init: Callable[[], T]
    """method of producing initial session state"""

    respond: Callable[[str, T], str]
    """method of producing chat responses"""

    allow_unsafe_output: bool = False
    """
    if True, greeting and replies allow HTML;
    importantly, this means NOT to include
    raw user msg in response
    """
