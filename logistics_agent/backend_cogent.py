"""
Chat app backend
based on a cogent
"""

from contextlib import suppress
from typing import Any, override

from cognition import (
    Cogent,
    DecisionProcess,
    Function,
    MutableWrapper,
    Supplier,
    create_actuator,
    create_sensor,
)

from logistics_agent.backend import ChatAppBackend

# ===


class ManagedStateDecisionProcess[S](DecisionProcess[S]):
    """
    DP allowing for external storage of state
    """

    def __init__(
        self,
        session_initializer: Supplier[S],
        enable_terminal_check: bool = True,
    ) -> None:
        self.__local_s = session_initializer()
        self.__session_init = session_initializer

        super().__init__(self.__mock_init, enable_terminal_check)

    def __mock_init(self) -> S:
        # supports self-reinit states
        with suppress(TypeError):
            self.__local_s()  # type: ignore[operator]

        return self.__local_s

    @property
    def session_initializer(self) -> Supplier[S]:
        """
        Provides appropriate initial value
        with which to initialize managed
        state store.
        """

        return self.__session_init

    @property
    def session_state(self) -> S:
        """
        Current managed state value
        (just for debugging - indirectly
        accessed via dp.reinit())
        """

        return self.__local_s

    @session_state.setter
    def session_state(self, val: S) -> None:
        self.__local_s = val


# ===

def _identity(arg_in: Any) -> Any:
    return arg_in

# pylint: disable=too-many-instance-attributes
class CogentChatAppBackend[T](ChatAppBackend[T]):
    """
    Impl of chat app backend where
    session state is DP state
    """

    SENSOR_NAME: str = "chat"
    ACTUATOR_NAME: str = "chat"

    # ===

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        title: str,
        greeting: str,
        cogent: Cogent[ManagedStateDecisionProcess[T]],
        allow_unsafe_output: bool = False,
        sensor_name: str = SENSOR_NAME,
        actuator_name: str = ACTUATOR_NAME,
        sensor_translation: Function[str, Any] = _identity,
        actuator_translation: Function[Any, str] = _identity,
    ) -> None:
        self._title = title
        self._greeting = greeting
        self._allow_unsafe_output = allow_unsafe_output
        self._cogent = cogent

        self._sensor_name = sensor_name
        self._actuator_name = actuator_name

        # ===

        self._msg_in = MutableWrapper[str]("")

        self._cogent.add_sensor(
            create_sensor(self._sensor_name, lambda: sensor_translation(self._msg_in.value))
        )

        # ===

        self._response_out: list[str] = []

        def _append_response(s: str) -> bool:
            self._response_out.append(actuator_translation(s))
            return True

        self._cogent.add_actuator(
            create_actuator(
                self._actuator_name,
                _append_response,
            )
        )

    @override
    @property
    def title(self) -> str:
        return self._title

    @override
    @property
    def greeting(self) -> str:
        return self._greeting

    @property
    def allow_unsafe_output(self) -> bool:
        return self._allow_unsafe_output

    def session_init(self) -> T:
        return self._cogent.dp.session_initializer()

    def _gate_run_until_response(self, _c: Cogent[Any]) -> bool:
        return not self._response_out

    def respond(self, utterance: str, state: T) -> str:
        self._msg_in.value = utterance
        self._response_out.clear()
        self._cogent.dp.session_state = state

        # print("== BEFORE ==")
        # print(state)
        # print("============")
        self._cogent(self._gate_run_until_response)
        # print("== AFTER ==")
        # print(state)
        # print("===========")

        return "".join(self._response_out)
