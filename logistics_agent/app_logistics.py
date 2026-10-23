"""
Logistics -> web
"""

from cognition import Cogent
from flask import Flask

from logistics_agent.app import create_app
from logistics_agent.assistant import ConversationState, add_dp_ops
from logistics_agent.backend_cogent import CogentChatAppBackend, ManagedStateDecisionProcess
from logistics_agent.communicator import Utterance
from logistics_agent.conductor import INTRODUCTION

# ===

SENSOR_ACTUATOR_NAME: str = "interaction"

# ===

def _translate_sensor(s: str) -> Utterance:
    # print(f"SENSE: {s}")
    return Utterance(phrase=s)

def _translate_actuator(u: Utterance) -> str:
    # print(f"ACTUATOR: {u}")
    return u.phrase

def _init_state() -> ConversationState:
    result = ConversationState()
    result.introduced = True

    return result

def logisticsbot() -> Flask:
    """
    produces a flask app that
    uses a cogent as a logistics
    assistant
    """

    cogent = Cogent(add_dp_ops(ManagedStateDecisionProcess(_init_state)))

    return create_app(
        CogentChatAppBackend(
            title="LogisticsBot",
            greeting=INTRODUCTION,
            cogent=cogent,
            sensor_name=SENSOR_ACTUATOR_NAME,
            actuator_name=SENSOR_ACTUATOR_NAME,
            sensor_translation=_translate_sensor,
            actuator_translation=_translate_actuator,
        )
    )
