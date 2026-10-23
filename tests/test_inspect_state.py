"""offline tests of the StateInspector sub-process, which answers questions about the state of the world from the world
graph

a fake EnumClassifier and a fake ModelPopulator stand in for the LLM, answering with the kind of question and what it
is about that they are given, or failing, so the sub-process runs deterministically with no API key
"""

import pytest
from pydantic_ai.exceptions import UnexpectedModelBehavior

from logistics_agent import inspect_state
from logistics_agent.communicator import AskFor, Utterance, asked_context, in_context
from logistics_agent.domain import At, ShippingRequest, Truck, VehicleType
from logistics_agent.inspect_state import (
    ClassifyQuestion, FillFleetQuestion, FleetQuestion, QuestionType, StateInspector, place_of,
)
from logistics_agent.reasoner import Reasoner

PHRASE = "which trucks do you have?"


class FakeEnumClassifier:
    """stands in for EnumClassifier: answers with the kind of question in `question_type`; records how it was built,
    and every text it classified"""

    question_type: QuestionType | None = QuestionType.FLEET
    built: list[tuple] = []
    calls: list[str] = []

    def __init__(self, enum_type, task_desc: str) -> None:
        FakeEnumClassifier.built.append((enum_type, task_desc))

    def __call__(self, text: str, llm, **kwargs):
        FakeEnumClassifier.calls.append(text)
        return FakeEnumClassifier.question_type, 1.0


class FakeModelPopulator:
    """stands in for ModelPopulator: answers with the fields in `given`, or raises the error in `given`; records how it
    was built, and every call"""

    given: dict | Exception = {}
    built: list[tuple] = []
    calls: list[tuple] = []

    def __init__(self, model_type, task_desc: str) -> None:
        self.model_type = model_type
        FakeModelPopulator.built.append((model_type, task_desc))

    def __call__(self, utterance: str, llm, *extra: str, **kwargs):
        FakeModelPopulator.calls.append((utterance, extra))
        if isinstance(FakeModelPopulator.given, Exception):
            raise FakeModelPopulator.given
        return self.model_type.model_validate(FakeModelPopulator.given)


@pytest.fixture(autouse=True)
def fake_language(monkeypatch):
    """replaces the LLM-backed EnumClassifier and ModelPopulator"""
    FakeEnumClassifier.question_type, FakeEnumClassifier.built, FakeEnumClassifier.calls = QuestionType.FLEET, [], []
    FakeModelPopulator.given, FakeModelPopulator.built, FakeModelPopulator.calls = {"kind": "truck"}, [], []
    monkeypatch.setattr(inspect_state, "EnumClassifier", FakeEnumClassifier)
    monkeypatch.setattr(inspect_state, "ModelPopulator", FakeModelPopulator)


def inspect(reasoner: Reasoner | None = None, answering=None):
    """runs a fresh StateInspector over the phrase, as the conductor does"""
    return StateInspector()(
        utterance=Utterance(PHRASE), answering=answering, reasoner=reasoner or Reasoner(), suppress_errors=False
    )


def answer(result) -> list[tuple[str, str | None]]:
    """the sightings of the result, as each vehicle's name and its place's name"""
    return [(s.vehicle.name, None if s.place is None else s.place.name) for s in result.sightings]


# --- understanding the question


def test_the_question_is_classified_among_the_question_types_and_its_subject_filled_with_the_fleet_question():
    ClassifyQuestion("op")
    FillFleetQuestion("op")
    assert FakeEnumClassifier.built == [(QuestionType, ClassifyQuestion.TASK_DESC)]
    assert FakeModelPopulator.built == [(FleetQuestion, FillFleetQuestion.TASK_DESC)]


def test_the_question_is_read_together_with_the_question_the_agent_asked():
    ask = AskFor(ShippingRequest, "destination")
    inspect(answering=ask)
    assert FakeEnumClassifier.calls == [in_context(ask.question, PHRASE)]
    assert FakeModelPopulator.calls == [(PHRASE, tuple(asked_context(ask.question)))]


def test_a_question_of_no_known_type_is_not_looked_up():
    FakeEnumClassifier.question_type = None
    result = inspect()
    assert result.question_type is None and result.sightings is None and FakeModelPopulator.calls == []


# --- the fleet


def test_the_trucks_are_looked_up_in_the_world_graph_with_where_each_is():
    result = inspect()
    assert result.fleet_question.kind is VehicleType.TRUCK
    assert answer(result) == [("truck1", "terminal_seattle"), ("truck2", "terminal_san_francisco")]


@pytest.mark.parametrize("given", [{}, {"kind": None}, UnexpectedModelBehavior("no answer")])
def test_without_a_kind_the_whole_fleet_is_looked_up(given):
    FakeModelPopulator.given = given
    result = inspect()
    assert result.fleet_question.kind is None
    assert answer(result) == [("air1", "SEA"), ("truck1", "terminal_seattle"), ("truck2", "terminal_san_francisco")]


def test_a_vehicle_added_to_the_fleet_is_in_the_answer():
    reasoner = Reasoner()
    truck = Truck(name="truck3", display_name="Road Runner", model="Ford F-750", max_payload=12000,
                  license_plate="7XYZ123")
    reasoner.add_entity(truck)
    reasoner.add_relation(At(entity1=truck, entity2=reasoner.find_entity("terminal_denver")))
    assert ("truck3", "terminal_denver") in answer(inspect(reasoner))


def test_a_vehicle_the_world_graph_does_not_place_is_in_the_answer_with_no_place():
    reasoner = Reasoner()
    reasoner.add_entity(Truck(name="truck3", display_name="Road Runner", model="Ford F-750", max_payload=12000,
                              license_plate="7XYZ123"))
    assert ("truck3", None) in answer(inspect(reasoner))


def test_place_of_gives_where_a_vehicle_is_or_none():
    reasoner = Reasoner()
    [bertha] = [v for v in reasoner.find_entities(Truck) if v.name == "truck1"]
    assert place_of(bertha, reasoner).name == "terminal_seattle"
