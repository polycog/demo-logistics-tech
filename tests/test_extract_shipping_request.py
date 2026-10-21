"""offline tests of the ExtractShippingRequest sub-process

a fake ModelPopulator stands in for the LLM, answering as it would for each part or failing, so the sub-process runs
deterministically with no API key
"""

import pytest
from cognition.language.description import basemodel_description
from pydantic import ValidationError
from pydantic_ai.exceptions import UnexpectedModelBehavior

from logistics_agent import extract_shipping_request
from logistics_agent.communicator import Utterance
from logistics_agent.domain import City, Location, Shipment
from logistics_agent.extract_shipping_request import (
    ExtractDestination,
    ExtractOrigin,
    ExtractShipment,
    ExtractShippingRequest,
    ShippingRequestState,
    all_populated,
    extract_city,
    served_place_model,
)
from logistics_agent.reasoner import Reasoner

SEATTLE = Location(name="pickup1", city=City(name="seattle", display_name="Seattle"))
DENVER = Location(name="dropoff1", city=City(name="denver", display_name="Denver"))
BOOKS = Shipment(name="shipment1", details="box of books", weight=200)


def test_the_sub_process_starts_with_the_origin_destination_and_shipment_none():
    state = ExtractShippingRequest().state
    assert state.origin is None and state.destination is None and state.shipment is None


@pytest.mark.parametrize("parts", [
    {},
    {"origin": SEATTLE},
    {"origin": SEATTLE, "destination": DENVER},
    {"origin": SEATTLE, "shipment": BOOKS},
    {"destination": DENVER, "shipment": BOOKS},
])
def test_the_sub_process_is_not_done_while_any_of_the_three_is_none(parts):
    assert not all_populated(ShippingRequestState(**parts), None)


def test_the_sub_process_is_done_once_the_origin_destination_and_shipment_are_all_populated():
    assert all_populated(ShippingRequestState(origin=SEATTLE, destination=DENVER, shipment=BOOKS), None)


def test_the_sub_process_terminates_on_all_populated():
    assert all_populated in ExtractShippingRequest()._termination_checks


# === extracting the origin, destination, and shipment

PHRASE = "I need to ship a box of books, 200 pounds, from Seattle to Denver"


class FakeModelPopulator:
    """stands in for ModelPopulator: answers as the language model would for its task; records how each was built,
    and every call

    the canned result for a place is the display name of its city, or None, which is checked against the place model
    as the language model's answer would be, so a city the carrier does not serve fails; for the shipment it is a
    Shipment; an error is raised instead
    """

    results: dict = {}
    built: list[tuple] = []
    calls: list[tuple] = []

    def __init__(self, model_type, task_desc: str) -> None:
        self.model_type, self.task_desc = model_type, task_desc
        FakeModelPopulator.built.append((model_type, task_desc))

    def __call__(self, utterance: str, llm, *extra: str, **kwargs):
        FakeModelPopulator.calls.append((self.task_desc, utterance, llm))
        result = FakeModelPopulator.results[self.task_desc]
        if isinstance(result, Exception):
            raise result
        if self.model_type is Shipment:
            return result
        return self.model_type.model_validate({"city": result})


@pytest.fixture(autouse=True)
def fake_populator(monkeypatch):
    """replaces the LLM-backed ModelPopulator; returns the fake, whose results the test may change"""
    FakeModelPopulator.results = {
        ExtractOrigin.TASK_DESC: "Seattle", ExtractDestination.TASK_DESC: "Denver", ExtractShipment.TASK_DESC: BOOKS,
    }
    FakeModelPopulator.built, FakeModelPopulator.calls = [], []
    monkeypatch.setattr(extract_shipping_request, "ModelPopulator", FakeModelPopulator)
    return FakeModelPopulator


class IO:
    """stands in for the sub-process's io container, with the utterance and the reasoner supplied as arguments"""

    def __init__(self, phrase: str, reasoner: Reasoner | None = None) -> None:
        class A:
            pass

        A.utterance, A.reasoner = Utterance(phrase), reasoner or Reasoner()
        self.a = A


def place(location: Location | None) -> tuple[str, str] | None:
    """a location as its name and its city's name, or None"""
    return None if location is None else (location.name, location.city.name)


SHIPMENT = BOOKS.model_copy(update={"name": "shipment"})

# what each operator extracts: the operator, the part of the state it fills in, its flag, and what it keeps, with a
# place as its name and its city's name
PARTS = [
    (ExtractOrigin, "origin", "origin_extracted", ("shipping-origin", "seattle")),
    (ExtractDestination, "destination", "destination_extracted", ("shipping-destination", "denver")),
    (ExtractShipment, "shipment", "shipment_extracted", SHIPMENT),
]
PLACE_OPERATORS = [ExtractOrigin, ExtractDestination]


def kept(state: ShippingRequestState, part: str):
    """the part of the state, with a place as its name and its city's name"""
    value = getattr(state, part)
    return value if part == "shipment" else place(value)


def extract(op_type, state: ShippingRequestState | None = None) -> ShippingRequestState:
    """has a fresh operator of the type handle the phrase; checks that it applies; returns the state"""
    state = state or ShippingRequestState()
    op, io = op_type("op"), IO(PHRASE)
    assert op.can_perform(state, io)
    op.perform(state, io)
    return state


# --- the place model


def served_cities() -> list[City]:
    return Reasoner().find_entities(City)


def test_the_place_model_takes_a_city_the_carrier_serves_or_none():
    model = served_place_model(served_cities())
    assert model.model_validate({"city": "Seattle"}).city.name == "seattle"
    assert model.model_validate({}).city is None and model.model_validate({"city": None}).city is None


def test_the_place_model_refuses_a_city_the_carrier_does_not_serve():
    with pytest.raises(ValidationError):
        served_place_model(served_cities()).model_validate({"city": "Portland"})


def test_the_place_model_shows_the_language_model_every_city_the_carrier_serves():
    description = basemodel_description(served_place_model(served_cities()), True)
    assert all(f"* {name}" in description for name in ("Boston", "Denver", "Detroit", "San Francisco", "Seattle"))


def test_extract_city_gives_the_world_graphs_own_city():
    reasoner = Reasoner()
    city = extract_city(PHRASE, ExtractOrigin.TASK_DESC, reasoner)
    assert city in reasoner.find_entities(City) and city.name == "seattle"


# --- each operator


@pytest.mark.parametrize("op_type", PLACE_OPERATORS)
def test_each_place_is_populated_with_the_place_model_of_the_served_cities(fake_populator, op_type):
    extract(op_type)
    [(model_type, task_desc)] = fake_populator.built
    assert task_desc == op_type.TASK_DESC and model_type.model_validate({"city": "Boston"}).city.name == "boston"


def test_the_shipment_populator_is_set_up_once_for_the_shipment_model(fake_populator):
    ExtractShipment("op")
    assert fake_populator.built == [(Shipment, ExtractShipment.TASK_DESC)]


@pytest.mark.parametrize("op_type, part, flag, expected", PARTS)
def test_each_operator_fills_in_its_part_from_the_utterance(fake_populator, op_type, part, flag, expected):
    state = extract(op_type)
    assert kept(state, part) == expected and getattr(state, flag)
    assert fake_populator.calls == [(op_type.TASK_DESC, PHRASE, extract_shipping_request.DEFAULT_LLM)]


@pytest.mark.parametrize("op_type, part, flag, expected", PARTS)
def test_each_operator_leaves_its_part_none_when_the_language_model_fails(fake_populator, op_type, part, flag, expected):
    fake_populator.results[op_type.TASK_DESC] = UnexpectedModelBehavior(f"no {part}")
    state = extract(op_type)
    assert getattr(state, part) is None and getattr(state, flag)


@pytest.mark.parametrize("op_type", PLACE_OPERATORS)
@pytest.mark.parametrize("city", [None, "Portland"])
def test_a_place_is_left_none_when_its_city_is_not_one_the_carrier_serves(fake_populator, op_type, city):
    fake_populator.results[op_type.TASK_DESC] = city
    state = extract(op_type)
    assert state.origin is None and state.destination is None


@pytest.mark.parametrize("op_type, part, flag, expected", PARTS)
def test_each_operator_applies_once_per_utterance(fake_populator, op_type, part, flag, expected):
    fake_populator.results[op_type.TASK_DESC] = UnexpectedModelBehavior(f"no {part}")
    state = extract(op_type)
    assert not op_type("op").can_perform(state, IO(PHRASE))


@pytest.mark.parametrize("op_type, part", [(ExtractOrigin, "origin"), (ExtractDestination, "destination"),
                                           (ExtractShipment, "shipment")])
def test_each_operator_does_not_apply_once_its_part_is_populated(op_type, part):
    value = BOOKS if part == "shipment" else SEATTLE
    assert not op_type("op").can_perform(ShippingRequestState(**{part: value}), IO(PHRASE))


@pytest.mark.parametrize("shipment", [
    Shipment(name="shipment1", details=" ", weight=200),
    Shipment(name="shipment1", details="box of books", weight=0),
])
def test_a_shipment_without_details_or_a_weight_is_left_none(fake_populator, shipment):
    fake_populator.results[ExtractShipment.TASK_DESC] = shipment
    assert extract(ExtractShipment).shipment is None


def test_the_shipment_gets_its_default_name_whatever_name_the_language_model_fills_in():
    assert extract(ExtractShipment).shipment.name == "shipment" != BOOKS.name


# --- the sub-process


def test_all_three_operators_apply_together_at_the_start():
    state, io = ShippingRequestState(), IO(PHRASE)
    assert all(op_type("op").can_perform(state, io) for op_type in (ExtractOrigin, ExtractDestination, ExtractShipment))


def test_the_sub_process_ranks_every_action_equally():
    sub_process = ExtractShippingRequest()
    [evaluator] = sub_process._action_evaluators
    assert {rank.rank for rank in evaluator(sub_process.state, None, ["extract_origin", "extract_shipment"])} == {0}


def test_the_sub_process_extracts_the_origin_destination_and_shipment_and_finishes():
    result = ExtractShippingRequest()(utterance=Utterance(PHRASE), reasoner=Reasoner())
    assert result is not None
    assert place(result.origin) == ("shipping-origin", "seattle")
    assert place(result.destination) == ("shipping-destination", "denver")
    assert result.shipment == SHIPMENT


def test_the_sub_process_does_not_finish_when_a_city_is_not_one_the_carrier_serves(fake_populator):
    fake_populator.results[ExtractDestination.TASK_DESC] = "Portland"
    sub_process = ExtractShippingRequest()
    assert sub_process(utterance=Utterance(PHRASE), reasoner=Reasoner()) is None
    state = sub_process.state
    assert (place(state.origin), state.destination, state.shipment) == (("shipping-origin", "seattle"), None, SHIPMENT)


def test_restarting_the_sub_process_extracts_everything_again():
    sub_process = ExtractShippingRequest()
    sub_process(utterance=Utterance(PHRASE), reasoner=Reasoner())
    sub_process.reinit()
    state = sub_process.state
    assert state.origin is None and state.destination is None and state.shipment is None
    assert not (state.origin_extracted or state.destination_extracted or state.shipment_extracted)
