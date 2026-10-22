"""offline tests of the ShippingRequestFiller sub-process

a fake ModelPopulator stands in for the LLM, answering as it would for each part or failing, so the sub-process runs
deterministically with no API key
"""

import pytest
from cognition.language.description import basemodel_description
from pydantic import ValidationError
from pydantic_ai.exceptions import UnexpectedModelBehavior

from logistics_agent import fill_shipping_request
from logistics_agent.communicator import AskFor, Utterance, asked_context
from logistics_agent.domain import City, Location, Shipment, ShippingRequest
from logistics_agent.fill_shipping_request import (
    ComposeAsks,
    ExtractDestination,
    ExtractOrigin,
    ExtractShipment,
    ShippingRequestFiller,
    ShippingRequestState,
    TakeRequest,
    asks_composed,
    extract_city,
    served_place_model,
)
from logistics_agent.reasoner import Reasoner

SEATTLE = Location(name="pickup1", city=City(name="seattle", display_name="Seattle"))
DENVER = Location(name="dropoff1", city=City(name="denver", display_name="Denver"))
BOOKS = Shipment(name="shipment1", details="box of books", weight=200)
QUESTION = "Which city is the shipment going to?"


def test_the_sub_process_starts_with_no_request_nothing_tried_and_no_asks():
    state = ShippingRequestFiller().state
    assert state.request is None and state.tried == set() and state.asks is None


def test_the_sub_process_is_done_once_it_has_composed_the_asks_even_none():
    assert not asks_composed(ShippingRequestState(request=ShippingRequest()), None)
    assert asks_composed(ShippingRequestState(asks=[]), None)
    assert asks_composed(ShippingRequestState(asks=[AskFor(ShippingRequest, "origin")]), None)


def test_the_sub_process_terminates_on_asks_composed():
    assert asks_composed in ShippingRequestFiller()._termination_checks


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
        FakeModelPopulator.calls.append((self.task_desc, utterance, llm, extra))
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
    monkeypatch.setattr(fill_shipping_request, "ModelPopulator", FakeModelPopulator)
    return FakeModelPopulator


class IO:
    """stands in for the sub-process's io container, with the utterance, the question it answers, the reasoner, and
    the request supplied as arguments"""

    def __init__(
        self, phrase: str, reasoner: Reasoner | None = None, request: ShippingRequest | None = None,
        asked: str | None = None,
    ) -> None:
        class A:
            pass

        A.utterance, A.asked, A.reasoner = Utterance(phrase), asked, reasoner or Reasoner()
        A.request = request or ShippingRequest()
        self.a = A


def place(location: Location | None) -> tuple[str, str] | None:
    """a location as its name and its city's name, or None"""
    return None if location is None else (location.name, location.city.name)


SHIPMENT = BOOKS.model_copy(update={"name": "shipment"})

# what each operator extracts: the operator, the part of the request it fills in, and what it keeps, with a place as
# its name and its city's name
PARTS = [
    (ExtractOrigin, "origin", ("shipping-origin", "seattle")),
    (ExtractDestination, "destination", ("shipping-destination", "denver")),
    (ExtractShipment, "shipment", SHIPMENT),
]
PLACE_OPERATORS = [ExtractOrigin, ExtractDestination]
EXTRACT_OPERATORS = [ExtractOrigin, ExtractDestination, ExtractShipment]


def kept(request: ShippingRequest, part: str):
    """the part of the request, with a place as its name and its city's name"""
    value = getattr(request, part)
    return value if part == "shipment" else place(value)


def taken(request: ShippingRequest | None = None) -> ShippingRequestState:
    """a state that has taken the request, an empty one by default"""
    return ShippingRequestState(request=request or ShippingRequest())


def extract(op_type, state: ShippingRequestState | None = None, asked: str | None = None) -> ShippingRequestState:
    """has a fresh operator of the type handle the phrase; checks that it applies; returns the state"""
    state = state or taken()
    op, io = op_type("op"), IO(PHRASE, asked=asked)
    assert op.can_perform(state, io)
    op.perform(state, io)
    return state


# --- taking the request


def test_take_request_takes_a_copy_of_the_conductors_request_once():
    request = ShippingRequest(origin=SEATTLE)
    state, io, op = ShippingRequestState(), IO(PHRASE, request=request), TakeRequest("op")
    assert op.can_perform(state, io)
    op.perform(state, io)
    assert state.request == request and state.request is not request
    assert not op.can_perform(state, io)


@pytest.mark.parametrize("op_type", [*EXTRACT_OPERATORS, ComposeAsks])
def test_nothing_else_applies_before_the_request_is_taken(op_type):
    assert not op_type("op").can_perform(ShippingRequestState(), IO(PHRASE))


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


@pytest.mark.parametrize("op_type, part, expected", PARTS)
def test_each_operator_fills_in_its_part_from_the_utterance(fake_populator, op_type, part, expected):
    state = extract(op_type)
    assert kept(state.request, part) == expected and part in state.tried
    assert fake_populator.calls == [(op_type.TASK_DESC, PHRASE, fill_shipping_request.DEFAULT_LLM, ())]


@pytest.mark.parametrize("op_type", EXTRACT_OPERATORS)
def test_each_operator_reads_the_utterance_as_the_answer_to_the_question_asked(fake_populator, op_type):
    extract(op_type, asked=QUESTION)
    [(_, _, _, extra)] = fake_populator.calls
    assert extra == tuple(asked_context(QUESTION))


@pytest.mark.parametrize("op_type, part, expected", PARTS)
def test_each_operator_leaves_its_part_missing_when_the_language_model_fails(fake_populator, op_type, part, expected):
    fake_populator.results[op_type.TASK_DESC] = UnexpectedModelBehavior(f"no {part}")
    state = extract(op_type)
    assert getattr(state.request, part) is None and part in state.tried


@pytest.mark.parametrize("op_type", PLACE_OPERATORS)
@pytest.mark.parametrize("city", [None, "Portland"])
def test_a_place_is_left_missing_when_its_city_is_not_one_the_carrier_serves(fake_populator, op_type, city):
    fake_populator.results[op_type.TASK_DESC] = city
    state = extract(op_type)
    assert state.request.origin is None and state.request.destination is None


@pytest.mark.parametrize("op_type, part, expected", PARTS)
def test_each_operator_applies_once_per_utterance(fake_populator, op_type, part, expected):
    fake_populator.results[op_type.TASK_DESC] = UnexpectedModelBehavior(f"no {part}")
    state = extract(op_type)
    assert not op_type("op").can_perform(state, IO(PHRASE))


@pytest.mark.parametrize("op_type, part", [(ExtractOrigin, "origin"), (ExtractDestination, "destination"),
                                           (ExtractShipment, "shipment")])
def test_each_operator_does_not_apply_to_a_part_the_request_already_has(op_type, part):
    value = BOOKS if part == "shipment" else SEATTLE
    assert not op_type("op").can_perform(taken(ShippingRequest(**{part: value})), IO(PHRASE))


@pytest.mark.parametrize("shipment", [
    Shipment(name="shipment1", details=" ", weight=200),
    Shipment(name="shipment1", details="box of books", weight=0),
])
def test_a_shipment_without_details_or_a_weight_is_left_missing(fake_populator, shipment):
    fake_populator.results[ExtractShipment.TASK_DESC] = shipment
    assert extract(ExtractShipment).request.shipment is None


def test_the_shipment_gets_its_default_name_whatever_name_the_language_model_fills_in():
    assert extract(ExtractShipment).request.shipment.name == "shipment" != BOOKS.name


# --- composing the asks


def test_compose_asks_waits_until_every_missing_part_has_been_tried():
    state = taken(ShippingRequest(origin=SEATTLE))
    state.tried = {"destination"}
    assert not ComposeAsks("op").can_perform(state, IO(PHRASE))
    state.tried.add("shipment")
    assert ComposeAsks("op").can_perform(state, IO(PHRASE))


def test_compose_asks_asks_for_each_part_still_missing_in_order():
    state = taken(ShippingRequest(destination=DENVER))
    state.tried = {"origin", "shipment"}
    ComposeAsks("op").perform(state, IO(PHRASE))
    assert state.asks == [AskFor(ShippingRequest, "origin"), AskFor(ShippingRequest, "shipment")]


def test_compose_asks_asks_for_nothing_once_the_request_is_complete_and_applies_once():
    state, op = taken(ShippingRequest(origin=SEATTLE, destination=DENVER, shipment=BOOKS)), ComposeAsks("op")
    assert op.can_perform(state, IO(PHRASE))
    op.perform(state, IO(PHRASE))
    assert state.asks == [] and not op.can_perform(state, IO(PHRASE))


# --- the sub-process


def test_all_three_operators_apply_together_once_an_empty_request_is_taken():
    state, io = taken(), IO(PHRASE)
    assert all(op_type("op").can_perform(state, io) for op_type in EXTRACT_OPERATORS)


def test_the_sub_process_ranks_every_action_equally():
    sub_process = ShippingRequestFiller()
    [evaluator] = sub_process._action_evaluators
    assert {rank.rank for rank in evaluator(sub_process.state, None, ["extract_origin", "extract_shipment"])} == {0}


def fill(request: ShippingRequest | None = None, asked: str | None = None) -> ShippingRequestState | None:
    """runs a fresh sub-process over the phrase, as the conductor does"""
    return ShippingRequestFiller()(
        utterance=Utterance(PHRASE), asked=asked, reasoner=Reasoner(), request=request or ShippingRequest()
    )


def test_the_sub_process_fills_in_the_whole_request_and_asks_for_nothing():
    result = fill()
    assert result is not None and result.asks == []
    assert place(result.request.origin) == ("shipping-origin", "seattle")
    assert place(result.request.destination) == ("shipping-destination", "denver")
    assert result.request.shipment == SHIPMENT


def test_the_sub_process_asks_for_a_part_the_utterance_does_not_give(fake_populator):
    fake_populator.results[ExtractDestination.TASK_DESC] = "Portland"
    result = fill()
    assert result is not None and result.asks == [AskFor(ShippingRequest, "destination")]
    request = result.request
    assert (place(request.origin), request.destination, request.shipment) == (
        ("shipping-origin", "seattle"), None, SHIPMENT
    )


def test_the_sub_process_extracts_only_the_parts_missing_from_the_request_it_is_given(fake_populator):
    result = fill(ShippingRequest(origin=SEATTLE, shipment=BOOKS), asked=QUESTION)
    assert result is not None and result.asks == []
    assert (result.request.origin, result.request.shipment) == (SEATTLE, BOOKS)
    assert [(task_desc, extra) for task_desc, _, _, extra in fake_populator.calls] == [
        (ExtractDestination.TASK_DESC, tuple(asked_context(QUESTION)))
    ]


def test_the_sub_process_leaves_the_request_it_is_given_as_it_was():
    request = ShippingRequest(origin=SEATTLE)
    fill(request)
    assert request == ShippingRequest(origin=SEATTLE)


def test_restarting_the_sub_process_fills_in_from_scratch():
    sub_process = ShippingRequestFiller()
    sub_process(utterance=Utterance(PHRASE), asked=None, reasoner=Reasoner(), request=ShippingRequest())
    sub_process.reinit()
    state = sub_process.state
    assert state.request is None and state.tried == set() and state.asks is None
