"""offline tests of the ShippingRequestFiller sub-process

a fake ModelPopulator stands in for the LLM, answering with the parts it is given or failing, so the sub-process runs
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
    ExtractParts,
    ShippingRequestFiller,
    ShippingRequestState,
    TakeRequest,
    asks_composed,
    city_named,
    parts_model,
)
from logistics_agent.reasoner import Reasoner

SEATTLE = Location(name="pickup1", city=City(name="seattle", display_name="Seattle"))
DENVER = Location(name="dropoff1", city=City(name="denver", display_name="Denver"))
BOOKS = Shipment(name="shipment1", details="box of books", weight=200)
SHIPMENT = Shipment(name="shipment", details="box of books", weight=200)
ASK_ORIGIN = AskFor(ShippingRequest, "origin")
ALL_PARTS = ["origin", "destination", "shipment"]

PHRASE = "I need to ship a box of books, 200 pounds, from Seattle to Denver"
# what the language model gives for the phrase, part by part
GIVEN = {"origin": "Seattle", "destination": "Denver", "shipment": {"details": "box of books", "weight": 200}}


def test_the_sub_process_starts_with_no_request_nothing_to_fill_nothing_extracted_and_no_asks():
    state = ShippingRequestFiller().state
    assert state.request is None and state.to_fill == [] and not state.extracted and state.asks is None


def test_the_sub_process_is_done_once_it_has_composed_the_asks_even_none():
    assert not asks_composed(ShippingRequestState(request=ShippingRequest()), None)
    assert asks_composed(ShippingRequestState(asks=[]), None)
    assert asks_composed(ShippingRequestState(asks=[ASK_ORIGIN]), None)


def test_the_sub_process_terminates_on_asks_composed():
    assert asks_composed in ShippingRequestFiller()._termination_checks


class FakeModelPopulator:
    """stands in for ModelPopulator: answers with the parts in `given` that the model it was built for has, or raises
    the error in `given`; records how each was built, and every call"""

    given: dict | Exception = {}
    built: list[tuple] = []
    calls: list[tuple] = []

    def __init__(self, model_type, task_desc: str) -> None:
        self.model_type, self.task_desc = model_type, task_desc
        FakeModelPopulator.built.append((model_type, task_desc))

    def __call__(self, utterance: str, llm, *extra: str, **kwargs):
        FakeModelPopulator.calls.append((utterance, llm, extra))
        if isinstance(FakeModelPopulator.given, Exception):
            raise FakeModelPopulator.given
        fields = self.model_type.model_fields
        return self.model_type.model_validate({k: v for k, v in FakeModelPopulator.given.items() if k in fields})


@pytest.fixture(autouse=True)
def fake_populator(monkeypatch):
    """replaces the LLM-backed ModelPopulator; returns the fake, whose answer the test may change"""
    FakeModelPopulator.given = dict(GIVEN)
    FakeModelPopulator.built, FakeModelPopulator.calls = [], []
    monkeypatch.setattr(fill_shipping_request, "ModelPopulator", FakeModelPopulator)
    return FakeModelPopulator


class IO:
    """stands in for the sub-process's io container, with the utterance, the ask it answers, the reasoner, and the
    request supplied as arguments"""

    def __init__(
        self, phrase: str = PHRASE, request: ShippingRequest | None = None, answering: AskFor | None = None,
    ) -> None:
        class A:
            pass

        A.utterance, A.answering, A.reasoner = Utterance(phrase), answering, Reasoner()
        A.request = request or ShippingRequest()
        self.a = A


def place(location: Location | None) -> tuple[str, str] | None:
    """a location as its name and its city's name, or None"""
    return None if location is None else (location.name, location.city.name)


def taken(request: ShippingRequest | None = None) -> ShippingRequestState:
    """a state that has taken the request, an empty one by default, as TakeRequest leaves it"""
    state = ShippingRequestState()
    TakeRequest("op").perform(state, IO(request=request))
    return state


def extract(state: ShippingRequestState | None = None, answering: AskFor | None = None) -> ShippingRequestState:
    """has a fresh ExtractParts handle the phrase; checks that it applies; returns the state"""
    state = state or taken()
    op, io = ExtractParts("op"), IO(answering=answering)
    assert op.can_perform(state, io)
    op.perform(state, io)
    return state


# --- taking the request


def test_take_request_takes_a_copy_of_the_conductors_request_once_to_fill_in_its_missing_parts():
    request = ShippingRequest(origin=SEATTLE)
    state, io, op = ShippingRequestState(), IO(request=request), TakeRequest("op")
    assert op.can_perform(state, io)
    op.perform(state, io)
    assert state.request == request and state.request is not request
    assert state.to_fill == ["destination", "shipment"]
    assert not op.can_perform(state, io)


@pytest.mark.parametrize("op_type", [ExtractParts, ComposeAsks])
def test_nothing_else_applies_before_the_request_is_taken(op_type):
    assert not op_type("op").can_perform(ShippingRequestState(), IO())


# --- the parts model


def served_cities() -> list[City]:
    return Reasoner().find_entities(City)


def test_the_parts_model_has_only_the_parts_to_fill_each_optional():
    model = parts_model(["destination", "shipment"], served_cities())
    assert set(model.model_fields) == {"destination", "shipment"}
    empty = model.model_validate({})
    assert empty.destination is None and empty.shipment is None


def test_the_parts_model_takes_a_shipment_without_its_weight_so_the_other_parts_are_kept():
    parts = parts_model(ALL_PARTS, served_cities()).model_validate({"origin": "Seattle", "shipment": {"details": "x"}})
    assert parts.origin == "Seattle" and parts.shipment.weight is None


def test_the_parts_model_asks_with_the_models_own_questions_and_names_every_city_the_carrier_serves():
    description = basemodel_description(parts_model(ALL_PARTS, served_cities()), True)
    for part in ALL_PARTS:
        assert ShippingRequest.model_fields[part].description in description
    assert all(name in description for name in ("Boston", "Denver", "Detroit", "San Francisco", "Seattle"))


@pytest.mark.parametrize("name, expected", [
    ("Seattle", "seattle"), ("san francisco", "san_francisco"), (" Denver ", "denver"), ("Portland", None), (None, None),
])
def test_city_named_gives_the_world_graphs_own_city_or_none_for_a_city_the_carrier_does_not_serve(name, expected):
    cities = served_cities()
    city = city_named(name, cities)
    assert (city.name if city else None) == expected and (city is None or city in cities)


# --- extracting the parts


def test_extract_parts_fills_in_every_missing_part_in_one_call(fake_populator):
    state = extract()
    request = state.request
    assert place(request.origin) == ("shipping-origin", "seattle")
    assert place(request.destination) == ("shipping-destination", "denver")
    assert request.shipment == SHIPMENT and state.extracted
    [(model_type, task_desc)] = fake_populator.built
    assert set(model_type.model_fields) == set(ALL_PARTS) and task_desc == ExtractParts.TASK_DESC
    assert fake_populator.calls == [(PHRASE, fill_shipping_request.DEFAULT_LLM, ())]


def test_extract_parts_fills_in_only_the_parts_missing_from_the_request(fake_populator):
    state = extract(taken(ShippingRequest(origin=SEATTLE, shipment=BOOKS)))
    [(model_type, _)] = fake_populator.built
    assert set(model_type.model_fields) == {"destination"}
    assert (state.request.origin, state.request.shipment) == (SEATTLE, BOOKS)
    assert place(state.request.destination) == ("shipping-destination", "denver")


def test_extract_parts_reads_the_utterance_as_the_answer_to_the_question_asked(fake_populator):
    extract(answering=ASK_ORIGIN)
    [(_, _, extra)] = fake_populator.calls
    assert extra == tuple(asked_context(ASK_ORIGIN.question))


def test_an_answer_with_more_than_was_asked_fills_in_every_part_it_gives(fake_populator):
    fake_populator.given = {"origin": "Boston", "destination": "Denver"}
    state = extract(answering=ASK_ORIGIN)
    assert place(state.request.origin) == ("shipping-origin", "boston")
    assert place(state.request.destination) == ("shipping-destination", "denver")
    assert state.request.shipment is None


@pytest.mark.parametrize("part", ["origin", "destination"])
@pytest.mark.parametrize("city", [None, "Portland"])
def test_a_place_is_left_missing_when_its_city_is_not_one_the_carrier_serves_and_the_rest_are_kept(
    fake_populator, part, city
):
    fake_populator.given[part] = city
    request = extract().request
    assert getattr(request, part) is None and request.shipment == SHIPMENT


@pytest.mark.parametrize("shipment", [None, {"details": " ", "weight": 200}, {"details": "box of books"},
                                      {"details": "box of books", "weight": 0}, {"weight": 200}])
def test_a_shipment_without_details_or_a_weight_is_left_missing(fake_populator, shipment):
    fake_populator.given["shipment"] = shipment
    request = extract().request
    assert request.shipment is None and request.origin is not None


@pytest.mark.parametrize("error", [UnexpectedModelBehavior("no answer"), ValidationError.from_exception_data("x", [])])
def test_every_part_is_left_missing_when_the_language_model_fails(fake_populator, error):
    fake_populator.given = error
    state = extract()
    assert state.request == ShippingRequest() and state.extracted


def test_extract_parts_applies_once_per_utterance_and_not_when_nothing_is_missing():
    state = extract()
    assert not ExtractParts("op").can_perform(state, IO())
    complete = ShippingRequest(origin=SEATTLE, destination=DENVER, shipment=BOOKS)
    assert not ExtractParts("op").can_perform(taken(complete), IO())


# --- composing the asks


def test_compose_asks_waits_until_the_parts_have_been_extracted():
    state = taken(ShippingRequest(origin=SEATTLE))
    assert not ComposeAsks("op").can_perform(state, IO())
    state.extracted = True
    assert ComposeAsks("op").can_perform(state, IO())


def test_compose_asks_asks_for_each_part_still_missing_in_order():
    state = taken(ShippingRequest(destination=DENVER))
    state.extracted = True
    ComposeAsks("op").perform(state, IO())
    assert state.asks == [ASK_ORIGIN, AskFor(ShippingRequest, "shipment")]


def test_compose_asks_asks_for_nothing_once_the_request_is_complete_and_applies_once():
    state, op = taken(ShippingRequest(origin=SEATTLE, destination=DENVER, shipment=BOOKS)), ComposeAsks("op")
    assert op.can_perform(state, IO())
    op.perform(state, IO())
    assert state.asks == [] and not op.can_perform(state, IO())


# --- the sub-process


def test_the_sub_process_ranks_every_action_equally():
    sub_process = ShippingRequestFiller()
    [evaluator] = sub_process._action_evaluators
    assert {rank.rank for rank in evaluator(sub_process.state, None, ["take_request", "extract_parts"])} == {0}


def fill(request: ShippingRequest | None = None, answering: AskFor | None = None) -> ShippingRequestState | None:
    """runs a fresh sub-process over the phrase, as the conductor does"""
    return ShippingRequestFiller()(
        utterance=Utterance(PHRASE), answering=answering, reasoner=Reasoner(), request=request or ShippingRequest()
    )


def test_the_sub_process_fills_in_the_whole_request_and_asks_for_nothing():
    result = fill()
    assert result is not None and result.asks == []
    assert place(result.request.origin) == ("shipping-origin", "seattle")
    assert place(result.request.destination) == ("shipping-destination", "denver")
    assert result.request.shipment == SHIPMENT


def test_the_sub_process_asks_for_a_part_the_utterance_does_not_give(fake_populator):
    fake_populator.given["destination"] = "Portland"
    result = fill()
    assert result is not None and result.asks == [AskFor(ShippingRequest, "destination")]
    request = result.request
    assert (place(request.origin), request.destination, request.shipment) == (
        ("shipping-origin", "seattle"), None, SHIPMENT
    )


def test_the_sub_process_answers_a_question_with_one_call_of_the_language_model(fake_populator):
    fake_populator.given = {"origin": "Boston"}
    result = fill(answering=ASK_ORIGIN)
    assert result is not None and place(result.request.origin) == ("shipping-origin", "boston")
    assert result.asks == [AskFor(ShippingRequest, "destination"), AskFor(ShippingRequest, "shipment")]
    assert len(fake_populator.calls) == 1


def test_the_sub_process_leaves_the_request_it_is_given_as_it_was():
    request = ShippingRequest(origin=SEATTLE)
    fill(request)
    assert request == ShippingRequest(origin=SEATTLE)


def test_the_sub_process_with_nothing_missing_asks_for_nothing_without_calling_the_language_model(fake_populator):
    result = fill(ShippingRequest(origin=SEATTLE, destination=DENVER, shipment=BOOKS))
    assert result is not None and result.asks == [] and fake_populator.calls == []


def test_restarting_the_sub_process_fills_in_from_scratch():
    sub_process = ShippingRequestFiller()
    sub_process(utterance=Utterance(PHRASE), answering=None, reasoner=Reasoner(), request=ShippingRequest())
    sub_process.reinit()
    state = sub_process.state
    assert state.request is None and state.to_fill == [] and not state.extracted and state.asks is None
