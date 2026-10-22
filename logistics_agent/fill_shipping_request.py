"""ShippingRequestFiller: the decision sub-process that fills in a shipping request from what the human says about a
delivery they want, and asks for what the utterance does not give

the conductor's ReactToShippingRequestIntent operator creates it once, then for every utterance the human makes about
a delivery, restarts it and calls it with four arguments, which its operators read from io.a:
    utterance: the utterance the human made
    asked: the question the utterance answers, or None; the language model reads the utterance with it, so a short
        answer such as "Denver" is understood
    reasoner: the reasoner, whose world graph holds the cities the carrier serves
    request: the shipping request filled in so far; only its missing parts are extracted from the utterance
every missing part is extracted once; then the sub-process finishes with the request and the asks for the parts that
are still missing, in the order to ask them, which is empty when the request is complete
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from cognition import DecisionProcess, IOContainer, ModelPopulator, Operator, uniform_evaluator
from pydantic import BaseModel, Field, ValidationError, create_model
from pydantic_ai.exceptions import UnexpectedModelBehavior

from logistics_agent.communicator import DEFAULT_LLM, AskFor, Utterance, asked_context
from logistics_agent.domain import City, Location, Shipment, ShippingRequest
from logistics_agent.reasoner import Reasoner


@dataclass
class ShippingRequestState:
    """state of the sub-process, kept for one utterance"""

    # the shipping request being filled in, taken from the conductor; None until it has been taken
    request: ShippingRequest | None = None
    # the parts of the request extracted from the utterance, whether or not the utterance gave them
    tried: set[str] = field(default_factory=set)
    # the parts still missing once every missing part has been tried, as asks in the order to ask them; None until then
    asks: list[AskFor] | None = None


class TakeRequest(Operator[ShippingRequestState]):
    """
    WHEN: the sub-process has not taken the shipping request yet
    THEN: take a copy of the shipping request filled in so far from the conductor
    """

    def can_perform(self, state: ShippingRequestState, io: IOContainer) -> bool:
        """applies once, at the start"""
        return state.request is None

    def perform(self, state: ShippingRequestState, io: IOContainer) -> None:
        """copies the request, so the conductor's stays as it was until it takes the result"""
        request: ShippingRequest = io.a.request
        state.request = request.model_copy()


def to_extract(state: ShippingRequestState, part: str) -> bool:
    """whether the part is missing from the request taken, and has not been extracted from the utterance yet"""
    return state.request is not None and getattr(state.request, part) is None and part not in state.tried


def served_place_model(cities: list[City]) -> type[BaseModel]:
    """a model of a place whose city can only be one of the cities the carrier serves, or None; the language model
    filling it in must pick one of them, by its display name, or leave the city empty"""
    served_city = Enum("ServedCity", {city.name: city.display_name for city in cities})
    return create_model(
        "Place",
        __doc__="a place in one of the cities the carrier serves",
        city=(served_city | None, Field(
            default=None, description="city the place is in; None when it is not one of the cities the carrier serves"
        )),
    )


def extract_city(phrase: str, task_desc: str, reasoner: Reasoner, asked: str | None = None) -> City | None:
    """the city of the place the phrase describes, as the world graph knows it: the language model fills in a place
    whose city can only be one the world graph's cities, or None, reading the phrase as the answer to the question
    asked, if any; None also when it cannot fill in the place at all"""
    cities = reasoner.find_entities(City)
    populator = ModelPopulator(served_place_model(cities), task_desc=task_desc)
    try:
        place = populator(phrase, DEFAULT_LLM, *asked_context(asked))
    except (UnexpectedModelBehavior, ValidationError):
        return None
    if place.city is None:
        return None
    return next(city for city in cities if city.name == place.city.name)


class ExtractOrigin(Operator[ShippingRequestState]):
    """
    WHEN: the request's origin is missing, and has not been extracted from the utterance yet
    THEN: fill in the city of the origin, the place the shipment is picked up from, from the utterance, choosing among
          the cities in the world graph, and name the origin shipping-origin; the origin stays missing when the
          utterance does not give one of those cities
    """

    TASK_DESC = (
        "a human is describing a delivery they want from a logistics support agent; fill in the place the shipment is "
        "picked up from, the place that is the origin of the shipment. None if origin can't be determined."
    )
    # the name the origin is given
    NAME = "shipping-origin"

    def can_perform(self, state: ShippingRequestState, io: IOContainer) -> bool:
        """applies while the origin is missing, once per utterance, so an utterance that does not give the origin is
        not populated again and again"""
        return to_extract(state, "origin")

    def perform(self, state: ShippingRequestState, io: IOContainer) -> None:
        """extracts the origin's city from the utterance; keeps the origin, named shipping-origin, only when the city
        is one the world graph knows"""
        utterance: Utterance = io.a.utterance
        assert utterance.phrase is not None and state.request is not None, "expect the conductor to supply both"
        state.tried.add("origin")
        city = extract_city(utterance.phrase, self.TASK_DESC, io.a.reasoner, io.a.asked)
        if city is not None:
            state.request.origin = Location(name=self.NAME, city=city)


class ExtractDestination(Operator[ShippingRequestState]):
    """
    WHEN: the request's destination is missing, and has not been extracted from the utterance yet
    THEN: fill in the city of the destination, the place the shipment is dropped off at, from the utterance, choosing
          among the cities in the world graph, and name the destination shipping-destination; the destination stays
          missing when the utterance does not give one of those cities
    """

    TASK_DESC = (
        "a human is describing a delivery they want from a logistics support agent; fill in the place that is the destination of the shipment, a place where the shipment is sent to. None if destination can't be determined."
    )
    # the name the destination is given
    NAME = "shipping-destination"

    def can_perform(self, state: ShippingRequestState, io: IOContainer) -> bool:
        """applies while the destination is missing, once per utterance, so an utterance that does not give the
        destination is not populated again and again"""
        return to_extract(state, "destination")

    def perform(self, state: ShippingRequestState, io: IOContainer) -> None:
        """extracts the destination's city from the utterance; keeps the destination, named shipping-destination, only
        when the city is one the world graph knows"""
        utterance: Utterance = io.a.utterance
        assert utterance.phrase is not None and state.request is not None, "expect the conductor to supply both"
        state.tried.add("destination")
        city = extract_city(utterance.phrase, self.TASK_DESC, io.a.reasoner, io.a.asked)
        if city is not None:
            state.request.destination = Location(name=self.NAME, city=city)


class ExtractShipment(Operator[ShippingRequestState]):
    """
    WHEN: the request's shipment is missing, and has not been extracted from the utterance yet
    THEN: fill in the shipment, what is shipped and how much it weighs, from the utterance with a ModelPopulator, and
          name it shipment; the shipment stays missing when the utterance does not give both
    """

    TASK_DESC = (
        "a human is describing a delivery they want from a logistics support agent; fill in what is shipped and how "
        "much it weighs, in pounds. None if the shipment details can't be determined."
    )
    # the name the shipment is given, in place of whatever name the language model fills in
    NAME = "shipment"

    def __init__(self, name: str, **kwargs: Any) -> None:
        """sets up the shipment populator once, so its prompt is reused for every utterance"""
        super().__init__(name, **kwargs)
        self._shipment_populator = ModelPopulator(Shipment, task_desc=self.TASK_DESC)

    def can_perform(self, state: ShippingRequestState, io: IOContainer) -> bool:
        """applies while the shipment is missing, once per utterance, so an utterance that does not give the shipment
        is not populated again and again"""
        return to_extract(state, "shipment")

    def perform(self, state: ShippingRequestState, io: IOContainer) -> None:
        """populates the shipment from the utterance; keeps it, named shipment, only when it has details and a
        weight"""
        utterance: Utterance = io.a.utterance
        assert utterance.phrase is not None and state.request is not None, "expect the conductor to supply both"
        state.tried.add("shipment")
        try:
            shipment = self._shipment_populator(utterance.phrase, DEFAULT_LLM, *asked_context(io.a.asked))
        except (UnexpectedModelBehavior, ValidationError):
            return
        if shipment.details.strip() and shipment.weight > 0:
            state.request.shipment = shipment.model_copy(update={"name": self.NAME})


class ComposeAsks(Operator[ShippingRequestState]):
    """
    WHEN: every part missing from the request has been extracted from the utterance
    THEN: ask for each part that is still missing, in the order origin, destination, shipment; ask for nothing when
          the request is complete
    """

    def can_perform(self, state: ShippingRequestState, io: IOContainer) -> bool:
        """applies once nothing is left to extract, and only once"""
        return (
            state.request is not None
            and state.asks is None
            and all(part in state.tried for part in state.request.missing())
        )

    def perform(self, state: ShippingRequestState, io: IOContainer) -> None:
        """records an ask for every missing part"""
        assert state.request is not None
        state.asks = [AskFor(ShippingRequest, part) for part in state.request.missing()]


def asks_composed(state: ShippingRequestState, _io: IOContainer) -> bool:
    """the sub-process is done once it has composed the asks, which are empty when the request is complete"""
    return state.asks is not None


class ShippingRequestFiller(DecisionProcess[ShippingRequestState]):
    """decision sub-process that fills in a shipping request from the human's utterance and asks for what is missing;
    see the module docstring for how it is called"""

    def __init__(self) -> None:
        """registers the termination check, and an evaluator that ranks every action equally, so one of several
        operators applying together is picked at random; starts over a fresh ShippingRequestState"""
        super().__init__(ShippingRequestState)
        self.add_operator(TakeRequest("take_request"))
        self.add_operator(ExtractOrigin("extract_origin"))
        self.add_operator(ExtractDestination("extract_destination"))
        self.add_operator(ExtractShipment("extract_shipment"))
        self.add_operator(ComposeAsks("compose_asks"))
        self.add_termination_check(asks_composed)
        self.add_action_evaluator(uniform_evaluator(0))


if __name__ == "__main__":
    import sys

    # the phrase to fill the request from: the command-line arguments, or a sample description of a delivery
    phrase = " ".join(sys.argv[1:]) or "the customer wants to ship a set of books, 100 pounds from Boston"

    # run the sub-process as the conductor does: restarted, and called with the human's utterance and an empty request
    sub_process = ShippingRequestFiller()
    sub_process.reinit()
    result = sub_process(utterance=Utterance(phrase=phrase), asked=None, reasoner=Reasoner(), request=ShippingRequest())

    print(f"phrase: {phrase}")
    print(f"finished: {result is not None}")
    if result is not None:
        print(f"request: {result.request}")
        print(f"asks: {[(ask.attribute, ask.question) for ask in result.asks or []]}")
