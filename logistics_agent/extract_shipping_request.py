"""ExtractShippingRequest: the decision sub-process that turns what the human says about a delivery they want into a
shipping request

the conductor's ReactToShippingRequestIntent operator creates it once, then for every utterance the human makes about
a delivery, restarts it, links it to the conductor's channel for saying something to the human, io.o.interaction,
and calls it with two arguments, which its operators read from io.a:
    utterance: the utterance the human made
    reasoner: the reasoner, whose world graph holds the cities the carrier serves
it runs until the origin, destination, and shipment are all populated
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from cognition import DecisionProcess, IOContainer, ModelPopulator, Operator, uniform_evaluator
from pydantic import BaseModel, Field, ValidationError, create_model
from pydantic_ai.exceptions import UnexpectedModelBehavior

from logistics_agent.communicator import DEFAULT_LLM, Utterance
from logistics_agent.domain import City, Location, Shipment, ShippingRequest
from logistics_agent.reasoner import Reasoner


@dataclass
class ShippingRequestState:
    """state of the sub-process, kept for one utterance"""

    origin: Location | None = None
    destination: Location | None = None
    shipment: Shipment | None = None

    shipping_request: ShippingRequest = field(default_factory=ShippingRequest)

    # whether the origin, destination, and shipment have been extracted from the utterance, whether or not the
    # utterance gave them
    origin_extracted: bool = False
    destination_extracted: bool = False
    shipment_extracted: bool = False


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


def extract_city(phrase: str, task_desc: str, reasoner: Reasoner) -> City | None:
    """the city of the place the phrase describes, as the world graph knows it: the language model fills in a place
    whose city can only be one the world graph's cities, or None; None also when it cannot fill in the place at all"""
    cities = reasoner.find_entities(City)
    populator = ModelPopulator(served_place_model(cities), task_desc=task_desc)
    try:
        place = populator(phrase, DEFAULT_LLM)
    except (UnexpectedModelBehavior, ValidationError):
        return None
    if place.city is None:
        return None
    return next(city for city in cities if city.name == place.city.name)


class ExtractOrigin(Operator[ShippingRequestState]):
    """
    WHEN: the origin is None, and has not been extracted from the utterance yet
    THEN: fill in the city of the origin, the place the shipment is picked up from, from the utterance, choosing among
          the cities in the world graph, and name the origin shipping-origin; the origin stays None when the utterance
          does not give one of those cities
    """

    TASK_DESC = (
        "a human is describing a delivery they want from a logistics support agent; fill in the place the shipment is "
        "picked up from, the place that is the origin of the shipment. None if origin can't be determined."
    )
    # the name the origin is given
    NAME = "shipping-origin"

    def can_perform(self, state: ShippingRequestState, io: IOContainer) -> bool:
        """applies while the origin is None, once per utterance, so an utterance that does not give the origin is not
        populated again and again"""
        return state.origin is None and not state.origin_extracted

    def perform(self, state: ShippingRequestState, io: IOContainer) -> None:
        """extracts the origin's city from the utterance; keeps the origin, named shipping-origin, only when the city
        is one the world graph knows"""
        utterance: Utterance = io.a.utterance
        assert utterance.phrase is not None, "expect the conductor to supply what the human said"
        state.origin_extracted = True
        city = extract_city(utterance.phrase, self.TASK_DESC, io.a.reasoner)
        if city is not None:
            state.origin = Location(name=self.NAME, city=city)


class ExtractDestination(Operator[ShippingRequestState]):
    """
    WHEN: the destination is None, and has not been extracted from the utterance yet
    THEN: fill in the city of the destination, the place the shipment is dropped off at, from the utterance, choosing
          among the cities in the world graph, and name the destination shipping-destination; the destination stays
          None when the utterance does not give one of those cities
    """

    TASK_DESC = (
        "a human is describing a delivery they want from a logistics support agent; fill in the place that is the destination of the shipment, a place where the shipment is sent to. None if destination can't be determined."
    )
    # the name the destination is given
    NAME = "shipping-destination"

    def can_perform(self, state: ShippingRequestState, io: IOContainer) -> bool:
        """applies while the destination is None, once per utterance, so an utterance that does not give the
        destination is not populated again and again"""
        return state.destination is None and not state.destination_extracted

    def perform(self, state: ShippingRequestState, io: IOContainer) -> None:
        """extracts the destination's city from the utterance; keeps the destination, named shipping-destination, only
        when the city is one the world graph knows"""
        utterance: Utterance = io.a.utterance
        assert utterance.phrase is not None, "expect the conductor to supply what the human said"
        state.destination_extracted = True
        city = extract_city(utterance.phrase, self.TASK_DESC, io.a.reasoner)
        if city is not None:
            state.destination = Location(name=self.NAME, city=city)


class ExtractShipment(Operator[ShippingRequestState]):
    """
    WHEN: the shipment is None, and has not been extracted from the utterance yet
    THEN: fill in the shipment, what is shipped and how much it weighs, from the utterance with a ModelPopulator, and
          name it shipment; the shipment stays None when the utterance does not give both
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
        """applies while the shipment is None, once per utterance, so an utterance that does not give the shipment is
        not populated again and again"""
        return state.shipment is None and not state.shipment_extracted

    def perform(self, state: ShippingRequestState, io: IOContainer) -> None:
        """populates the shipment from the utterance; keeps it, named shipment, only when it has details and a
        weight"""
        utterance: Utterance = io.a.utterance
        assert utterance.phrase is not None, "expect the conductor to supply what the human said"
        state.shipment_extracted = True
        try:
            shipment = self._shipment_populator(utterance.phrase, DEFAULT_LLM)
        except (UnexpectedModelBehavior, ValidationError):
            return
        if shipment.details.strip() and shipment.weight > 0:
            state.shipment = shipment.model_copy(update={"name": self.NAME})


def all_populated(state: ShippingRequestState, _io: IOContainer) -> bool:
    """the sub-process is done once the origin, destination, and shipment are all populated"""
    return state.origin is not None and state.destination is not None and state.shipment is not None


class ExtractShippingRequest(DecisionProcess[ShippingRequestState]):
    """decision sub-process that extracts a shipping request from the human's utterance; see the module docstring for
    how it is called"""

    def __init__(self) -> None:
        """registers the termination check, and an evaluator that ranks every action equally, so one of several
        operators applying together is picked at random; starts over a fresh ShippingRequestState, with no cities served
        yet and a shipping request with every part None"""
        super().__init__(ShippingRequestState)
        self.add_operator(ExtractOrigin("extract_origin"))
        self.add_operator(ExtractDestination("extract_destination"))
        self.add_operator(ExtractShipment("extract_shipment"))
        self.add_termination_check(all_populated)
        self.add_action_evaluator(uniform_evaluator(0))


if __name__ == "__main__":
    import sys

    # the phrase to extract from: the command-line arguments, or a sample description of a delivery
    phrase = " ".join(sys.argv[1:]) or "the customer wants to ship a set of books, 100 pounds from Boston to Denver"

    # run the sub-process as the conductor does: restarted, linked to a channel to the human, and called with the
    # human's utterance
    sub_process = ExtractShippingRequest()
    sub_process.reinit()
    result = sub_process(utterance=Utterance(phrase=phrase), reasoner=Reasoner())

    # what the sub-process extracted, even when it did not finish
    state = sub_process.state
    print(f"phrase: {phrase}")
    print(f"finished: {result is not None}")
    print(f"origin: {state.origin}")
    print(f"destination: {state.destination}")
    print(f"shipment: {state.shipment}")
