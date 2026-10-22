"""ShippingRequestFiller: the decision sub-process that fills in a shipping request from what the human says about a
delivery they want, and asks for what the utterance does not give

the conductor's ReactToShippingRequestIntent operator creates it once, then for every utterance the human makes about
a delivery, restarts it and calls it with four arguments, which its operators read from io.a:
    utterance: the utterance the human made
    answering: the ask the utterance answers, or None; the language model reads the utterance with its question, so a
        short answer such as "Denver" is understood as the part asked for
    reasoner: the reasoner, whose world graph holds the cities the carrier serves
    request: the shipping request filled in so far; only its missing parts are extracted from the utterance
the missing parts are extracted together, in one call to the language model, so each thing the human mentions fills in
one part only: "Boston" in answer to where the shipment is picked up from fills in the origin, not the destination too,
while "Boston, going to Denver" fills in both; then the sub-process finishes with the request and the asks for the
parts that are still missing, in the order to ask them, which is empty when the request is complete
"""

from dataclasses import dataclass, field

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
    # the parts missing from the request taken, which are extracted from the utterance
    to_fill: list[str] = field(default_factory=list)
    # whether the parts to fill in have been extracted from the utterance, whether or not the utterance gave them
    extracted: bool = False
    # the parts still missing once they have been extracted, as asks in the order to ask them; None until then
    asks: list[AskFor] | None = None


class TakeRequest(Operator[ShippingRequestState]):
    """
    WHEN: the sub-process has not taken the shipping request yet
    THEN: take a copy of the shipping request filled in so far from the conductor, to fill in its missing parts
    """

    def can_perform(self, state: ShippingRequestState, io: IOContainer) -> bool:
        """applies once, at the start"""
        return state.request is None

    def perform(self, state: ShippingRequestState, io: IOContainer) -> None:
        """copies the request, so the conductor's stays as it was until it takes the result"""
        request: ShippingRequest = io.a.request
        state.request = request.model_copy()
        state.to_fill = request.missing()


class ShipmentParts(BaseModel):
    """what is shipped and how much it weighs, each None when the utterance does not give it"""
    details: str | None = Field(default=None, description=Shipment.model_fields["details"].description)
    weight: float | None = Field(default=None, description=Shipment.model_fields["weight"].description)


def parts_model(parts: list[str], cities: list[City]) -> type[BaseModel]:
    """one model of the parts of a shipping request to fill in, each None when the utterance does not give it, so the
    language model fills them in together and gives each thing the human mentions to one part

    a place is the name of its city, which can only be one the carrier serves; it is a plain name, checked against
    the cities afterwards, so a city the carrier does not serve leaves that part missing instead of failing the rest
    """
    served = ", ".join(city.display_name for city in cities)
    fields = {
        "origin": (str | None, Field(default=None, description=(
            f"{ShippingRequest.model_fields['origin'].description} The name of a city, one of {served}; None when "
            "the human does not say where the shipment is picked up from"
        ))),
        "destination": (str | None, Field(default=None, description=(
            f"{ShippingRequest.model_fields['destination'].description} The name of a city, one of {served}; None "
            "when the human does not say where the shipment is going"
        ))),
        "shipment": (ShipmentParts | None, Field(default=None, description=(
            f"{ShippingRequest.model_fields['shipment'].description} None when the human does not say what is shipped"
        ))),
    }
    return create_model(
        "ShippingRequestParts",
        __doc__="the parts of a delivery the human gives; each thing the human mentions fills in one part only",
        **{part: fields[part] for part in parts},
    )


def city_named(name: str | None, cities: list[City]) -> City | None:
    """the city with the name, as the world graph knows it, by its display name or its own name; None when the
    carrier does not serve it"""
    if name is None:
        return None
    wanted = name.strip().casefold()
    return next((city for city in cities if wanted in (city.display_name.casefold(), city.name.casefold())), None)


class ExtractParts(Operator[ShippingRequestState]):
    """
    WHEN: the request has been taken, and its missing parts have not been extracted from the utterance yet
    THEN: fill in the missing parts together from the utterance, read as the answer to the question asked, if any:
          the origin and the destination as places, named shipping-origin and shipping-destination, in cities the
          world graph knows, and the shipment, named shipment, with what is shipped and how much it weighs; a part
          stays missing when the utterance does not give it, or gives a city the carrier does not serve
    """

    TASK_DESC = (
        "a human is describing a delivery they want from a logistics support agent; fill in the parts of the "
        "delivery the human gives, and leave a part None when the human does not give it. Each thing the human "
        "mentions fills in one part only: a city that answers where the shipment is picked up from is the origin, "
        "not the destination too."
    )
    # the names the parts are given
    ORIGIN = "shipping-origin"
    DESTINATION = "shipping-destination"
    SHIPMENT = "shipment"

    def can_perform(self, state: ShippingRequestState, io: IOContainer) -> bool:
        """applies once per utterance, after the request has been taken, and only when parts are missing"""
        return state.request is not None and bool(state.to_fill) and not state.extracted

    def perform(self, state: ShippingRequestState, io: IOContainer) -> None:
        """populates the parts model in one call, then keeps each part it gives that is valid"""
        utterance: Utterance = io.a.utterance
        answering: AskFor | None = io.a.answering
        assert utterance.phrase is not None and state.request is not None, "expect the conductor to supply both"
        state.extracted = True
        cities = io.a.reasoner.find_entities(City)
        populator = ModelPopulator(parts_model(state.to_fill, cities), task_desc=self.TASK_DESC)
        asked = None if answering is None else answering.question
        try:
            parts = populator(utterance.phrase, DEFAULT_LLM, *asked_context(asked))
        except (UnexpectedModelBehavior, ValidationError):
            return
        request = state.request
        if "origin" in state.to_fill and (city := city_named(parts.origin, cities)) is not None:
            request.origin = Location(name=self.ORIGIN, city=city)
        if "destination" in state.to_fill and (city := city_named(parts.destination, cities)) is not None:
            request.destination = Location(name=self.DESTINATION, city=city)
        shipment = parts.shipment if "shipment" in state.to_fill else None
        if shipment is not None and shipment.details and shipment.details.strip() and (shipment.weight or 0) > 0:
            request.shipment = Shipment(name=self.SHIPMENT, details=shipment.details.strip(), weight=shipment.weight)


class ComposeAsks(Operator[ShippingRequestState]):
    """
    WHEN: the missing parts have been extracted from the utterance, or none were missing
    THEN: ask for each part that is still missing, in the order origin, destination, shipment; ask for nothing when
          the request is complete
    """

    def can_perform(self, state: ShippingRequestState, io: IOContainer) -> bool:
        """applies once nothing is left to extract, and only once"""
        return (
            state.request is not None
            and state.asks is None
            and (state.extracted or not state.to_fill)
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
        self.add_operator(ExtractParts("extract_parts"))
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
    result = sub_process(utterance=Utterance(phrase=phrase), answering=None, reasoner=Reasoner(),
                         request=ShippingRequest())

    print(f"phrase: {phrase}")
    print(f"finished: {result is not None}")
    if result is not None:
        print(f"request: {result.request}")
        print(f"asks: {[(ask.attribute, ask.question) for ask in result.asks or []]}")
