"""PlacementFiller: the decision sub-process that fills in where a vehicle being added to the fleet is, an At relation
from the vehicle to a location in the world graph, from what the human says, and asks for it when the utterance does
not give it

the conductor fills in the vehicle first, with the VehicleFiller sub-process, then starts a draft of its At relation
with the vehicle as entity1; its ReactToAddToFleetIntent operator creates this sub-process once, then for every
utterance the human makes about where the vehicle is, restarts it and calls it with four arguments, which its
operators read from io.a:
    utterance: the utterance the human made
    answering: the ask the utterance answers, or None; the language model reads the utterance with its question
    reasoner: the reasoner, whose world graph holds the locations
    draft: the draft of the At relation, with the vehicle as entity1
the location is one the world graph already has: an airport for an airplane, and a terminal or an airport for any
other vehicle; the sub-process finishes with the draft and the asks for what is still missing, which is empty once the
location is known
"""

from dataclasses import dataclass, field

from cognition import DecisionProcess, IOContainer, ModelPopulator, Operator, uniform_evaluator
from pydantic import BaseModel, Field, ValidationError, create_model
from pydantic_ai.exceptions import UnexpectedModelBehavior

from logistics_agent.communicator import DEFAULT_LLM, AskFor, Utterance, asked_context, label
from logistics_agent.domain import Airplane, Airport, At, Location
from logistics_agent.drafts import missing
from logistics_agent.reasoner import Reasoner


@dataclass
class PlacementState:
    """state of the sub-process, kept for one utterance"""

    # the draft of the At relation being filled in, taken from the conductor; None until it has been taken
    draft: BaseModel | None = None
    # the fields missing from the draft, which are extracted from the utterance
    to_fill: list[str] = field(default_factory=list)
    # whether the fields to fill in have been extracted from the utterance
    extracted: bool = False
    # what is still missing once the utterance has been used, as asks; None until then
    asks: list[AskFor] | None = None


class TakeDraft(Operator[PlacementState]):
    """
    WHEN: the sub-process has not taken the draft yet
    THEN: take a copy of the draft of the At relation from the conductor, to fill in its missing fields
    """

    def can_perform(self, state: PlacementState, io: IOContainer) -> bool:
        """applies once, at the start"""
        return state.draft is None

    def perform(self, state: PlacementState, io: IOContainer) -> None:
        """copies the draft, so the conductor's stays as it was until it takes the result"""
        draft: BaseModel = io.a.draft
        state.draft = draft.model_copy()
        state.to_fill = missing(draft)


def places_for(vehicle: object, reasoner: Reasoner) -> list[Location]:
    """the locations the vehicle can be at: the airports for an airplane, and every terminal and airport otherwise"""
    return reasoner.find_entities(Airport if isinstance(vehicle, Airplane) else Location)


def location_named(name: str | None, places: list[Location]) -> Location | None:
    """the location with the name, by its display name or its own name; None when it is not one of the places"""
    if name is None:
        return None
    wanted = name.strip().casefold()
    return next((place for place in places if wanted in (label(place).casefold(), place.name.casefold())), None)


def location_model(places: list[Location]) -> type[BaseModel]:
    """a model of where the vehicle is: the name of one of the places, checked against them afterwards, so the
    language model can only pick a location the world graph has"""
    names = "; ".join(label(place) for place in places)
    return create_model(
        "Whereabouts",
        __doc__="where the vehicle is now",
        entity2=(str | None, Field(default=None, description=(
            f"{At.model_fields['entity2'].description} The name of one of: {names}. None when the human does not say"
        ))),
    )


class ExtractLocation(Operator[PlacementState]):
    """
    WHEN: the draft's location is missing, and has not been extracted from the utterance yet
    THEN: fill in the location from the utterance, read as the answer to the question asked, if any, choosing among the
          locations the vehicle can be at; it stays missing when the utterance does not give one of them
    """

    TASK_DESC = (
        "a human is adding a vehicle to a logistics carrier's fleet and says where the vehicle is now; fill in the "
        "terminal or airport it is at. None if it can't be determined."
    )

    def can_perform(self, state: PlacementState, io: IOContainer) -> bool:
        """applies once per utterance, while the location is missing"""
        return state.draft is not None and "entity2" in state.to_fill and not state.extracted

    def perform(self, state: PlacementState, io: IOContainer) -> None:
        """populates the location model, and keeps the location it names when the vehicle can be there"""
        utterance: Utterance = io.a.utterance
        answering: AskFor | None = io.a.answering
        assert utterance.phrase is not None and state.draft is not None
        state.extracted = True
        places = places_for(state.draft.entity1, io.a.reasoner)
        populator = ModelPopulator(location_model(places), task_desc=self.TASK_DESC)
        asked = None if answering is None else answering.question
        try:
            whereabouts = populator(utterance.phrase, DEFAULT_LLM, *asked_context(asked))
        except (UnexpectedModelBehavior, ValidationError):
            return
        state.draft.entity2 = location_named(whereabouts.entity2, places)


class ComposeAsks(Operator[PlacementState]):
    """
    WHEN: the draft's missing fields have been extracted from the utterance, or none were missing
    THEN: ask for each field still missing; ask for nothing once the location is known
    """

    def can_perform(self, state: PlacementState, io: IOContainer) -> bool:
        """applies once nothing is left to extract, and only once"""
        return state.draft is not None and state.asks is None and (state.extracted or not state.to_fill)

    def perform(self, state: PlacementState, io: IOContainer) -> None:
        """records an ask for every missing field"""
        assert state.draft is not None
        state.asks = [AskFor(At, name) for name in missing(state.draft)]


def asks_composed(state: PlacementState, _io: IOContainer) -> bool:
    """the sub-process is done once it has composed the asks, which are empty once the location is known"""
    return state.asks is not None


class PlacementFiller(DecisionProcess[PlacementState]):
    """decision sub-process that fills in where a vehicle being added to the fleet is, and asks for it when the
    utterance does not give it; see the module docstring for how it is called"""

    def __init__(self) -> None:
        """registers the termination check, and an evaluator that ranks every action equally; starts over a fresh
        PlacementState"""
        super().__init__(PlacementState)
        self.add_operator(TakeDraft("take_draft"))
        self.add_operator(ExtractLocation("extract_location"))
        self.add_operator(ComposeAsks("compose_asks"))
        self.add_termination_check(asks_composed)
        self.add_action_evaluator(uniform_evaluator(0))
