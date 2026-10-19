"""ExtractAndSetCurrent: the decision sub-process that turns what the human says about where packages are now into
the planner's current facts

a parent operator creates it once, then for every utterance restarts it and calls it with three arguments, which its
operators read from io.a:
    location: the human's utterance saying where packages are, such as where one is shipped from
    reasoner: the reasoner, whose world graph names the things the human can refer to
    planner: the planner whose current facts are set
it runs until it has worded a reply, and returns its final CurrentState
"""

from dataclasses import dataclass
from typing import Any

from cognition import DecisionProcess, IOContainer, ModelPopulator, Operator
from pydantic import ValidationError

from logistics_agent.communicator import DEFAULT_LLM, CurrentLocation, Utterance, known_entities
from logistics_agent.domain import At

ASK_WHERE = "Sorry, I did not catch where the package is now. Where is it?"


@dataclass
class CurrentState:
    """state of the sub-process, kept for one utterance

    current_location: where packages are, as the language model filled it in, naming things by name
    current: the location resolved into At facts; empty when nothing in it could be resolved
    reply: what to tell the human; the sub-process is done once it is set
    """

    current_location: CurrentLocation | None = None
    current: list[At] | None = None
    reply: str | None = None


class ExtractCurrentLocation(Operator[CurrentState]):
    """
    WHEN: the location has not been extracted yet
    THEN: fill in a CurrentLocation from the human's utterance, naming the things the reasoner knows about
    """

    TASK_DESC = (
        "a human is telling a logistics support agent where their packages are right now, before delivery, such as "
        "where a package is shipped from; the agent has one office and one airport in each city it serves, and a "
        "package in a city is at that city's office unless the human names the airport or a vehicle"
    )

    def __init__(self, name: str, **kwargs: Any) -> None:
        """sets up the location populator once, so its prompt is reused for every utterance"""
        super().__init__(name, **kwargs)
        self._location_populator = ModelPopulator(CurrentLocation, task_desc=self.TASK_DESC)

    def can_perform(self, state: CurrentState, io: IOContainer) -> bool:
        """applies once, at the start of the sub-process"""
        return state.current_location is None

    def perform(self, state: CurrentState, io: IOContainer) -> None:
        """populates the current location from the utterance the parent supplied"""
        location: Utterance = io.a.location
        assert location.phrase is not None, "expect the parent operator to supply what the human said"
        state.current_location = self._location_populator(
            location.phrase, DEFAULT_LLM, *known_entities(io.a.reasoner.world)
        )


class ResolveCurrent(Operator[CurrentState]):
    """
    WHEN: the location has been extracted but not resolved into facts
    THEN: look up each named thing with the reasoner and build the At facts; no facts if any name is unknown or a
          fact cannot hold
    """

    def can_perform(self, state: CurrentState, io: IOContainer) -> bool:
        """applies once the current location is known"""
        return state.current_location is not None and state.current is None

    def perform(self, state: CurrentState, io: IOContainer) -> None:
        """resolves the current location into At facts, or into no facts when it cannot"""
        assert state.current_location is not None
        try:
            state.current = state.current_location.to_relations(io.a.reasoner.find_entity)
        except (KeyError, ValidationError):
            state.current = []


class SetCurrent(Operator[CurrentState]):
    """
    WHEN: the location resolved into facts
    THEN: set them as the planner's current facts, and confirm them back to the human
    """

    def can_perform(self, state: CurrentState, io: IOContainer) -> bool:
        """applies when there are facts and no reply yet"""
        return bool(state.current) and state.reply is None

    def perform(self, state: CurrentState, io: IOContainer) -> None:
        """sets the planner's current facts and words the confirmation"""
        assert state.current
        io.a.planner.set_current(state.current)
        places = "; ".join(f"the {fact.entity1.display_name} is at the {fact.entity2.display_name}" for fact in state.current)
        state.reply = f"Got it: {places}. Ask me for a price estimate whenever you are ready."


class AskWhere(Operator[CurrentState]):
    """
    WHEN: the location resolved into no facts
    THEN: leave the planner's current facts as they are, and ask the human where the package is
    """

    def can_perform(self, state: CurrentState, io: IOContainer) -> bool:
        """applies when resolving produced no facts and there is no reply yet"""
        return state.current == [] and state.reply is None

    def perform(self, state: CurrentState, io: IOContainer) -> None:
        """words the request to say where the package is"""
        state.reply = ASK_WHERE


def has_reply(state: CurrentState, _io: IOContainer) -> bool:
    """the sub-process is done once it has worded a reply"""
    return state.reply is not None


class ExtractAndSetCurrent(DecisionProcess[CurrentState]):
    """decision sub-process that extracts where packages are now from the human's utterance, sets it as the planner's
    current facts, and words the reply; see the module docstring for the arguments it is called with"""

    def __init__(self) -> None:
        """registers the operators and the termination check over a fresh CurrentState"""
        super().__init__(CurrentState)
        self.add_operator(ExtractCurrentLocation("extract_current_location"))
        self.add_operator(ResolveCurrent("resolve_current"))
        self.add_operator(SetCurrent("set_current"))
        self.add_operator(AskWhere("ask_where"))
        self.add_termination_check(has_reply)
