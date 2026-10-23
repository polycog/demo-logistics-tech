"""StateInspector: the decision sub-process that answers the human's questions about the state of the world, such as
which vehicles the carrier's fleet has, from the world graph

the language model only understands the question: it classifies which kind of question it is, then fills in what the
question is about; the facts of the answer come from the reasoner, never from the language model

the conductor's ReactToInspectStateIntent operator creates it once, then for every question the human asks about the
state, restarts it and calls it with three arguments, which its operators read from io.a:
    utterance: the utterance the human made
    answering: the ask the utterance answers, or None; the utterance is read together with its question
    reasoner: the reasoner, whose world graph holds the state
the sub-process finishes with the kind of question, None when it is not one the agent can answer, what it is about,
and the facts that answer it
"""

from dataclasses import dataclass
from typing import Any

from cognition import DecisionProcess, DocEnum, EnumClassifier, IOContainer, ModelPopulator, Operator, uniform_evaluator
from pydantic import BaseModel, Field, ValidationError
from pydantic_ai.exceptions import UnexpectedModelBehavior

from logistics_agent.communicator import DEFAULT_LLM, AskFor, AskForKind, Utterance, asked_context, in_context
from logistics_agent.domain import At, Location, Vehicle, VehicleType
from logistics_agent.reasoner import Reasoner


class QuestionType(DocEnum):
    """the kinds of question about the state of the world the agent can answer"""

    FLEET = ("fleet", "which vehicles the carrier has, such as its trucks or its airplanes, or its whole fleet")


class FleetQuestion(BaseModel):
    """which part of the fleet the human asks about"""
    kind: VehicleType | None = Field(
        default=None, description="the kind of vehicle the human asks about; None when they ask about the whole fleet"
    )


@dataclass(frozen=True)
class Sighting:
    """a fact of the answer: a vehicle, and where it is; None when the world graph does not say where it is"""

    vehicle: Vehicle
    place: Location | None


@dataclass
class InspectionState:
    """state of the sub-process, kept for one question"""

    # whether the question has been classified, and the kind of question; None when it is not one the agent answers
    classified: bool = False
    question_type: QuestionType | None = None
    # what a question about the fleet is about; None until it has been filled in
    fleet_question: FleetQuestion | None = None
    # the facts that answer the question; None until they have been looked up
    sightings: list[Sighting] | None = None


def question(io: IOContainer) -> str | None:
    """the question of the ask the utterance answers, or None"""
    answering: AskFor | AskForKind | None = io.a.answering
    return None if answering is None else answering.question


class ClassifyQuestion(Operator[InspectionState]):
    """
    WHEN: the question has not been classified
    THEN: classify which kind of question about the state of the world the utterance asks, among the QuestionTypes,
          read as the answer to the question the agent asked, if any; None when it is none of them
    """

    TASK_DESC = (
        "a human asks a logistics support agent about the state of the carrier's world; which kind of question is it"
    )

    def __init__(self, name: str, **kwargs: Any) -> None:
        """sets up the question classifier once, so its prompt is reused for every question"""
        super().__init__(name, **kwargs)
        self._question_classifier = EnumClassifier(QuestionType, task_desc=self.TASK_DESC)

    def can_perform(self, state: InspectionState, io: IOContainer) -> bool:
        """applies once, at the start"""
        return not state.classified

    def perform(self, state: InspectionState, io: IOContainer) -> None:
        """classifies the question"""
        utterance: Utterance = io.a.utterance
        assert utterance.phrase is not None, "expect the conductor to supply what the human said"
        state.classified = True
        state.question_type, _confidence = self._question_classifier(
            in_context(question(io), utterance.phrase), DEFAULT_LLM, num_trials=1
        )


class FillFleetQuestion(Operator[InspectionState]):
    """
    WHEN: the question is about the fleet, and what it is about has not been filled in
    THEN: fill in which kind of vehicle it asks about, or None for the whole fleet; the whole fleet also when the
          language model cannot fill it in
    """

    TASK_DESC = (
        "a human asks a logistics support agent about the carrier's fleet; fill in which kind of vehicle they ask "
        "about, or None when they ask about the whole fleet"
    )

    def __init__(self, name: str, **kwargs: Any) -> None:
        """sets up the populator once, so its prompt is reused for every question"""
        super().__init__(name, **kwargs)
        self._populator = ModelPopulator(FleetQuestion, task_desc=self.TASK_DESC)

    def can_perform(self, state: InspectionState, io: IOContainer) -> bool:
        """applies once a question about the fleet has been classified"""
        return state.question_type is QuestionType.FLEET and state.fleet_question is None

    def perform(self, state: InspectionState, io: IOContainer) -> None:
        """fills in the kind of vehicle asked about"""
        utterance: Utterance = io.a.utterance
        assert utterance.phrase is not None
        try:
            state.fleet_question = self._populator(utterance.phrase, DEFAULT_LLM, *asked_context(question(io)))
        except (UnexpectedModelBehavior, ValidationError):
            state.fleet_question = FleetQuestion()


def place_of(vehicle: Vehicle, reasoner: Reasoner) -> Location | None:
    """where the vehicle is, as the world graph has it, or None when it does not say"""
    found = reasoner.find_relations(At, entity1=vehicle.name)
    return next((at.entity2 for at in found if isinstance(at.entity2, Location)), None)


class LookUpFleet(Operator[InspectionState]):
    """
    WHEN: what a question about the fleet is about has been filled in, and the facts have not been looked up
    THEN: look up in the world graph the vehicles of the kind asked about, or every vehicle, and where each is, in
          the order of their names
    """

    def can_perform(self, state: InspectionState, io: IOContainer) -> bool:
        """applies once what the question is about is known"""
        return state.fleet_question is not None and state.sightings is None

    def perform(self, state: InspectionState, io: IOContainer) -> None:
        """records each vehicle asked about, and where it is"""
        assert state.fleet_question is not None
        reasoner: Reasoner = io.a.reasoner
        kind = state.fleet_question.kind
        vehicles = sorted(reasoner.find_entities(Vehicle if kind is None else kind.model), key=lambda v: v.name)
        state.sightings = [Sighting(vehicle, place_of(vehicle, reasoner)) for vehicle in vehicles]


def answered(state: InspectionState, _io: IOContainer) -> bool:
    """the sub-process is done once the question is classified as none the agent answers, or its facts are looked up"""
    return (state.classified and state.question_type is None) or state.sightings is not None


class StateInspector(DecisionProcess[InspectionState]):
    """decision sub-process that answers the human's questions about the state of the world from the world graph; see
    the module docstring for how it is called"""

    def __init__(self) -> None:
        """registers the termination check, and an evaluator that ranks every action equally; starts over a fresh
        InspectionState"""
        super().__init__(InspectionState)
        self.add_operator(ClassifyQuestion("classify_question"))
        self.add_operator(FillFleetQuestion("fill_fleet_question"))
        self.add_operator(LookUpFleet("look_up_fleet"))
        self.add_termination_check(answered)
        self.add_action_evaluator(uniform_evaluator(0))
