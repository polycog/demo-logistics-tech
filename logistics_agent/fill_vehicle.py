"""VehicleFiller: the decision sub-process that fills in a vehicle the human wants to add to the fleet, such as a truck
or an airplane, from what they say, and asks for what the utterance does not give

the conductor's ReactToAddToFleetIntent operator creates it once, then for every utterance the human makes about the
vehicle, restarts it and calls it with four arguments, which its operators read from io.a:
    utterance: the utterance the human made
    answering: the ask the utterance answers, or None; the language model reads the utterance with its question
    reasoner: the reasoner, which gives the new vehicle a name no other entity has
    draft: the draft of the vehicle filled in so far, or None before the kind of vehicle is known
the kind is chosen first, among the VehicleTypes, which starts a draft named by the agent; then the fields missing
from the draft are extracted together, in one call to the language model, so each thing the human mentions fills in
one field only; the sub-process finishes with the draft, or None when the kind is still unknown, and the asks for
what is still missing: the kind, or the draft's missing fields in order, which is empty when the vehicle is complete
"""

from dataclasses import dataclass, field
from typing import Any

from cognition import DecisionProcess, EnumClassifier, IOContainer, ModelPopulator, Operator, uniform_evaluator
from pydantic import BaseModel, Field, ValidationError, create_model
from pydantic_ai.exceptions import UnexpectedModelBehavior

from logistics_agent.communicator import DEFAULT_LLM, AskFor, AskForKind, Utterance, asked_context, in_context
from logistics_agent.domain import VehicleType
from logistics_agent.drafts import draft_model, drafted, missing
from logistics_agent.reasoner import Reasoner


@dataclass
class VehicleState:
    """state of the sub-process, kept for one utterance"""

    # whether the draft has been taken from the conductor
    taken: bool = False
    # the draft of the vehicle being filled in; None until the kind of vehicle is known
    draft: BaseModel | None = None
    # whether the kind of vehicle has been extracted from the utterance, whether or not the utterance gave it
    kind_tried: bool = False
    # the fields missing from the draft, which are extracted from the utterance
    to_fill: list[str] = field(default_factory=list)
    # whether the fields to fill in have been extracted from the utterance
    extracted: bool = False
    # what is still missing once the utterance has been used, as asks in the order to ask them; None until then
    asks: list[AskFor | AskForKind] | None = None


class TakeDraft(Operator[VehicleState]):
    """
    WHEN: the sub-process has not taken the draft yet
    THEN: take a copy of the draft of the vehicle filled in so far from the conductor, if there is one, to fill in
          its missing fields
    """

    def can_perform(self, state: VehicleState, io: IOContainer) -> bool:
        """applies once, at the start"""
        return not state.taken

    def perform(self, state: VehicleState, io: IOContainer) -> None:
        """copies the draft, so the conductor's stays as it was until it takes the result"""
        draft: BaseModel | None = io.a.draft
        state.taken = True
        if draft is not None:
            state.draft = draft.model_copy()
            state.to_fill = missing(draft)


def question(io: IOContainer) -> str | None:
    """the question of the ask the utterance answers, or None"""
    answering: AskFor | AskForKind | None = io.a.answering
    return None if answering is None else answering.question


class ChooseKind(Operator[VehicleState]):
    """
    WHEN: there is no draft yet, and the kind of vehicle has not been extracted from the utterance
    THEN: classify the kind of vehicle in the utterance among the VehicleTypes, read as the answer to the question
          asked, if any; start a draft of its model, named by the agent with a name no other entity has; there stays no
          draft when the utterance does not give a kind
    """

    TASK_DESC = (
        "a human is talking to a logistics support agent about adding a vehicle to the carrier's fleet; fill in the "
        "kind of vehicle the human wants to add. None if the kind can't be determined."
    )

    def __init__(self, name: str, **kwargs: Any) -> None:
        """sets up the kind classifier once, so its prompt is reused for every utterance"""
        super().__init__(name, **kwargs)
        self._kind_classifier = EnumClassifier(VehicleType, task_desc=self.TASK_DESC)

    def can_perform(self, state: VehicleState, io: IOContainer) -> bool:
        """applies once per utterance, after the draft has been taken, while there is none"""
        return state.taken and state.draft is None and not state.kind_tried

    def perform(self, state: VehicleState, io: IOContainer) -> None:
        """classifies the kind; starts a draft of its model, with every field missing but the name"""
        utterance: Utterance = io.a.utterance
        reasoner: Reasoner = io.a.reasoner
        assert utterance.phrase is not None, "expect the conductor to supply what the human said"
        state.kind_tried = True
        kind, _confidence = self._kind_classifier(in_context(question(io), utterance.phrase), DEFAULT_LLM, num_trials=1)
        if kind is None:
            return
        model = kind.model
        state.draft = draft_model(model)(name=reasoner.fresh_name(model.name_prefix))
        state.to_fill = missing(state.draft)


def fields_model(draft: BaseModel, fields: list[str]) -> type[BaseModel]:
    """one model of the fields of the vehicle to fill in, each optional and None when the utterance does not give it,
    so the language model fills them in together and gives each thing the human mentions to one field"""
    model = drafted(draft)
    return create_model(
        f"{model.__name__}Fields",
        __doc__=f"the parts of a {model.__name__.lower()} the human gives; each thing the human mentions fills in one "
                "field only",
        **{
            name: (model.model_fields[name].annotation | None, Field(
                default=None, description=f"{model.model_fields[name].description} None when the human does not say"
            ))
            for name in fields
        },
    )


def valid(value: object) -> bool:
    """whether a value the language model filled in can be kept: text that is not blank, or a number above zero"""
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, int | float):
        return value > 0
    return value is not None


class ExtractFields(Operator[VehicleState]):
    """
    WHEN: there is a draft with missing fields, and they have not been extracted from the utterance yet
    THEN: fill in the missing fields together from the utterance, read as the answer to the question asked, if any;
          a field stays missing when the utterance does not give it, or gives blank text or a number that is not
          above zero
    """

    TASK_DESC = (
        "a human is describing a vehicle they want to add to a logistics carrier's fleet; fill in the details of the "
        "vehicle the human gives, and leave a detail None when the human does not give it. Each thing the human "
        "mentions fills in one detail only."
    )

    def can_perform(self, state: VehicleState, io: IOContainer) -> bool:
        """applies once per utterance, while the draft has missing fields"""
        return state.draft is not None and bool(state.to_fill) and not state.extracted

    def perform(self, state: VehicleState, io: IOContainer) -> None:
        """populates the fields model in one call, then keeps each field it gives that is valid"""
        utterance: Utterance = io.a.utterance
        assert utterance.phrase is not None and state.draft is not None
        state.extracted = True
        populator = ModelPopulator(fields_model(state.draft, state.to_fill), task_desc=self.TASK_DESC)
        try:
            fields = populator(utterance.phrase, DEFAULT_LLM, *asked_context(question(io)))
        except (UnexpectedModelBehavior, ValidationError):
            return
        for name in state.to_fill:
            value = getattr(fields, name)
            if valid(value):
                setattr(state.draft, name, value.strip() if isinstance(value, str) else value)


class ComposeAsks(Operator[VehicleState]):
    """
    WHEN: the utterance has been used: the kind tried while there is no draft, or the draft's missing fields extracted
    THEN: ask for the kind when there is still no draft, and otherwise for each field still missing from the draft, in
          the model's order; ask for nothing when the vehicle is complete
    """

    def can_perform(self, state: VehicleState, io: IOContainer) -> bool:
        """applies once nothing is left to extract, and only once"""
        if not state.taken or state.asks is not None:
            return False
        if state.draft is None:
            return state.kind_tried
        return state.extracted or not state.to_fill

    def perform(self, state: VehicleState, io: IOContainer) -> None:
        """records the asks"""
        if state.draft is None:
            state.asks = [AskForKind(VehicleType)]
        else:
            state.asks = [AskFor(drafted(state.draft), name) for name in missing(state.draft)]


def asks_composed(state: VehicleState, _io: IOContainer) -> bool:
    """the sub-process is done once it has composed the asks, which are empty when the vehicle is complete"""
    return state.asks is not None


class VehicleFiller(DecisionProcess[VehicleState]):
    """decision sub-process that fills in a vehicle to add to the fleet from the human's utterance and asks for what is
    missing; see the module docstring for how it is called"""

    def __init__(self) -> None:
        """registers the termination check, and an evaluator that ranks every action equally; starts over a fresh
        VehicleState"""
        super().__init__(VehicleState)
        self.add_operator(TakeDraft("take_draft"))
        self.add_operator(ChooseKind("choose_kind"))
        self.add_operator(ExtractFields("extract_fields"))
        self.add_operator(ComposeAsks("compose_asks"))
        self.add_termination_check(asks_composed)
        self.add_action_evaluator(uniform_evaluator(0))
