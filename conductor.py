"""decision process of the pricing support cogent: the conversation state it keeps, and the operators that
decide what to do with each utterance"""

import re
from dataclasses import dataclass, field
from typing import Any

from cognition import EnumClassifier, IOContainer, ModelPopulator, Operator, describe_facts

from cognition import Entity
from pydantic import ValidationError
from pydantic_ai.exceptions import UnexpectedModelBehavior

from communicator import (
    DEFAULT_LLM,
    DesiredOutcome,
    EntityChange,
    EntityKind,
    EntityQuery,
    HumanIntent,
    NewEntity,
    StateQuery,
    Utterance,
    known_entities,
)
from domain import At
from planner import LogisticsPlanner
from reasoner import Reasoner

QUIT_PHRASES = {"quit", "exit", "bye", "goodbye"}
QUIT_PATTERN = re.compile(rf"\b({'|'.join(QUIT_PHRASES)})\b", re.IGNORECASE)
INTRODUCTION = "Hi - I am your pricing support agent. I can help you estimate prices for various deliveries."


@dataclass
class ConversationState:
    """state of the conversation, kept across decision cycles"""

    introduced: bool = False
    done: bool = False
    human_utterance: Utterance | None = None
    human_intent: HumanIntent | None = None

    reasoner: Reasoner = field(default_factory=Reasoner)
    planner: LogisticsPlanner = field(init=False)

    def __post_init__(self) -> None:
        """the planner plans over the world graph the reasoner maintains"""
        self.planner = LogisticsPlanner(self.reasoner.world)


class Introduce(Operator[ConversationState]):
    """
    WHEN: the cogent has not introduced itself yet
    THEN: introduce the cogent and what it can help with
    """

    def can_perform(self, state: ConversationState, io: IOContainer) -> bool:
        """applies once, at the start of the conversation"""
        return not state.introduced

    def perform(self, state: ConversationState, io: IOContainer) -> None:
        """says the introduction and remembers that it has been said"""
        io.o.interaction(Utterance(phrase=INTRODUCTION))
        state.introduced = True

class SayGoodbye(Operator[ConversationState]):
    """
    WHEN: the human's utterance contains a quit phrase
    THEN: say goodbye and end the conversation
    """

    def can_perform(self, state: ConversationState, io: IOContainer) -> bool:
        """applies when the utterance contains one of the quit phrases as a word, e.g. "ok bye now" or "Exit!" """
        utterance: Utterance = io.i.interaction
        return utterance.phrase is not None and QUIT_PATTERN.search(utterance.phrase) is not None

    def perform(self, state: ConversationState, io: IOContainer) -> None:
        """replies with goodbye and marks the conversation as done"""
        io.o.interaction(Utterance(phrase="Goodbye!"))
        state.done = True


class RecognizeIntent(Operator[ConversationState]):
    """
    WHEN: the human said something other than a quit phrase, and it has not been interpreted yet
    THEN: recognize the intent expressed in it, using an EnumClassifier over the known HumanIntents
    """

    TASK_DESC = (
        "a human is talking to a logistics support agent; "
        "what is the human asking the agent to do"
    )

    def __init__(self, name: str, **kwargs: Any) -> None:
        """sets up the intent classifier once, so its prompt is reused for every utterance"""
        super().__init__(name, **kwargs)
        self._intent_classifier = EnumClassifier(HumanIntent, task_desc=self.TASK_DESC)

    def can_perform(self, state: ConversationState, io: IOContainer) -> bool:
        """applies to a new non-empty utterance that is not a quit phrase"""
        utterance: Utterance = io.i.interaction
        return (
            state.human_utterance is None
            and utterance.phrase is not None
            and QUIT_PATTERN.search(utterance.phrase) is None
        )

    def perform(self, state: ConversationState, io: IOContainer) -> None:
        """stores the utterance and its recognized intent (None if no known intent matched)"""
        utterance: Utterance = io.i.interaction
        assert utterance.phrase is not None
        state.human_utterance = utterance
        state.human_intent, _confidence = self._intent_classifier(utterance.phrase, DEFAULT_LLM, num_trials=1)

class ReactToEstimateCostIntent(Operator[ConversationState]):
    """
    WHEN: the human asked for a price estimate
    THEN: plan the delivery with the planner, reply with a description of the plan and its cost, and get ready for
          the next utterance
    """

    TASK_DESC = "a pricing support agent is telling a human how it would deliver a package, and that it costs {cost} in total. Describe the plan as a compact list with step number, step action, and step cost. End with a final estimate."

    def can_perform(self, state: ConversationState, io: IOContainer) -> bool:
        """applies when the recognized intent of the current utterance is a price estimate"""
        return state.human_intent is HumanIntent.GET_PRICE_ESTIMATE

    def perform(self, state: ConversationState, io: IOContainer) -> None:
        """plans toward the planner's current goal, replies with a natural language description of the plan and its
        cost, and clears the utterance"""
        plan, cost = state.planner.plan()
        action_sq = [action for action in plan]
        description = describe_facts(action_sq, self.TASK_DESC.format(cost=cost), DEFAULT_LLM)
        io.o.interaction(Utterance(phrase=description))
        state.human_utterance = None
        state.human_intent = None


class ReactToDesiredOutcomeIntent(Operator[ConversationState]):
    """
    WHEN: the human said what outcome they want
    THEN: turn the outcome into goal facts for the planner, confirm it back to the human, and get ready for the
          next utterance
    """

    TASK_DESC = (
        "a human is telling a logistics support agent where they want packages or vehicles to end up; "
        "the agent has one office and one airport in each city it serves, and a package sent to a city "
        "should end up at that city's office unless the human names the airport or a vehicle"
    )

    def __init__(self, name: str, **kwargs: Any) -> None:
        """sets up the outcome populator once, so its prompt is reused for every utterance"""
        super().__init__(name, **kwargs)
        self._outcome_populator = ModelPopulator(DesiredOutcome, task_desc=self.TASK_DESC)

    def can_perform(self, state: ConversationState, io: IOContainer) -> bool:
        """applies when the recognized intent of the current utterance is a desired outcome"""
        return state.human_intent is HumanIntent.STATE_DESIRED_OUTCOME

    def perform(self, state: ConversationState, io: IOContainer) -> None:
        """extracts the At facts the human wants to hold, sets them as the planner's goal, confirms them, and
        clears the utterance; asks again when the human named nothing the cogent knows"""
        assert state.human_utterance is not None and state.human_utterance.phrase is not None
        world = state.reasoner.world
        outcome = self._outcome_populator(state.human_utterance.phrase, DEFAULT_LLM, *known_entities(world))
        try:
            goal = outcome.to_relations(state.reasoner.find_entity)
        except (KeyError, ValidationError):
            goal = []
        if not goal:
            reply = "Sorry, I did not catch where things should end up. Where should the package go?"
        else:
            state.planner.set_goal(goal)
            outcomes = "; ".join(f"the {fact.entity1.display_name} at the {fact.entity2.display_name}" for fact in goal)
            reply = f"Got it: {outcomes}. Ask me for a price estimate whenever you are ready."
        io.o.interaction(Utterance(phrase=reply))
        state.human_utterance = None
        state.human_intent = None


class ReactToAskAboutEntitiesIntent(Operator[ConversationState]):
    """
    WHEN: the human asked what things the carrier has, such as its trucks, airplanes, or the gas price
    THEN: work out which kinds of things they asked about, look them up in the world graph with the reasoner,
          describe them to the human, and get ready for the next utterance
    """

    QUERY_DESC = "a human is asking a logistics support agent what things it has, such as trucks, airplanes, or costs"
    ANSWER_DESC = "a logistics support agent is answering a human who asked: {question}. Describe these things briefly, using their display names."

    def __init__(self, name: str, **kwargs: Any) -> None:
        """sets up the query populator once, so its prompt is reused for every utterance"""
        super().__init__(name, **kwargs)
        self._query_populator = ModelPopulator(EntityQuery, task_desc=self.QUERY_DESC)

    def can_perform(self, state: ConversationState, io: IOContainer) -> bool:
        """applies when the recognized intent of the current utterance is a question about entities"""
        return state.human_intent is HumanIntent.ASK_ABOUT_ENTITIES

    def perform(self, state: ConversationState, io: IOContainer) -> None:
        """extracts the kinds asked about, finds the matching entities, describes them, and clears the utterance;
        says so when it cannot tell what was asked about or has nothing of that kind"""
        assert state.human_utterance is not None and state.human_utterance.phrase is not None
        question = state.human_utterance.phrase
        query = self._query_populator(question, DEFAULT_LLM)
        entities = list({e.name: e for kind in query.kinds for e in state.reasoner.find_entities(kind.entity_type)}.values())
        if not query.kinds:
            reply = "Sorry, I did not catch what you are asking about. I can tell you about our offices, airports, packages, trucks, airplanes, gas price, and labor costs."
        elif not entities:
            reply = "We do not have any of those right now."
        else:
            reply = describe_facts(entities, self.ANSWER_DESC.format(question=question), DEFAULT_LLM)
        io.o.interaction(Utterance(phrase=reply))
        state.human_utterance = None
        state.human_intent = None


class ReactToAskCurrentStateIntent(Operator[ConversationState]):
    """
    WHEN: the human asked where something currently is, or what is currently at a place
    THEN: work out which package, vehicle, or place they asked about, find the matching At relations with the
          reasoner, describe them to the human, and get ready for the next utterance
    """

    QUERY_DESC = (
        "a human is asking a logistics support agent where packages or vehicles currently are, "
        "or what is currently at an office, airport, or inside a vehicle"
    )
    ANSWER_DESC = "a logistics support agent is answering a human who asked: {question}. Each fact says that entity1 is currently at entity2. Answer briefly, using display names."

    def __init__(self, name: str, **kwargs: Any) -> None:
        """sets up the query populator once, so its prompt is reused for every utterance"""
        super().__init__(name, **kwargs)
        self._query_populator = ModelPopulator(StateQuery, task_desc=self.QUERY_DESC)

    def can_perform(self, state: ConversationState, io: IOContainer) -> bool:
        """applies when the recognized intent of the current utterance is a question about the current state"""
        return state.human_intent is HumanIntent.ASK_CURRENT_STATE

    def perform(self, state: ConversationState, io: IOContainer) -> None:
        """extracts the names asked about, finds the matching At relations, describes them, and clears the
        utterance; says so when it cannot tell what was asked about or finds nothing"""
        assert state.human_utterance is not None and state.human_utterance.phrase is not None
        question = state.human_utterance.phrase
        query = self._query_populator(question, DEFAULT_LLM, *known_entities(state.reasoner.world))
        if query.entity1 is None and query.entity2 is None:
            reply = "Sorry, I did not catch which package, vehicle, or place you are asking about."
        else:
            facts = state.reasoner.find_relations(At, query.entity1, query.entity2)
            if facts:
                reply = describe_facts(facts, self.ANSWER_DESC.format(question=question), DEFAULT_LLM)
            else:
                reply = "Nothing matches that right now."
        io.o.interaction(Utterance(phrase=reply))
        state.human_utterance = None
        state.human_intent = None


def _label(entity: Entity) -> str:
    """how to refer to an entity in a reply: its display name, or its name with spaces"""
    return getattr(entity, "display_name", entity.name.replace("_", " "))


class ReactToAddEntityIntent(Operator[ConversationState]):
    """
    WHEN: the human wants to add a new package, truck, or airplane
    THEN: work out what kind of thing it is and where it is, fill in its details from the utterance, add it to the
          world graph with the reasoner, confirm it, and get ready for the next utterance
    """

    ADDABLE = (EntityKind.PACKAGE, EntityKind.TRUCK, EntityKind.AIRPLANE)
    NEW_DESC = "a human is asking a logistics support agent to add a new package, truck, or airplane, at some place"
    DETAILS_DESC = "a human is asking a logistics support agent to add a new {kind}; fill in its details from what the human said"

    def __init__(self, name: str, **kwargs: Any) -> None:
        """sets up the new-entity populator once; detail populators are made per kind on first use"""
        super().__init__(name, **kwargs)
        self._new_populator = ModelPopulator(NewEntity, task_desc=self.NEW_DESC)
        self._details_populators: dict[EntityKind, ModelPopulator[Any]] = {}

    def can_perform(self, state: ConversationState, io: IOContainer) -> bool:
        """applies when the recognized intent of the current utterance is adding an entity"""
        return state.human_intent is HumanIntent.ADD_ENTITY

    def perform(self, state: ConversationState, io: IOContainer) -> None:
        """adds the new entity under a fresh name, placed where the human said, and confirms it; says so when the
        kind cannot be added, the place is missing or unknown, or the details do not fit"""
        assert state.human_utterance is not None and state.human_utterance.phrase is not None
        phrase = state.human_utterance.phrase
        new = self._new_populator(phrase, DEFAULT_LLM, *known_entities(state.reasoner.world))
        if new.kind not in self.ADDABLE:
            reply = "Sorry, I can only add packages, trucks, and airplanes."
        elif new.location is None:
            reply = f"Where is the new {new.kind.value}? Please tell me again, including the office, airport, or vehicle it is at."
        else:
            populator = self._details_populators.setdefault(
                new.kind, ModelPopulator(new.kind.entity_type, task_desc=self.DETAILS_DESC.format(kind=new.kind.value))
            )
            try:
                entity = populator(phrase, DEFAULT_LLM).model_copy(update={"name": state.reasoner.fresh_name(new.kind.value)})
                state.reasoner.add_entity(entity, new.location)
            except (UnexpectedModelBehavior, ValidationError):
                fields = ", ".join(f for f in new.kind.entity_type.model_fields if f != "name")
                reply = f"Sorry, I could not add that {new.kind.value}. Please tell me its {fields}, and where it is."
            except KeyError:
                reply = f"Sorry, I do not know of a place called {new.location}."
            else:
                reply = f"Added the {_label(entity)} ({new.kind.value}) at the {_label(state.reasoner.find_entity(new.location))}."
        io.o.interaction(Utterance(phrase=reply))
        state.human_utterance = None
        state.human_intent = None


class ReactToUpdateEntityIntent(Operator[ConversationState]):
    """
    WHEN: the human wants to change something that already exists, such as the gas price
    THEN: work out which entity it is, fill in its new values from the utterance and its current values, update it
          in the world graph with the reasoner, confirm what changed, and get ready for the next utterance
    """

    CHANGE_DESC = "a human is asking a logistics support agent to change something it has, such as the gas price, a labor cost, or a package"
    DETAILS_DESC = (
        "a human is asking a logistics support agent to change a {kind}; its current values are given as extra "
        "information. Keep every value the human did not ask to change"
    )

    def __init__(self, name: str, **kwargs: Any) -> None:
        """sets up the change populator once; detail populators are made per entity type on first use"""
        super().__init__(name, **kwargs)
        self._change_populator = ModelPopulator(EntityChange, task_desc=self.CHANGE_DESC)
        self._details_populators: dict[type[Entity], ModelPopulator[Any]] = {}

    def can_perform(self, state: ConversationState, io: IOContainer) -> bool:
        """applies when the recognized intent of the current utterance is updating an entity"""
        return state.human_intent is HumanIntent.UPDATE_ENTITY

    def perform(self, state: ConversationState, io: IOContainer) -> None:
        """updates the entity the human named and confirms the new values; says so when it cannot tell which entity,
        or nothing changed"""
        assert state.human_utterance is not None and state.human_utterance.phrase is not None
        phrase = state.human_utterance.phrase
        change = self._change_populator(phrase, DEFAULT_LLM, *known_entities(state.reasoner.world))
        current = state.reasoner.find_entity(change.entity) if change.entity is not None else None
        if current is None:
            reply = "Sorry, I did not catch what you want to change."
        else:
            entity_type = type(current)
            populator = self._details_populators.setdefault(
                entity_type, ModelPopulator(entity_type, task_desc=self.DETAILS_DESC.format(kind=entity_type.__name__))
            )
            try:
                updated = populator(phrase, DEFAULT_LLM, f"current values: {current.model_dump_json()}")
                updated = updated.model_copy(update={"name": current.name})
                state.reasoner.update_entity(updated)
            except (UnexpectedModelBehavior, ValidationError):
                reply = f"Sorry, I could not work out the new values for the {_label(current)}."
            else:
                old, new = dict(current.custom_fields), dict(updated.custom_fields)
                changes = [f"{field} is now {value}" for field, value in new.items() if old.get(field) != value]
                if changes:
                    reply = f"Updated the {_label(current)}: {', '.join(changes)}."
                else:
                    reply = f"The {_label(current)} already has those values; nothing changed."
        io.o.interaction(Utterance(phrase=reply))
        state.human_utterance = None
        state.human_intent = None


class RespondUnknown(Operator[ConversationState]):
    """
    WHEN: the human's utterance has been interpreted, and it matches none of the known intents
    THEN: tell the human the cogent is not designed to respond to that, and get ready for the next utterance
    """

    def can_perform(self, state: ConversationState, io: IOContainer) -> bool:
        """applies when the current utterance was interpreted but no known intent was recognized"""
        return state.human_utterance is not None and state.human_intent is None

    def perform(self, state: ConversationState, io: IOContainer) -> None:
        """replies that the cogent cannot respond, and clears the utterance"""
        io.o.interaction(Utterance(phrase="Sorry, I am not designed to respond to that."))
        state.human_utterance = None
        state.human_intent = None
