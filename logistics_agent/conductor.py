"""decision process of the logistics support cogent: the conversation state it keeps, and the operators that
decide what to do with each utterance

the conversation is managed with a stack of intents: the human's intents, pushed as they are recognized and popped
once they are done, and the agent's own: the asks for what it needs from the human, pushed when it finds something
missing and popped once it has it, the question whether to go ahead with a change to the world, such as adding a
vehicle to the fleet or closing a blocked route, and the note that it did not understand an utterance. RecognizeIntent only pushes;
every operator that works on an utterance applies to what is on top of the stack, so the top alone decides what
happens next.

within one turn, the operators that do the work write partial responses instead of speaking; once the stack is empty,
the turn ends with one reply that says them
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from cognition import EnumClassifier, IOContainer, Operator, describe_facts

from pydantic import BaseModel

from logistics_agent.communicator import (
    DEFAULT_LLM, AgentIntent, AskFor, AskForKind, ConfirmChange, HumanIntent, NotUnderstood, Utterance, in_context,
    label,
)
from logistics_agent.domain import (
    Action, Airport, At, City, RouteReport, RouteType, ShippingRequest, Vehicle, VehicleType,
)
from logistics_agent.drafts import build, draft_model, drafted
from logistics_agent.fill_placement import PlacementFiller
from logistics_agent.fill_route import RouteFiller
from logistics_agent.fill_shipping_request import ShippingRequestFiller
from logistics_agent.fill_vehicle import VehicleFiller
from logistics_agent.inspect_state import QuestionType, Sighting, StateInspector
from logistics_agent.planner import LogisticsPlanner
from logistics_agent.reasoner import Reasoner

QUIT_PHRASES = {"quit", "exit", "bye", "goodbye"}
QUIT_PATTERN = re.compile(rf"\b({'|'.join(QUIT_PHRASES)})\b", re.IGNORECASE)
INTRODUCTION = "Hi - I am your logistics support agent. I can help you estimate prices for various deliveries."

# utterances that only acknowledge the agent, such as "thanks!" or "ok"; the agent answers thanks, says nothing to the
# rest, and does not classify either
THANKS_PHRASES = ("thanks", "thank you", "thx", "cheers")
OK_PHRASES = ("ok", "okay", "k", "got it", "cool", "great", "sounds good", "alright", "all right", "perfect", "nice")
ACKNOWLEDGEMENT_PATTERN = re.compile(
    rf"\s*({'|'.join(map(re.escape, THANKS_PHRASES + OK_PHRASES))})(\s+(so much|a lot|very much))?[\s!.]*",
    re.IGNORECASE,
)


@dataclass
class ConversationState:
    """state of the conversation, kept across decision cycles"""

    # whether the agent has said its introduction; Introduce applies only until it has
    introduced: bool = False
    # whether the human said a quit phrase and the conversation is over
    done: bool = False
    # whether the agent is waiting for the human's next utterance, rather than working on the last one
    listening: bool = True
    # the utterance the agent is working on, which the operator for the intent it pushed reads; None while listening
    human_utterance: Utterance | None = None
    # what the conversation is about, most recent on top: what the human wants, and what the agent asked for; the top
    # decides what happens next
    intent_stack: list[HumanIntent | AgentIntent] = field(default_factory=list)
    # what the agent will say at the end of this turn, ahead of any question
    partial_responses: list[str] = field(default_factory=list)
    # the question the agent ended its last turn with, if any, which the human's next utterance may answer
    asked: str | None = None

    # maintains the world graph: terminals, airports, routes, vehicles, the shipment, and costs
    reasoner: Reasoner = field(default_factory=Reasoner)
    # plans deliveries over the reasoner's world graph, from its current facts toward its goal
    planner: LogisticsPlanner = field(init=False)
    # the delivery the human wants priced, as filled in from what they said, with a part None until they have given it;
    # None until they have described any of it
    shipping_request: ShippingRequest | None = None
    # whether the delivery has been planned for the price question being answered
    planned: bool = False
    # the cheapest actions found that deliver the shipping request; None until it has been planned, or when no
    # delivery could be planned
    plan: Sequence[Action] | None = None
    # what delivering the shipping request costs, in US dollars: the cost of the plan; None whenever plan is None
    price: float | None = None
    # the draft of the vehicle the human is adding to the fleet; None until the kind of vehicle is known
    vehicle_draft: BaseModel | None = None
    # the draft of where that vehicle is, an At relation with the vehicle as entity1; None until the vehicle is complete
    placement_draft: BaseModel | None = None
    # the route the human reports blocked: the cities at its ends; None until the human reports one
    route_report: RouteReport | None = None
    # the kind of the blocked route, a road or the flights; None while it is not known
    route_type: RouteType | None = None
    # the change to the world the agent asked the human to confirm, such as adding a vehicle or closing a route; None
    # while there is none
    pending_change: VehicleAddition | RouteClosure | None = None

    def __post_init__(self) -> None:
        """the planner plans over the world graph the reasoner maintains"""
        self.planner = LogisticsPlanner(self.reasoner.world)

    @property
    def top(self) -> HumanIntent | AgentIntent | None:
        """the intent on top of the stack, or None when the stack is empty"""
        return self.intent_stack[-1] if self.intent_stack else None

    def push(self, intent: HumanIntent | AgentIntent) -> None:
        """puts the intent on top of the stack, moving it there if an equal one is already further down"""
        if intent in self.intent_stack:
            self.intent_stack.remove(intent)
        self.intent_stack.append(intent)

    def pop(self, intent: HumanIntent | AgentIntent) -> None:
        """removes the intent from the top of the stack once it is done"""
        assert self.top == intent, f"expected {intent} on top of {self.intent_stack}"
        self.intent_stack.pop()

    def on_top(self, intent: HumanIntent | AgentIntent) -> bool:
        """whether the agent is working, with the intent on top of the stack"""
        return not self.listening and self.top == intent

    def ask_on_top(self) -> AskFor | AskForKind | None:
        """the ask on top of the stack while the agent is working, or None when something else is on top"""
        top = self.top
        return top if not self.listening and isinstance(top, AskFor | AskForKind) else None

    def has(self, ask: AskFor | AskForKind) -> bool:
        """whether the agent has what the ask asks for: the kind of vehicle once there is a draft of one, the kind of
        route once it is known, and otherwise what the model being filled in, or its draft, has"""
        if ask == AskForKind(VehicleType):
            return self.vehicle_draft is not None
        if ask == AskForKind(RouteType):
            return self.route_type is not None
        assert isinstance(ask, AskFor), f"the agent does not ask for {ask}"
        return ask.satisfied(self.model_of(ask.model))

    def model_of(self, model: type[BaseModel]) -> BaseModel | None:
        """the instance of the model the agent is filling in, or its draft, which the asks for the model are satisfied
        by"""
        if model is ShippingRequest:
            return self.shipping_request
        if model is At:
            return self.placement_draft
        if model is RouteReport:
            return self.route_report
        assert issubclass(model, Vehicle), f"the agent does not fill in {model.__name__}"
        draft = self.vehicle_draft
        return draft if draft is not None and drafted(draft) is model else None

    def asking_for_vehicle(self) -> bool:
        """whether the agent has asked for the kind of vehicle to add to the fleet, or the vehicle, and is still waiting
        for it"""
        return any(
            intent == AskForKind(VehicleType) or (isinstance(intent, AskFor) and issubclass(intent.model, Vehicle))
            for intent in self.intent_stack
        )

    def asking_for_route(self) -> bool:
        """whether the agent has asked which route is blocked, or its kind, and is still waiting for it"""
        return self.asking_for(RouteReport) or AskForKind(RouteType) in self.intent_stack

    def confirming(self) -> bool:
        """whether the agent has asked the human to confirm a change to the world, and is waiting for the answer"""
        return isinstance(self.top, ConfirmChange)

    def answered_ask(self) -> AskFor | AskForKind | None:
        """the ask the utterance answers: the agent ends a turn with a question only with its ask on top, so the ask
        right under the intent the utterance pushed is the one it answers; None when no ask is there"""
        below_top = self.intent_stack[-2] if len(self.intent_stack) >= 2 else None
        return below_top if isinstance(below_top, AskFor | AskForKind) else None

    def asking_for(self, model: type[BaseModel]) -> bool:
        """whether the agent has asked for the model, or an attribute of it, and is still waiting for it"""
        return any(isinstance(intent, AskFor) and intent.model is model for intent in self.intent_stack)

    def say(self, io: IOContainer, asking: str | None = None) -> None:
        """ends the turn: says the partial responses, then the question being asked, if any, and starts listening for
        the human again; remembers the question so the answer can be understood with it"""
        question = [asking] if asking is not None else []
        io.o.interaction(Utterance(phrase=" ".join([*self.partial_responses, *question])))
        self.partial_responses.clear()
        self.listen(asking)

    def listen(self, asking: str | None = None) -> None:
        """starts listening for the human again, after something else has spoken to them, done with the utterance;
        remembers the question they were asked, if any, so the answer can be understood with it"""
        self.listening = True
        self.human_utterance = None
        self.asked = asking


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
        state.partial_responses.append(INTRODUCTION)
        state.say(io)
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


def is_acknowledgement(phrase: str) -> bool:
    """whether the whole phrase only acknowledges the agent, such as "Thanks so much!" or "ok"; "ok, ship it from
    seattle" says more, so it is not"""
    return ACKNOWLEDGEMENT_PATTERN.fullmatch(phrase) is not None


class Acknowledge(Operator[ConversationState]):
    """
    WHEN: the agent is listening, and the human's utterance only acknowledges it, such as "thanks" or "ok", but not
          while the agent waits for a yes or no to adding a vehicle, where "ok" means yes
    THEN: reply to thanks, asking again the question the agent last asked, if any; say nothing to anything else, and
          keep waiting for the human
    """

    def can_perform(self, state: ConversationState, io: IOContainer) -> bool:
        """applies to an acknowledgement while the agent is listening"""
        utterance: Utterance = io.i.interaction
        return (
            state.listening
            and not state.confirming()
            and utterance.phrase is not None
            and is_acknowledgement(utterance.phrase)
        )

    def perform(self, state: ConversationState, io: IOContainer) -> None:
        """replies "You're welcome." to thanks, repeating the pending question so the conversation picks up where it
        was; leaves the state as it is for anything else, so the agent stays silent and the question stays pending"""
        utterance: Utterance = io.i.interaction
        assert utterance.phrase is not None
        if utterance.phrase.strip().lower().startswith(THANKS_PHRASES):
            state.partial_responses.append("You're welcome.")
            state.say(io, asking=state.asked)


class RecognizeIntent(Operator[ConversationState]):
    """
    WHEN: the agent is listening, and the human said something other than a quit phrase or an acknowledgement; an
          acknowledgement too while the agent waits for a yes or no to adding a vehicle
    THEN: recognize the intent expressed in it, using an EnumClassifier over the known HumanIntents, and push it onto
          the intent stack; push NotUnderstood when no known intent matched
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
        """applies to a new non-empty utterance that is not a quit phrase or an acknowledgement, while the agent is
        listening"""
        utterance: Utterance = io.i.interaction
        return (
            state.listening
            and utterance.phrase is not None
            and QUIT_PATTERN.search(utterance.phrase) is None
            and (state.confirming() or not is_acknowledgement(utterance.phrase))
        )

    def perform(self, state: ConversationState, io: IOContainer) -> None:
        """stores the utterance, stops listening, and pushes its recognized intent, or NotUnderstood, onto the stack"""
        utterance: Utterance = io.i.interaction
        assert utterance.phrase is not None
        state.listening = False
        state.human_utterance = utterance
        intent, _confidence = self._intent_classifier(
            in_context(state.asked, utterance.phrase), DEFAULT_LLM, num_trials=1
        )
        state.push(NotUnderstood() if intent is None else intent)


class RespondUnknown(Operator[ConversationState]):
    """
    WHEN: NotUnderstood is on top: the human's utterance matched none of the known intents
    THEN: note that the cogent is not designed to respond to that, and pop it, leaving the rest of the stack as it is
    """

    REPLY = "Sorry, I am not designed to respond to that."

    def can_perform(self, state: ConversationState, io: IOContainer) -> bool:
        """applies with NotUnderstood on top"""
        return state.on_top(NotUnderstood())

    def perform(self, state: ConversationState, io: IOContainer) -> None:
        """notes that the cogent cannot respond, and pops NotUnderstood"""
        state.partial_responses.append(self.REPLY)
        state.pop(NotUnderstood())


# === small talk


class ReactToGreetingIntent(Operator[ConversationState]):
    """
    WHEN: GREETING is on top: the human said hello, thanked the agent, or made small talk
    THEN: note a friendly reply, and pop the intent
    """

    REPLY = "Hello! I am happy to help with your deliveries."

    def can_perform(self, state: ConversationState, io: IOContainer) -> bool:
        """applies with GREETING on top"""
        return state.on_top(HumanIntent.GREETING)

    def perform(self, state: ConversationState, io: IOContainer) -> None:
        """notes the reply, and pops the intent"""
        state.partial_responses.append(self.REPLY)
        state.pop(HumanIntent.GREETING)


class ReactToAskCapabilitiesIntent(Operator[ConversationState]):
    """
    WHEN: ASK_CAPABILITIES is on top: the human asked what the agent can do
    THEN: note what the agent can help with, and pop the intent
    """

    REPLY = (
        "I can estimate what a delivery costs, from where a shipment is picked up to where it is dropped off."
    )

    def can_perform(self, state: ConversationState, io: IOContainer) -> bool:
        """applies with ASK_CAPABILITIES on top"""
        return state.on_top(HumanIntent.ASK_CAPABILITIES)

    def perform(self, state: ConversationState, io: IOContainer) -> None:
        """notes what the agent can do, and pops the intent"""
        state.partial_responses.append(self.REPLY)
        state.pop(HumanIntent.ASK_CAPABILITIES)


# === taking shipping requests


class ReactToShippingRequestIntent(Operator[ConversationState]):
    """
    WHEN: SHIPPING_REQUEST is on top: the human described a delivery they want, or part of it
    THEN: fill in the shipping request from the utterance with the ShippingRequestFiller sub-process: the request being
          asked for, or else a new one, reading the utterance as the answer to the ask just asked, if any; pop the
          intent, and ask for the request, with the asks for its missing parts
          above, so the request is confirmed once they have what they asked for
    """

    def __init__(self, name: str, **kwargs: Any) -> None:
        """sets up the ShippingRequestFiller sub-process once; it is restarted for every utterance"""
        super().__init__(name, **kwargs)
        self._shipping_request_filler = ShippingRequestFiller()

    def can_perform(self, state: ConversationState, io: IOContainer) -> bool:
        """applies with SHIPPING_REQUEST on top; it is pushed with the utterance that describes the delivery, and popped
        before the agent listens again, so the utterance is always the one that pushed it"""
        return state.on_top(HumanIntent.SHIPPING_REQUEST)

    def perform(self, state: ConversationState, io: IOContainer) -> None:
        """takes the request the sub-process fills in, pops the intent, and pushes the asks"""
        request = state.shipping_request if state.asking_for(ShippingRequest) else None
        sub_process = self._shipping_request_filler
        sub_process.reinit()
        result = sub_process(
            utterance=state.human_utterance, answering=state.answered_ask(), reasoner=state.reasoner,
            request=request or ShippingRequest(), suppress_errors=False,
        )
        assert result is not None and result.request is not None and result.asks is not None
        state.shipping_request = result.request
        state.pop(HumanIntent.SHIPPING_REQUEST)
        for ask in [AskFor(ShippingRequest), *reversed(result.asks)]:
            state.push(ask)


class ConfirmShippingRequest(Operator[ConversationState]):
    """
    WHEN: the ask for the whole shipping request is on top, and the request is complete
    THEN: note what the agent understood the delivery to be, inviting the price question unless the human has already
          asked it, and pop the ask
    """

    def can_perform(self, state: ConversationState, io: IOContainer) -> bool:
        """applies with the satisfied ask for the whole request on top"""
        return state.on_top(AskFor(ShippingRequest)) and AskFor(ShippingRequest).satisfied(state.shipping_request)

    def perform(self, state: ConversationState, io: IOContainer) -> None:
        """notes the confirmation, and pops the ask"""
        assert state.shipping_request is not None
        priced = HumanIntent.ESTIMATE_PRICE in state.intent_stack
        state.partial_responses.append(confirmation(state.shipping_request, priced))
        state.pop(AskFor(ShippingRequest))


def confirmation(request: ShippingRequest, priced: bool = False) -> str:
    """what the agent says back once it has the whole shipping request; it invites the price question unless the
    human has already asked it"""
    assert request.origin is not None and request.destination is not None and request.shipment is not None
    got_it = (
        f"Got it: the {request.shipment.details} ({request.shipment.weight:,g} lb) from "
        f"{request.origin.city.display_name} to {request.destination.city.display_name}."
    )
    return got_it if priced else f"{got_it} Ask me what it costs whenever you are ready."


# === asking for what the agent needs


class PopSatisfiedAsk(Operator[ConversationState]):
    """
    WHEN: an ask for an attribute, or for the kind of vehicle, is on top, and the agent has what it asked for
    THEN: pop the ask; the one below it is looked at next, so the ask for the whole model is reached once the asks for
          its attributes have popped, and the operator for that model takes it from there: ConfirmShippingRequest,
          PlaceVehicle, ProposeAddition, or ProposeClosure
    """

    def can_perform(self, state: ConversationState, io: IOContainer) -> bool:
        """applies with a satisfied ask for an attribute on top"""
        ask = state.ask_on_top()
        return ask is not None and (isinstance(ask, AskForKind) or ask.attribute is not None) and state.has(ask)

    def perform(self, state: ConversationState, io: IOContainer) -> None:
        """pops the ask"""
        assert state.top is not None
        state.pop(state.top)


class AskQuestion(Operator[ConversationState]):
    """
    WHEN: an ask is on top, and the agent does not have what it asked for yet
    THEN: end the turn by saying the partial responses and the ask's question; the ask stays on the stack until it has
          what it asked for
    """

    def can_perform(self, state: ConversationState, io: IOContainer) -> bool:
        """applies with an unsatisfied ask on top"""
        ask = state.ask_on_top()
        return ask is not None and not state.has(ask)

    def perform(self, state: ConversationState, io: IOContainer) -> None:
        """says the partial responses and the question, and listens for the answer"""
        assert isinstance(state.top, AskFor | AskForKind)
        state.say(io, asking=state.top.question)


# === adding vehicles to the fleet


class ReactToAddToFleetIntent(Operator[ConversationState]):
    """
    WHEN: ADD_TO_FLEET is on top: the human wants to add a vehicle to the fleet, or describes the one being added
    THEN: when the utterance answers where the vehicle is, fill in its At relation with the PlacementFiller
          sub-process; otherwise fill in the vehicle with the VehicleFiller sub-process: the one being asked for, or
          else a new one; pop the intent, and ask for the vehicle or its placement, with the asks for what is missing
          above
    """

    def __init__(self, name: str, **kwargs: Any) -> None:
        """sets up both sub-processes once; each is restarted for every utterance it handles"""
        super().__init__(name, **kwargs)
        self._vehicle_filler = VehicleFiller()
        self._placement_filler = PlacementFiller()

    def can_perform(self, state: ConversationState, io: IOContainer) -> bool:
        """applies with ADD_TO_FLEET on top"""
        return state.on_top(HumanIntent.ADD_TO_FLEET)

    def perform(self, state: ConversationState, io: IOContainer) -> None:
        """runs the sub-process the utterance is for, takes the draft it returns, pops the intent, and pushes the
        asks so the first is on top"""
        answering = state.answered_ask()
        state.pop(HumanIntent.ADD_TO_FLEET)
        if isinstance(answering, AskFor) and answering.model is At and state.placement_draft is not None:
            result = self._fill(self._placement_filler, state, answering, state.placement_draft)
            state.placement_draft = result.draft
            asks = [AskFor(At), *reversed(result.asks)]
        else:
            if not state.asking_for_vehicle():
                forget_fleet_addition(state)
            result = self._fill(self._vehicle_filler, state, answering, state.vehicle_draft)
            state.vehicle_draft = result.draft
            whole = [] if result.draft is None else [AskFor(drafted(result.draft))]
            asks = [*whole, *reversed(result.asks)]
        for ask in asks:
            state.push(ask)

    @staticmethod
    def _fill(
        sub_process: Any, state: ConversationState, answering: AskFor | AskForKind | None, draft: BaseModel | None
    ) -> Any:
        """runs a fresh sub-process over the utterance and the draft; returns its final state"""
        sub_process.reinit()
        result = sub_process(
            utterance=state.human_utterance, answering=answering, reasoner=state.reasoner, draft=draft,
            suppress_errors=False,
        )
        assert result is not None and result.asks is not None
        return result


def forget_fleet_addition(state: ConversationState) -> None:
    """drops the vehicle being added to the fleet and where it is, with every ask for them still on the stack, and
    the confirmation of adding it, if the agent asked for one"""
    confirming = isinstance(state.pending_change, VehicleAddition)
    state.intent_stack = [
        intent for intent in state.intent_stack
        if intent != AskForKind(VehicleType)
        and not (confirming and isinstance(intent, ConfirmChange))
        and not (isinstance(intent, AskFor) and (intent.model is At or issubclass(intent.model, Vehicle)))
    ]
    state.vehicle_draft = state.placement_draft = None
    if confirming:
        state.pending_change = None


class PlaceVehicle(Operator[ConversationState]):
    """
    WHEN: the ask for the whole vehicle being added is on top, and the vehicle is complete
    THEN: build the vehicle, start a draft of where it is, an At relation with the vehicle as entity1, and pop the ask;
          ask where the vehicle is
    """

    def can_perform(self, state: ConversationState, io: IOContainer) -> bool:
        """applies with the satisfied ask for a whole fleet vehicle on top"""
        ask = state.ask_on_top()
        return (
            isinstance(ask, AskFor) and ask.attribute is None and issubclass(ask.model, Vehicle) and state.has(ask)
        )

    def perform(self, state: ConversationState, io: IOContainer) -> None:
        """builds the vehicle, and asks for its placement"""
        assert state.vehicle_draft is not None and state.top is not None
        state.placement_draft = draft_model(At)(entity1=build(state.vehicle_draft))
        state.pop(state.top)
        state.push(AskFor(At))
        state.push(AskFor(At, "entity2"))


class ProposeAddition(Operator[ConversationState]):
    """
    WHEN: the ask for the whole placement of the vehicle being added is on top, and it is complete
    THEN: pop the ask, and ask the human to confirm adding the vehicle there
    """

    def can_perform(self, state: ConversationState, io: IOContainer) -> bool:
        """applies with the satisfied ask for the whole placement on top"""
        return state.on_top(AskFor(At)) and AskFor(At).satisfied(state.placement_draft)

    def perform(self, state: ConversationState, io: IOContainer) -> None:
        """swaps the ask for the placement for the confirmation of adding the vehicle there"""
        assert state.placement_draft is not None
        state.pending_change = VehicleAddition(build(state.placement_draft))
        state.pop(AskFor(At))
        state.push(ConfirmChange())


@dataclass(frozen=True)
class VehicleAddition:
    """a change the agent asks the human to confirm: adding a vehicle to the fleet, where it is"""

    # where the vehicle is: an At relation from the new vehicle to a terminal or airport
    placement: At

    @property
    def question(self) -> str:
        """what the agent asks the human to confirm"""
        vehicle, place = self.placement.entity1, self.placement.entity2
        return (
            f"Shall I add {vehicle.display_name}, a {vehicle.model} {type(vehicle).__name__.lower()} that carries up "
            f"to {vehicle.max_payload:,g} lb, at {label(place)}, to the fleet?"
        )

    def make(self, state: ConversationState) -> str:
        """adds the vehicle, and the At relation the human described for where it is; returns what to tell them"""
        state.reasoner.add_entity(self.placement.entity1)
        state.reasoner.add_relation(self.placement)
        return f"Added {self.placement.entity1.display_name} to the fleet, at {label(self.placement.entity2)}."

    def declined(self) -> str:
        """what to tell the human when they say no"""
        return f"OK, I did not add {self.placement.entity1.display_name} to the fleet."

    def forget(self, state: ConversationState) -> None:
        """drops everything the agent kept for the addition"""
        forget_fleet_addition(state)


# === closing blocked routes


class ReactToObstructRouteIntent(Operator[ConversationState]):
    """
    WHEN: OBSTRUCT_ROUTE is on top: the human reports a route blocked, or says which route or which kind it is
    THEN: fill in the route with the RouteFiller sub-process: the one being asked for, or else a new one; pop the
          intent, and ask for the route, with the asks for what is missing above
    """

    def __init__(self, name: str, **kwargs: Any) -> None:
        """sets up the sub-process once; it is restarted for every utterance"""
        super().__init__(name, **kwargs)
        self._route_filler = RouteFiller()

    def can_perform(self, state: ConversationState, io: IOContainer) -> bool:
        """applies with OBSTRUCT_ROUTE on top"""
        return state.on_top(HumanIntent.OBSTRUCT_ROUTE)

    def perform(self, state: ConversationState, io: IOContainer) -> None:
        """takes the report and the kind the sub-process fills in, pops the intent, and pushes the asks so the first
        is on top"""
        answering = state.answered_ask()
        state.pop(HumanIntent.OBSTRUCT_ROUTE)
        if not state.asking_for_route():
            forget_route_report(state)
        sub_process = self._route_filler
        sub_process.reinit()
        result = sub_process(
            utterance=state.human_utterance, answering=answering, reasoner=state.reasoner,
            report=state.route_report or RouteReport(), kind=state.route_type, suppress_errors=False,
        )
        assert result is not None and result.report is not None and result.asks is not None
        state.route_report, state.route_type = result.report, result.kind
        for ask in [AskFor(RouteReport), *reversed(result.asks)]:
            state.push(ask)


def forget_route_report(state: ConversationState) -> None:
    """drops the route reported blocked and its kind, with every ask for them still on the stack, and the confirmation
    of closing it, if the agent asked for one"""
    confirming = isinstance(state.pending_change, RouteClosure)
    state.intent_stack = [
        intent for intent in state.intent_stack
        if intent != AskForKind(RouteType)
        and not (confirming and isinstance(intent, ConfirmChange))
        and not (isinstance(intent, AskFor) and intent.model is RouteReport)
    ]
    state.route_report, state.route_type = None, None
    if confirming:
        state.pending_change = None


def blocked_route(state: ConversationState) -> Any:
    """the route of the kind between the two cities reported, in either direction, as the world graph has it: the
    road between the cities, or the air route between their airports; None when there is none"""
    report, kind = state.route_report, state.route_type
    assert report is not None and report.city1 is not None and report.city2 is not None and kind is not None
    ends = [report.city1, report.city2]
    if kind is RouteType.AIR:
        airports = state.reasoner.find_entities(Airport)
        ends = [next((airport for airport in airports if airport.city.name == city.name), city) for city in ends]
    for one, other in (ends, ends[::-1]):
        found = state.reasoner.find_relations(kind.model, one.name, other.name)
        if found:
            return found[0]
    return None


def route_phrase(kind: RouteType, city1: City, city2: City) -> str:
    """how the agent refers to the route: the road, or the flights, between the two cities"""
    what = "road" if kind is RouteType.ROAD else "flights"
    return f"the {what} between {city1.display_name} and {city2.display_name}"


class ProposeClosure(Operator[ConversationState]):
    """
    WHEN: the ask for the whole route report is on top, both cities are known, and so is the kind of route
    THEN: pop the ask; when the world graph has such a route, ask the human to confirm closing it, and otherwise note
          that there is no such route and drop the report
    """

    def can_perform(self, state: ConversationState, io: IOContainer) -> bool:
        """applies with the satisfied ask for the whole report on top, once the kind is known"""
        return (
            state.on_top(AskFor(RouteReport))
            and AskFor(RouteReport).satisfied(state.route_report)
            and state.route_type is not None
        )

    def perform(self, state: ConversationState, io: IOContainer) -> None:
        """swaps the ask for the report for the confirmation of closing the route, when there is one"""
        report, kind = state.route_report, state.route_type
        assert report is not None and report.city1 is not None and report.city2 is not None and kind is not None
        state.pop(AskFor(RouteReport))
        route = blocked_route(state)
        if route is None:
            state.partial_responses.append(
                f"I don't know of {route_phrase(kind, report.city1, report.city2)}, so there is nothing to close."
            )
            forget_route_report(state)
            return
        state.pending_change = RouteClosure(route, kind, report.city1, report.city2)
        state.push(ConfirmChange())


@dataclass(frozen=True)
class RouteClosure:
    """a change the agent asks the human to confirm: closing a blocked route, so deliveries are planned around it"""

    # the route as the world graph has it, a TruckRoute or an AirRoute
    route: Any
    kind: RouteType
    # the cities the human named, at the ends of the route
    city1: City
    city2: City

    @property
    def question(self) -> str:
        """what the agent asks the human to confirm"""
        return f"Shall I close {route_phrase(self.kind, self.city1, self.city2)}?"

    def make(self, state: ConversationState) -> str:
        """removes the route from the world graph; returns what to tell the human"""
        state.reasoner.remove_relation(self.kind.model, self.route.entity1.name, self.route.entity2.name)
        return f"Closed {route_phrase(self.kind, self.city1, self.city2)}; I will plan deliveries around it."

    def declined(self) -> str:
        """what to tell the human when they say no"""
        return f"OK, I left {route_phrase(self.kind, self.city1, self.city2)} open."

    def forget(self, state: ConversationState) -> None:
        """drops everything the agent kept for the closure"""
        forget_route_report(state)


# === answering questions about the state of the world


class ReactToInspectStateIntent(Operator[ConversationState]):
    """
    WHEN: INSPECT_STATE is on top: the human asked about the state of the world, such as which vehicles the fleet has
    THEN: answer it with the StateInspector sub-process, which looks the facts up in the world graph; note the answer,
          worded from the facts, or that the agent cannot answer that kind of question; pop the intent
    """

    CANNOT_ANSWER = "Sorry, I can only answer questions about the fleet: which vehicles it has, and where they are."

    def __init__(self, name: str, **kwargs: Any) -> None:
        """sets up the sub-process once; it is restarted for every question"""
        super().__init__(name, **kwargs)
        self._state_inspector = StateInspector()

    def can_perform(self, state: ConversationState, io: IOContainer) -> bool:
        """applies with INSPECT_STATE on top"""
        return state.on_top(HumanIntent.INSPECT_STATE)

    def perform(self, state: ConversationState, io: IOContainer) -> None:
        """runs a fresh sub-process over the question, notes its answer, and pops the intent"""
        answering = state.answered_ask()
        sub_process = self._state_inspector
        sub_process.reinit()
        result = sub_process(
            utterance=state.human_utterance, answering=answering, reasoner=state.reasoner, suppress_errors=False
        )
        assert result is not None
        if result.question_type is QuestionType.FLEET:
            assert result.fleet_question is not None and result.sightings is not None
            state.partial_responses.append(fleet_answer(result.fleet_question.kind, result.sightings))
        else:
            state.partial_responses.append(self.CANNOT_ANSWER)
        state.pop(HumanIntent.INSPECT_STATE)


def with_article(noun: str) -> str:
    """the noun with a or an in front, as it sounds, such as "a truck" or "an airplane" """
    return f"{'an' if noun[:1].lower() in 'aeiou' else 'a'} {noun}"


def kind_of(vehicle: Vehicle) -> VehicleType:
    """the type of the vehicle; the world graph gives back frozen copies of the vehicles, whose class is not the
    model, so the type is the one whose model the vehicle is an instance of"""
    return next(kind for kind in VehicleType if isinstance(vehicle, kind.model))


def fleet_answer(kind: VehicleType | None, sightings: list[Sighting]) -> str:
    """the answer to a question about the fleet, or the vehicles of one kind in it, worded from the facts: how many
    there are, then each with its make and model, its payload, and where it is"""
    one = "vehicle" if kind is None else kind.value
    if not sightings:
        return f"The fleet has no {one}s."
    vehicles = "; ".join(
        f"{v.display_name} ({v.model}), {with_article(kind_of(v).value)} that carries up to {v.max_payload:,g} lb, "
        f"at {label(place) if place is not None else 'a place I do not know'}"
        for v, place in ((sighting.vehicle, sighting.place) for sighting in sightings)
    )
    count = f"{len(sightings)} {one}" if len(sightings) == 1 else f"{len(sightings)} {one}s"
    return f"The fleet has {count}: {vehicles}."


# === confirming changes to the world


class AskToConfirmChange(Operator[ConversationState]):
    """
    WHEN: ConfirmChange is on top: the agent has a change to make to the world, and the human has not answered whether
          to make it
    THEN: end the turn by saying the partial responses and asking whether to make the change; ConfirmChange stays on
          the stack until the human says yes or no
    """

    def can_perform(self, state: ConversationState, io: IOContainer) -> bool:
        """applies with ConfirmChange on top"""
        return state.on_top(ConfirmChange())

    def perform(self, state: ConversationState, io: IOContainer) -> None:
        """says the partial responses and the change's question, and listens for the answer"""
        assert state.pending_change is not None
        state.say(io, asking=state.pending_change.question)


class ReactToConfirmIntent(Operator[ConversationState]):
    """
    WHEN: CONFIRM is on top: the human said yes
    THEN: pop it; when the agent asked whether to make a change, make it, note what was done, pop ConfirmChange, and
          drop what was kept for the change; a yes to nothing is let go
    """

    def can_perform(self, state: ConversationState, io: IOContainer) -> bool:
        """applies with CONFIRM on top"""
        return state.on_top(HumanIntent.CONFIRM)

    def perform(self, state: ConversationState, io: IOContainer) -> None:
        """makes the change when there is one to confirm"""
        state.pop(HumanIntent.CONFIRM)
        if not state.confirming():
            return
        change = state.pending_change
        assert change is not None
        state.partial_responses.append(change.make(state))
        state.pop(ConfirmChange())
        change.forget(state)


class ReactToDenyIntent(Operator[ConversationState]):
    """
    WHEN: DENY is on top: the human said no
    THEN: pop it; when the agent asked whether to make a change, note that it was not made, pop ConfirmChange, and drop
          what was kept for the change; a no to nothing is let go
    """

    def can_perform(self, state: ConversationState, io: IOContainer) -> bool:
        """applies with DENY on top"""
        return state.on_top(HumanIntent.DENY)

    def perform(self, state: ConversationState, io: IOContainer) -> None:
        """drops the change when there is one to confirm"""
        state.pop(HumanIntent.DENY)
        if not state.confirming():
            return
        change = state.pending_change
        assert change is not None
        state.partial_responses.append(change.declined())
        state.pop(ConfirmChange())
        change.forget(state)


# === estimating prices


class ReactToEstimatePriceIntent(Operator[ConversationState]):
    """
    WHEN: ESTIMATE_PRICE is on top: the human asked what a delivery costs, and the shipping request is not complete
    THEN: ask for the request, and for each part still missing from it, so the price question waits under the asks
          until they have what they asked for; the intent stays on the stack until the plan has been described
    """

    def can_perform(self, state: ConversationState, io: IOContainer) -> bool:
        """applies with ESTIMATE_PRICE on top, while the request is not complete"""
        return state.on_top(HumanIntent.ESTIMATE_PRICE) and not AskFor(ShippingRequest).satisfied(state.shipping_request)

    def perform(self, state: ConversationState, io: IOContainer) -> None:
        """pushes the ask for the whole request, then the asks for its missing parts, the first of them on top; an ask
        already on the stack is moved up"""
        request = state.shipping_request
        state.push(AskFor(ShippingRequest))
        if request is not None:
            for part in reversed(request.missing()):
                state.push(AskFor(ShippingRequest, part))


class PlanDelivery(Operator[ConversationState]):
    """
    WHEN: ESTIMATE_PRICE is on top, the shipping request is complete, and the delivery has not been planned yet
    THEN: plan the delivery in the shipping request, from its origin to its destination, and record the plan and its
          price
    """

    def can_perform(self, state: ConversationState, io: IOContainer) -> bool:
        """applies with ESTIMATE_PRICE on top and a complete request to plan"""
        return (
            state.on_top(HumanIntent.ESTIMATE_PRICE)
            and not state.planned
            and AskFor(ShippingRequest).satisfied(state.shipping_request)
        )

    def perform(self, state: ConversationState, io: IOContainer) -> None:
        """puts the shipping request's locations and shipment in the world graph, plans moving the shipment from the
        origin to the destination, and takes them out again; records the plan and its cost as the price, or None for
        both when no delivery could be planned"""
        request, planner = state.shipping_request, state.planner
        assert request is not None and request.origin and request.destination and request.shipment
        planner.setup(request.origin, request.destination, request.shipment)
        planner.set_current([At(entity1=request.shipment, entity2=request.origin)])
        planner.set_goal([At(entity1=request.shipment, entity2=request.destination)])
        try:
            state.plan, state.price = planner.plan()
            #print(state.plan)
        except RuntimeError:
            state.plan, state.price = None, None
        finally:
            planner.clear()
        state.planned = True


class DescribePlan(Operator[ConversationState]):
    """
    WHEN: ESTIMATE_PRICE is on top, and the delivery has been planned
    THEN: note the price, and a description of the plan's steps written by describe_facts; or note that no delivery
          could be planned; pop the intent
    """

    TASK_DESC = (
        "a logistics support agent tells a human how their shipment would be delivered;"
        "describe all the steps as an itemized list in the order they happen, naming vehicles, places, and what each step costs"
    )

    def can_perform(self, state: ConversationState, io: IOContainer) -> bool:
        """applies with ESTIMATE_PRICE on top, once the delivery has been planned"""
        return state.on_top(HumanIntent.ESTIMATE_PRICE) and state.planned

    def perform(self, state: ConversationState, io: IOContainer) -> None:
        """notes what delivering the shipping request costs, which the agent states itself, and how the shipment
        would be delivered, which the language model describes from the plan's actions"""
        request = state.shipping_request
        assert request is not None and request.origin and request.destination and request.shipment
        delivery = (
            f"the {request.shipment.details} from {request.origin.city.display_name} to "
            f"{request.destination.city.display_name}"
        )
        if state.plan is None or state.price is None:
            state.partial_responses.append(f"Sorry, I could not find a way to deliver {delivery}.")
        else:
            #print(len(state.plan))
            steps = describe_facts(state.plan, self.TASK_DESC, DEFAULT_LLM)
            state.partial_responses.append(f"Delivering {delivery} costs ${state.price:,.2f}. \n \n {steps}")
        state.planned = False
        state.pop(HumanIntent.ESTIMATE_PRICE)


# === ending the turn


class Reply(Operator[ConversationState]):
    """
    WHEN: the agent is working and the intent stack is empty, so there is nothing left to do or ask
    THEN: end the turn by saying the partial responses
    """

    def can_perform(self, state: ConversationState, io: IOContainer) -> bool:
        """applies once the stack is empty"""
        return not state.listening and not state.intent_stack

    def perform(self, state: ConversationState, io: IOContainer) -> None:
        """says the partial responses"""
        state.say(io)
