"""decision process of the logistics support cogent: the conversation state it keeps, and the operators that
decide what to do with each utterance

the conversation is managed with a stack of intents: the human's intents, pushed as they are recognized and popped
once they are done, and the agent's own: the asks for what it needs from the human, pushed when it finds something
missing and popped once it has it, and the note that it did not understand an utterance. RecognizeIntent only pushes;
every operator that works on an utterance applies to what is on top of the stack, so the top alone decides what
happens next.

within one turn, the operators that do the work write partial responses instead of speaking; once the stack is empty,
the turn ends with one reply that says them
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from cognition import EnumClassifier, IOContainer, Operator, describe_facts

from pydantic import BaseModel

from logistics_agent.communicator import DEFAULT_LLM, AgentIntent, AskFor, HumanIntent, NotUnderstood, Utterance
from logistics_agent.domain import Action, At, ShippingRequest
from logistics_agent.fill_shipping_request import ShippingRequestFiller
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

    def ask_on_top(self) -> AskFor | None:
        """the ask on top of the stack while the agent is working, or None when something else is on top"""
        top = self.top
        return top if not self.listening and isinstance(top, AskFor) else None

    def model_of(self, model: type[BaseModel]) -> BaseModel | None:
        """the instance of the model the agent is filling in, which the asks for the model are satisfied by"""
        assert model is ShippingRequest, f"the agent does not fill in {model.__name__}"
        return self.shipping_request

    def answered_ask(self) -> AskFor | None:
        """the ask the utterance answers: the agent ends a turn with a question only with its ask on top, so the ask
        right under the intent the utterance pushed is the one it answers; None when no ask is there"""
        below_top = self.intent_stack[-2] if len(self.intent_stack) >= 2 else None
        return below_top if isinstance(below_top, AskFor) else None

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
    WHEN: the agent is listening, and the human's utterance only acknowledges it, such as "thanks" or "ok"
    THEN: reply to thanks, asking again the question the agent last asked, if any; say nothing to anything else, and
          keep waiting for the human
    """

    def can_perform(self, state: ConversationState, io: IOContainer) -> bool:
        """applies to an acknowledgement while the agent is listening"""
        utterance: Utterance = io.i.interaction
        return state.listening and utterance.phrase is not None and is_acknowledgement(utterance.phrase)

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
    WHEN: the agent is listening, and the human said something other than a quit phrase or an acknowledgement
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
            and not is_acknowledgement(utterance.phrase)
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


def in_context(asked: str | None, phrase: str) -> str:
    """the phrase as the intent classifier should see it: after the agent asked a question, a short answer such as
    "denver" is classified together with that question"""
    if asked is None:
        return phrase
    return f'the agent asked: "{asked}" and the human answered: "{phrase}"'


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
    WHEN: an ask for an attribute is on top, and the agent has what it asked for
    THEN: pop the ask; the one below it is looked at next, so the ask for the whole model is reached once the asks for
          its attributes have popped, and its own operator confirms it
    """

    def can_perform(self, state: ConversationState, io: IOContainer) -> bool:
        """applies with a satisfied ask for an attribute on top"""
        ask = state.ask_on_top()
        return ask is not None and ask.attribute is not None and ask.satisfied(state.model_of(ask.model))

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
        return ask is not None and not ask.satisfied(state.model_of(ask.model))

    def perform(self, state: ConversationState, io: IOContainer) -> None:
        """says the partial responses and the question, and listens for the answer"""
        assert isinstance(state.top, AskFor)
        state.say(io, asking=state.top.question)


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
        "describe the steps as an itemized list in the order they happen, naming vehicles, places, and what each step costs"
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
