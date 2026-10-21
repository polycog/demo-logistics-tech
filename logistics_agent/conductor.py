"""decision process of the logistics support cogent: the conversation state it keeps, and the operators that
decide what to do with each utterance

the conversation is managed with a stack of intents. The human's intents are pushed as they are recognized, and each
intent is popped once it is done. What is on top decides what happens next.

within one turn, the operators that do the work write partial responses instead of speaking; once the stack is empty,
the turn ends with one reply that says them
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from cognition import EnumClassifier, IOContainer, Operator, describe_facts

from logistics_agent.communicator import DEFAULT_LLM, HumanIntent, Utterance
from logistics_agent.domain import Action, At, ShippingRequest
from logistics_agent.extract_shipping_request import ExtractShippingRequest
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
    # the utterance being handled, until an operator handles it
    human_utterance: Utterance | None = None
    # the intent recognized in human_utterance; None when no known intent matched
    human_intent: HumanIntent | None = None
    # what the conversation is about, most recent on top; the top decides what happens next
    intent_stack: list[HumanIntent] = field(default_factory=list)
    # what the agent will say at the end of this turn, ahead of any question
    partial_responses: list[str] = field(default_factory=list)
    # the question the agent ended its last turn with, if any, which the human's next utterance may answer
    asked: str | None = None

    # maintains the world graph: terminals, airports, routes, vehicles, the shipment, and costs
    reasoner: Reasoner = field(default_factory=Reasoner)
    # plans deliveries over the reasoner's world graph, from its current facts toward its goal
    planner: LogisticsPlanner = field(init=False)
    # the delivery the human wants priced, as extracted from what they said; None until they have described it
    shipping_request: ShippingRequest | None = None
    # the cheapest actions found that deliver the shipping request; None until it has been planned, or when no
    # delivery could be planned
    plan: Sequence[Action] | None = None
    # what delivering the shipping request costs, in US dollars: the cost of the plan; None whenever plan is None
    price: float | None = None

    def __post_init__(self) -> None:
        """the planner plans over the world graph the reasoner maintains"""
        self.planner = LogisticsPlanner(self.reasoner.world)

    @property
    def top(self) -> HumanIntent | None:
        """the intent on top of the stack, or None when the stack is empty"""
        return self.intent_stack[-1] if self.intent_stack else None

    def push(self, intent: HumanIntent) -> None:
        """puts the intent on top of the stack, moving it there if it is already further down"""
        if intent in self.intent_stack:
            self.intent_stack.remove(intent)
        self.intent_stack.append(intent)

    def pop(self, intent: HumanIntent) -> None:
        """removes the intent from the top of the stack once it is done"""
        assert self.top is intent, f"expected {intent} on top of {self.intent_stack}"
        self.intent_stack.pop()

    def understood(self, intent: HumanIntent) -> bool:
        """whether there is an utterance still to handle, and the agent understood it as expressing the intent"""
        return not self.listening and self.human_utterance is not None and self.human_intent is intent

    def not_understood_any(self) -> bool:
        """whether there is an utterance still to handle, and the agent understood it as none of the intents it knows
        how to handle"""
        return not self.listening and self.human_utterance is not None and self.human_intent is None

    def in_focus(self, intent: HumanIntent) -> bool:
        """whether the agent is between utterances, with the intent on top of the stack"""
        return not self.listening and self.human_utterance is None and self.top is intent

    def handled(self) -> None:
        """forgets the utterance once it has been handled"""
        self.human_utterance = None
        self.human_intent = None

    def say(self, io: IOContainer, asking: str | None = None) -> None:
        """ends the turn: says the partial responses, then the question being asked, if any, and starts listening for
        the human again; remembers the question so the answer can be understood with it"""
        question = [asking] if asking is not None else []
        io.o.interaction(Utterance(phrase=" ".join([*self.partial_responses, *question])))
        self.partial_responses.clear()
        self.listen(asking)

    def listen(self, asking: str | None = None) -> None:
        """starts listening for the human again, after something else has spoken to them; remembers the question they
        were asked, if any, so the answer can be understood with it"""
        self.listening = True
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
          the intent stack
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
        """stores the utterance and its recognized intent (None if no known intent matched), stops listening, and
        pushes the intent onto the stack"""
        utterance: Utterance = io.i.interaction
        assert utterance.phrase is not None
        state.listening = False
        state.human_utterance = utterance
        state.human_intent, _confidence = self._intent_classifier(
            in_context(state.asked, utterance.phrase), DEFAULT_LLM, num_trials=1
        )
        if state.human_intent is not None:
            state.push(state.human_intent)


def in_context(asked: str | None, phrase: str) -> str:
    """the phrase as the intent classifier should see it: after the agent asked a question, a short answer such as
    "denver" is classified together with that question"""
    if asked is None:
        return phrase
    return f'the agent asked: "{asked}" and the human answered: "{phrase}"'


class RespondUnknown(Operator[ConversationState]):
    """
    WHEN: the human's utterance matches none of the known intents
    THEN: note that the cogent is not designed to respond to that, and leave the intent stack as it is
    """

    def can_perform(self, state: ConversationState, io: IOContainer) -> bool:
        """applies when the current utterance was interpreted but no known intent was recognized"""
        return state.not_understood_any()

    def perform(self, state: ConversationState, io: IOContainer) -> None:
        """notes that the cogent cannot respond, and forgets the utterance"""
        state.partial_responses.append("Sorry, I am not designed to respond to that.")
        state.handled()


# === small talk


class ReactToGreetingIntent(Operator[ConversationState]):
    """
    WHEN: the human said hello, thanked the agent, or made small talk
    THEN: note a friendly reply, and pop the intent
    """

    REPLY = "Hello! I am happy to help with your deliveries."

    def can_perform(self, state: ConversationState, io: IOContainer) -> bool:
        """applies to an utterance that is a greeting or small talk"""
        return state.understood(HumanIntent.GREETING)

    def perform(self, state: ConversationState, io: IOContainer) -> None:
        """notes the reply, and forgets the utterance"""
        state.partial_responses.append(self.REPLY)
        state.pop(HumanIntent.GREETING)
        state.handled()


class ReactToAskCapabilitiesIntent(Operator[ConversationState]):
    """
    WHEN: the human asked what the agent can do
    THEN: note what the agent can help with, and pop the intent
    """

    REPLY = (
        "I can estimate what a delivery costs, from where a shipment is picked up to where it is dropped off."
    )

    def can_perform(self, state: ConversationState, io: IOContainer) -> bool:
        """applies to an utterance asking what the agent can do"""
        return state.understood(HumanIntent.ASK_CAPABILITIES)

    def perform(self, state: ConversationState, io: IOContainer) -> None:
        """notes what the agent can do, and forgets the utterance"""
        state.partial_responses.append(self.REPLY)
        state.pop(HumanIntent.ASK_CAPABILITIES)
        state.handled()


# === taking shipping requests


class ReactToShippingRequestIntent(Operator[ConversationState]):
    """
    WHEN: the human described a delivery they want
    THEN: hand the utterance to the ExtractShippingRequest sub-process; take the shipping request once the sub-process
          has populated its origin, destination, and shipment, and confirm it; pop the intent
    """

    NOT_UNDERSTOOD = "Sorry, I did not understand the delivery you described."

    def __init__(self, name: str, **kwargs: Any) -> None:
        """sets up the ExtractShippingRequest sub-process once; it is restarted for every utterance"""
        super().__init__(name, **kwargs)
        self._extract_shipping_request = ExtractShippingRequest()

    def can_perform(self, state: ConversationState, io: IOContainer) -> bool:
        """applies to an utterance describing a delivery"""
        return state.understood(HumanIntent.SHIPPING_REQUEST)

    def perform(self, state: ConversationState, io: IOContainer) -> None:
        """runs a fresh sub-process with the human's utterance and the reasoner, which its operators read from io.a;
        takes the shipping request when the sub-process populated all of it and notes a confirmation, and otherwise
        notes that the delivery was not understood; forgets the utterance"""
        assert state.human_utterance is not None and state.human_utterance.phrase is not None
        sub_process = self._extract_shipping_request
        sub_process.reinit()
        result = sub_process(utterance=state.human_utterance, reasoner=state.reasoner)
        if result is not None and result.origin and result.destination and result.shipment:
            state.shipping_request = request = ShippingRequest(
                origin=result.origin, destination=result.destination, shipment=result.shipment
            )
            state.partial_responses.append(confirmation(request))
        else:
            state.partial_responses.append(self.NOT_UNDERSTOOD)
        state.pop(HumanIntent.SHIPPING_REQUEST)
        state.handled()


def confirmation(request: ShippingRequest) -> str:
    """what the agent says back once it has the whole shipping request"""
    assert request.origin is not None and request.destination is not None and request.shipment is not None
    return (
        f"Got it: the {request.shipment.details} ({request.shipment.weight:,g} lb) from "
        f"{request.origin.city.display_name} to {request.destination.city.display_name}. "
        "Ask me what it costs whenever you are ready."
    )


# === estimating prices


class AskForShippingRequest(Operator[ConversationState]):
    """
    WHEN: the human asked what a delivery costs, but has not described the delivery yet
    THEN: note that the agent needs the delivery first, and pop the intent
    """

    REPLY = (
        "Tell me about the delivery first: what you are shipping and how much it weighs, and the cities it goes from "
        "and to."
    )

    def can_perform(self, state: ConversationState, io: IOContainer) -> bool:
        """applies to an utterance asking for a price while there is no shipping request"""
        return state.understood(HumanIntent.ESTIMATE_PRICE) and state.shipping_request is None

    def perform(self, state: ConversationState, io: IOContainer) -> None:
        """notes what the agent needs, and forgets the utterance"""
        state.partial_responses.append(self.REPLY)
        state.pop(HumanIntent.ESTIMATE_PRICE)
        state.handled()


class ReactToEstimatePriceIntent(Operator[ConversationState]):
    """
    WHEN: the human asked what a delivery costs
    THEN: plan the delivery in the shipping request, from its origin to its destination, and record the plan and its
          price; the intent stays on the stack until the plan has been described
    """

    def can_perform(self, state: ConversationState, io: IOContainer) -> bool:
        """applies to an utterance asking for a price"""
        return state.understood(HumanIntent.ESTIMATE_PRICE) and state.shipping_request is not None

    def perform(self, state: ConversationState, io: IOContainer) -> None:
        """puts the shipping request's locations and shipment in the world graph, plans moving the shipment from the
        origin to the destination, and takes them out again; records the plan and its cost as the price, or None for
        both when no delivery could be planned; forgets the utterance"""
        request, planner = state.shipping_request, state.planner
        assert request is not None
        planner.setup(request.origin, request.destination, request.shipment)
        planner.set_current([At(entity1=request.shipment, entity2=request.origin)])
        planner.set_goal([At(entity1=request.shipment, entity2=request.destination)])
        try:
            state.plan, state.price = planner.plan()
        except RuntimeError:
            state.plan, state.price = None, None
        finally:
            planner.clear()
        state.handled()


class DescribePlan(Operator[ConversationState]):
    """
    WHEN: the agent is working on the human's question about what a delivery costs, and has planned the delivery
    THEN: note the price, and a description of the plan's steps written by describe_facts; or note that no delivery
          could be planned; pop the intent
    """

    TASK_DESC = (
        "a logistics support agent tells a human how their shipment would be delivered;"
        "describe the steps as an itemized list in the order they happen, naming vehicles, places, and what each step costs"
    )

    def can_perform(self, state: ConversationState, io: IOContainer) -> bool:
        """applies between utterances, with the price question on top of the stack"""
        return state.in_focus(HumanIntent.ESTIMATE_PRICE)

    def perform(self, state: ConversationState, io: IOContainer) -> None:
        """notes what delivering the shipping request costs, which the agent states itself, and how the shipment
        would be delivered, which the language model describes from the plan's actions"""
        request = state.shipping_request
        assert request is not None and request.shipment is not None
        delivery = (
            f"the {request.shipment.details} from {request.origin.city.display_name} to "
            f"{request.destination.city.display_name}"
        )
        if state.plan is None or state.price is None:
            state.partial_responses.append(f"Sorry, I could not find a way to deliver {delivery}.")
        else:
            steps = describe_facts(state.plan, self.TASK_DESC, DEFAULT_LLM)
            state.partial_responses.append(f"Delivering {delivery} costs ${state.price:,.2f}. \n \n {steps}")
        state.pop(HumanIntent.ESTIMATE_PRICE)


# === ending the turn


class Reply(Operator[ConversationState]):
    """
    WHEN: the utterance has been handled and the intent stack is empty, so there is nothing left to do or ask
    THEN: end the turn by saying the partial responses
    """

    def can_perform(self, state: ConversationState, io: IOContainer) -> bool:
        """applies between utterances once the stack is empty"""
        return not state.listening and state.human_utterance is None and not state.intent_stack

    def perform(self, state: ConversationState, io: IOContainer) -> None:
        """says the partial responses"""
        state.say(io)
