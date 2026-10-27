"""offline tests of the logistics support agent's conversation

a fake EnumClassifier stands in for the LLM (intents are keyed by phrase), and a fake describe_facts names the actions
it is given; the human's lines are fed to input() from a script, so the whole perceive-decide-act loop runs deterministically with no API key. When the script runs out,
input() raises EOFError, which the agent treats as quitting.
"""

import re
from collections import deque

import pytest

from logistics_agent import (
    assistant, conductor, fill_placement, fill_route, fill_shipping_request, fill_vehicle, inspect_state,
)
from logistics_agent.communicator import (
    AgentIntent, AskFor, AskForKind, ConfirmChange, HumanIntent, NotUnderstood, Utterance,
)
from logistics_agent.console import AGENT_PREFIX
from logistics_agent.trace import Step
from logistics_agent.conductor import (
    Acknowledge,
    AskQuestion,
    AskToConfirmChange,
    ConfirmShippingRequest,
    INTRODUCTION,
    ConversationState,
    DescribePlan,
    Introduce,
    PlaceVehicle,
    PlanDelivery,
    PopSatisfiedAsk,
    ProposeAddition,
    ProposeClosure,
    ReactToAddToFleetIntent,
    ReactToAskCapabilitiesIntent,
    ReactToConfirmIntent,
    ReactToDenyIntent,
    ReactToEstimatePriceIntent,
    ReactToGreetingIntent,
    ReactToInspectStateIntent,
    ReactToObstructRouteIntent,
    ReactToShippingRequestIntent,
    RecognizeIntent,
    Reply,
    RespondUnknown,
    SayGoodbye,
    VehicleAddition,
)
from logistics_agent.domain import (
    AirRoute, At, City, Location, RouteReport, RouteType, Shipment, ShippingRequest, Truck, TruckRoute, UnloadAction, VehicleType,
)
from logistics_agent.drafts import build, draft_model
from logistics_agent.inspect_state import QuestionType
from logistics_agent.fill_shipping_request import (
    ShippingRequestFiller,
    ShippingRequestState,
)

GIBBERISH = "what is the capital of france?"
HELLO = "hello there"
CAPABILITIES = "what can you do?"
PRICE = "how much will it cost?"
SHIPPING = "I need to ship a box of books, 200 pounds, from Seattle to Denver"
BOOKS_CONFIRMATION = (
    "Got it: the box of books (200 lb) from Seattle to Denver. Ask me what it costs whenever you are ready."
)
BOOKS_PRICE_REPLY = (
    "Delivering the box of books from Seattle to Denver costs $1,310.00.\n\n"
    "1. Big Bertha drives from Seattle Terminal to Seattle ($50.00)\n"
    "2. Load the box of books onto Big Bertha at Seattle ($120.00)\n"
    "3. Big Bertha drives long-distance from Seattle to Denver ($1,040.00)\n"
    "4. Unload the box of books from Big Bertha at Denver ($100.00)"
)
PRICE_REPLY = (
    "Delivering the pallet of office chairs from Seattle to San Francisco costs $1,810.00.\n\n"
    "1. Big Bertha drives from Seattle Terminal to Seattle ($50.00)\n"
    "2. Load the pallet of office chairs onto Big Bertha at Seattle ($480.00)\n"
    "3. Big Bertha drives long-distance from Seattle to San Francisco ($640.00)\n"
    "4. Unload the pallet of office chairs from Big Bertha at San Francisco ($640.00)"
)
QUESTION = "Where is the shipment now?"
# a delivery described in two parts: the books from Seattle, then where they go
BOOKS_FROM_SEATTLE = "200 pounds of books from Seattle"
TO_DENVER = "Denver"
# an answer about a delivery that gives none of it
NOWHERE = "somewhere far away"
# only what is shipped; then where it is picked up from, alone or with where it is going
JUST_BOOKS = "200 pounds of books"
BOSTON = "Boston"
BOSTON_TO_DENVER = "Boston, and it is going to Denver"
# adding a truck to the fleet: what it is called, its details, and where it is; then yes or no
ADD_TRUCK = "I want to add a new truck called Road Runner"
TRUCK_DETAILS = "a Ford F-750 that carries 12000 pounds, plate 7XYZ123"
AT_DENVER = "at the Denver terminal"
ADD_VEHICLE = "I want to add a vehicle to the fleet"
A_TRUCK = "a truck"
YES = "yes, go ahead"
NO = "no"
# reporting a blocked route: snow says it is the road; "blocked" does not say, so the agent asks, and "the road" answers
SNOWED_IN = "the route between Seattle and San Francisco is snowed in"
BLOCKED = "the route between Seattle and Denver is blocked"
THE_ROAD = "the road"
SAME_CITY = "the road between Seattle and Seattle is flooded"
# questions about the state of the world: the trucks, the whole fleet, and one the agent cannot answer
WHICH_TRUCKS = "which trucks do you have?"
WHOLE_FLEET = "what is in your fleet?"
WHO_DRIVES = "who drives Big Bertha?"
REQUEST_QUESTION = AskFor(ShippingRequest).question
DESTINATION_QUESTION = AskFor(ShippingRequest, "destination").question

INTENTS = {
    GIBBERISH: None,
    HELLO: HumanIntent.GREETING,
    CAPABILITIES: HumanIntent.ASK_CAPABILITIES,
    PRICE: HumanIntent.ESTIMATE_PRICE,
    SHIPPING: HumanIntent.SHIPPING_REQUEST,
    BOOKS_FROM_SEATTLE: HumanIntent.SHIPPING_REQUEST,
    TO_DENVER: HumanIntent.SHIPPING_REQUEST,
    NOWHERE: HumanIntent.SHIPPING_REQUEST,
    JUST_BOOKS: HumanIntent.SHIPPING_REQUEST,
    BOSTON: HumanIntent.SHIPPING_REQUEST,
    BOSTON_TO_DENVER: HumanIntent.SHIPPING_REQUEST,
    ADD_TRUCK: HumanIntent.ADD_TO_FLEET,
    TRUCK_DETAILS: HumanIntent.ADD_TO_FLEET,
    AT_DENVER: HumanIntent.ADD_TO_FLEET,
    ADD_VEHICLE: HumanIntent.ADD_TO_FLEET,
    A_TRUCK: HumanIntent.ADD_TO_FLEET,
    YES: HumanIntent.CONFIRM,
    "ok": HumanIntent.CONFIRM,
    NO: HumanIntent.DENY,
    SNOWED_IN: HumanIntent.OBSTRUCT_ROUTE,
    BLOCKED: HumanIntent.OBSTRUCT_ROUTE,
    THE_ROAD: HumanIntent.OBSTRUCT_ROUTE,
    SAME_CITY: HumanIntent.OBSTRUCT_ROUTE,
    WHICH_TRUCKS: HumanIntent.INSPECT_STATE,
    WHOLE_FLEET: HumanIntent.INSPECT_STATE,
    WHO_DRIVES: HumanIntent.INSPECT_STATE,
}


class FakeEnumClassifier:
    """stands in for EnumClassifier: canned intents, keyed by phrase; records how it was built and called"""

    def __init__(self, enum_type, task_desc: str) -> None:
        self.enum_type, self.task_desc = enum_type, task_desc
        self.calls: list[tuple] = []

    def __call__(self, text: str, llm, **kwargs) -> tuple[HumanIntent | None, float]:
        """classifies the phrase, or, for an answer given together with the agent's question, the answer"""
        self.calls.append((text, llm, kwargs))
        answer = re.search(r'the human answered: "(.*)"$', text)
        return INTENTS[answer.group(1) if answer else text], 1.0


class FakeDescribeFacts:
    """stands in for describe_facts: names the types of the facts, in order; records how it was called"""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __call__(self, instances, task_desc: str | None, llm, **kwargs) -> str:
        facts = tuple(instances)
        self.calls.append((facts, task_desc, llm))
        return ", ".join(type(fact).__name__ for fact in facts)


class FakeModelPopulator:
    """stands in for ModelPopulator in the ShippingRequestFiller sub-process: the delivery is a box of books, 200
    pounds, from Seattle to Denver, of which BOOKS_FROM_SEATTLE leaves out the destination and TO_DENVER gives only the
    destination; anything else gives none of it; answers with the parts the model it is asked to fill in has, as the
    language model filling them in together would"""

    BOOKS = {"details": "box of books", "weight": 200}
    # what each phrase gives: the origin's city, the destination's city, and the shipment, or None for a part it does
    # not give
    GIVES = {
        SHIPPING: ("Seattle", "Denver", BOOKS),
        BOOKS_FROM_SEATTLE: ("Seattle", None, BOOKS),
        TO_DENVER: (None, "Denver", None),
        JUST_BOOKS: (None, None, BOOKS),
        BOSTON: ("Boston", None, None),
        BOSTON_TO_DENVER: ("Boston", "Denver", None),
    }

    def __init__(self, model_type, task_desc: str) -> None:
        self.model_type, self.task_desc = model_type, task_desc

    def __call__(self, utterance: str, llm, *extra: str, **kwargs):
        given = dict(zip(("origin", "destination", "shipment"), self.GIVES.get(utterance, (None, None, None))))
        return self.model_type.model_validate({k: v for k, v in given.items() if k in self.model_type.model_fields})


class FakeFleetPopulator:
    """stands in for ModelPopulator in the VehicleFiller and PlacementFiller sub-processes: answers with the fields
    each phrase gives that the model it is asked to fill in has"""

    GIVES = {
        ADD_TRUCK: {"display_name": "Road Runner"},
        TRUCK_DETAILS: {"model": "Ford F-750", "max_payload": 12000, "license_plate": "7XYZ123"},
        AT_DENVER: {"entity2": "Denver Terminal"},
        SNOWED_IN: {"city1": "Seattle", "city2": "San Francisco"},
        BLOCKED: {"city1": "Seattle", "city2": "Denver"},
        SAME_CITY: {"city1": "Seattle", "city2": "Seattle"},
        WHICH_TRUCKS: {"kind": "truck"},
    }

    def __init__(self, model_type, task_desc: str) -> None:
        self.model_type = model_type

    def __call__(self, utterance: str, llm, *extra: str, **kwargs):
        given = self.GIVES.get(utterance, {})
        return self.model_type.model_validate({k: v for k, v in given.items() if k in self.model_type.model_fields})


class FakeKindClassifier:
    """stands in for the EnumClassifier of a kind, of vehicle or of route: the kind each phrase names, when it is one
    of the enum's, and no kind for anything else; an answer is classified together with the question asked"""

    KINDS = {
        ADD_TRUCK: VehicleType.TRUCK, A_TRUCK: VehicleType.TRUCK,
        SNOWED_IN: RouteType.ROAD, THE_ROAD: RouteType.ROAD, SAME_CITY: RouteType.ROAD,
        WHICH_TRUCKS: QuestionType.FLEET, WHOLE_FLEET: QuestionType.FLEET,
    }

    def __init__(self, enum_type, task_desc: str) -> None:
        self.enum_type = enum_type

    def __call__(self, text: str, llm, **kwargs):
        answer = re.search(r'the human answered: "(.*)"$', text)
        kind = self.KINDS.get(answer.group(1) if answer else text)
        return (kind if isinstance(kind, self.enum_type) else None), 1.0


@pytest.fixture(autouse=True)
def fake_language(monkeypatch):
    """replaces the LLM-backed EnumClassifier and describe_facts used by the operators; returns the fake
    describe_facts"""
    monkeypatch.setattr(conductor, "EnumClassifier", FakeEnumClassifier)
    fake_describe_facts = FakeDescribeFacts()
    monkeypatch.setattr(conductor, "describe_facts", fake_describe_facts)
    monkeypatch.setattr(fill_shipping_request, "ModelPopulator", FakeModelPopulator)
    monkeypatch.setattr(fill_vehicle, "ModelPopulator", FakeFleetPopulator)
    monkeypatch.setattr(fill_vehicle, "EnumClassifier", FakeKindClassifier)
    monkeypatch.setattr(fill_route, "EnumClassifier", FakeKindClassifier)
    monkeypatch.setattr(fill_route, "ModelPopulator", FakeFleetPopulator)
    monkeypatch.setattr(inspect_state, "EnumClassifier", FakeKindClassifier)
    monkeypatch.setattr(inspect_state, "ModelPopulator", FakeFleetPopulator)
    monkeypatch.setattr(fill_placement, "ModelPopulator", FakeFleetPopulator)
    return fake_describe_facts


@pytest.fixture
def chat(monkeypatch, capsys):
    """runs the agent through a script of human lines; returns the agent's replies, in order"""

    def run(*lines: str, args: tuple[str, ...] = ()) -> list[str]:
        script = deque(lines)

        def scripted_input(_prompt: str = "") -> str:
            if not script:
                raise EOFError
            return script.popleft()

        monkeypatch.setattr("builtins.input", scripted_input)
        assistant.main(list(args))
        out = capsys.readouterr().out
        replies = out.split(AGENT_PREFIX)[1:]
        return [reply.strip() for reply in replies]

    return run


# === whole conversations


def test_introduces_itself_first(chat):
    assert chat("quit")[0] == INTRODUCTION


@pytest.mark.parametrize("phrase", ["quit", "exit", "bye", "goodbye", "QUIT", "ok bye now", "Exit!", "I want to quit"])
def test_says_goodbye_on_anything_containing_a_quit_phrase(chat, phrase):
    assert chat(phrase) == [INTRODUCTION, "Goodbye!"]


def test_end_of_input_counts_as_quitting(chat):
    assert chat() == [INTRODUCTION, "Goodbye!"]


def test_unknown_intent_gets_a_polite_refusal(chat):
    assert chat(GIBBERISH, "quit") == [INTRODUCTION, "Sorry, I am not designed to respond to that.", "Goodbye!"]


def test_keeps_answering_after_each_reply(chat):
    replies = chat(GIBBERISH, HELLO, GIBBERISH, HELLO, "quit")
    assert replies[1] == replies[3] == "Sorry, I am not designed to respond to that."
    assert replies[2] == replies[4] == ReactToGreetingIntent.REPLY
    assert replies[-1] == "Goodbye!"


def test_empty_lines_are_ignored(chat):
    assert chat("", "   ", GIBBERISH, "quit") == [
        INTRODUCTION, "Sorry, I am not designed to respond to that.", "Goodbye!"
    ]


@pytest.mark.parametrize("phrase, reply", [
    (HELLO, ReactToGreetingIntent.REPLY), (CAPABILITIES, ReactToAskCapabilitiesIntent.REPLY),
])
def test_greetings_and_capability_questions_are_answered_and_the_conversation_goes_on(chat, phrase, reply):
    replies = chat(phrase, GIBBERISH, "quit")
    assert replies[1] == reply
    assert replies[2] == "Sorry, I am not designed to respond to that."


def test_asking_for_a_price_before_describing_the_delivery_asks_for_the_delivery(chat):
    assert chat(PRICE, "quit") == [INTRODUCTION, REQUEST_QUESTION, "Goodbye!"]


def test_a_price_asked_first_is_answered_once_the_delivery_and_its_missing_destination_are_given(chat):
    assert chat(PRICE, BOOKS_FROM_SEATTLE, TO_DENVER, "quit") == [
        INTRODUCTION,
        REQUEST_QUESTION,
        DESTINATION_QUESTION,
        f"{BOOKS_CONFIRMATION.removesuffix(' Ask me what it costs whenever you are ready.')} {BOOKS_PRICE_REPLY}",
        "Goodbye!",
    ]


def test_a_delivery_without_its_destination_is_confirmed_once_the_destination_is_given(chat):
    assert chat(BOOKS_FROM_SEATTLE, TO_DENVER, "quit") == [
        INTRODUCTION, DESTINATION_QUESTION, BOOKS_CONFIRMATION, "Goodbye!",
    ]


def test_an_answer_that_gives_nothing_gets_the_question_asked_again(chat):
    assert chat(BOOKS_FROM_SEATTLE, NOWHERE, "quit")[2] == DESTINATION_QUESTION


def test_an_answer_to_the_question_for_one_city_fills_in_only_that_city(chat):
    assert chat(JUST_BOOKS, BOSTON, TO_DENVER, "quit") == [
        INTRODUCTION,
        AskFor(ShippingRequest, "origin").question,
        DESTINATION_QUESTION,
        "Got it: the box of books (200 lb) from Boston to Denver. Ask me what it costs whenever you are ready.",
        "Goodbye!",
    ]


def test_an_answer_that_also_gives_another_part_fills_in_both(chat):
    assert chat(JUST_BOOKS, BOSTON_TO_DENVER, "quit") == [
        INTRODUCTION,
        AskFor(ShippingRequest, "origin").question,
        "Got it: the box of books (200 lb) from Boston to Denver. Ask me what it costs whenever you are ready.",
        "Goodbye!",
    ]


ROAD_RUNNER_QUESTION = (
    "Shall I add Road Runner, a Ford F-750 truck that carries up to 12,000 lb, at Denver Terminal, to the fleet?"
)
ROAD_RUNNER_ADDED = "Added Road Runner to the fleet, at Denver Terminal."


@pytest.mark.parametrize("yes", [YES, "ok"])
def test_a_truck_is_added_to_the_fleet_once_it_is_described_placed_and_confirmed(chat, yes):
    assert chat(ADD_TRUCK, TRUCK_DETAILS, AT_DENVER, yes, "quit") == [
        INTRODUCTION,
        Truck.model_fields["model"].description,
        At.model_fields["entity2"].description,
        ROAD_RUNNER_QUESTION,
        ROAD_RUNNER_ADDED,
        "Goodbye!",
    ]


def test_the_kind_of_vehicle_is_asked_for_when_the_human_does_not_say(chat):
    assert chat(ADD_VEHICLE, A_TRUCK, "quit")[1:3] == [
        AskForKind(VehicleType).question, Truck.model_fields["display_name"].description,
    ]


def test_a_truck_the_human_says_no_to_is_not_added(chat):
    assert chat(ADD_TRUCK, TRUCK_DETAILS, AT_DENVER, NO, "quit")[4] == "OK, I did not add Road Runner to the fleet."


def test_small_talk_before_confirming_is_answered_and_the_confirmation_asked_again(chat):
    assert chat(ADD_TRUCK, TRUCK_DETAILS, AT_DENVER, HELLO, "quit")[4] == (
        f"{ReactToGreetingIntent.REPLY} {ROAD_RUNNER_QUESTION}"
    )


SEATTLE_SF_ROAD = "the road between Seattle and San Francisco"


def test_a_snowed_in_road_is_closed_once_confirmed(chat):
    assert chat(SNOWED_IN, YES, "quit") == [
        INTRODUCTION,
        f"Shall I close {SEATTLE_SF_ROAD}?",
        f"Closed {SEATTLE_SF_ROAD}; I will plan deliveries around it.",
        "Goodbye!",
    ]


def test_the_kind_of_route_is_asked_for_when_the_cause_does_not_tell(chat):
    assert chat(BLOCKED, THE_ROAD, "quit")[1:3] == [
        RouteType.__doc__, "Shall I close the road between Seattle and Denver?",
    ]


def test_a_road_the_human_says_no_to_is_left_open(chat):
    assert chat(SNOWED_IN, NO, "quit")[2] == f"OK, I left {SEATTLE_SF_ROAD} open."


def test_a_route_the_world_graph_does_not_have_is_not_closed(chat):
    assert chat(SAME_CITY, "quit")[1] == (
        "I don't know of the road between Seattle and Seattle, so there is nothing to close."
    )


BERTHA = "Big Bertha (Freightliner M2 106 26ft box truck), a truck that carries up to 10,000 lb, at Seattle Terminal"
ROCKET = "Little Rocket (Isuzu NPR-HD 16ft box truck), a truck that carries up to 6,000 lb, at San Francisco Terminal"
PELICAN = (
    "The Pelican (Cessna 408 SkyCourier freighter), an airplane that carries up to 6,000 lb, at Seattle-Tacoma "
    "International Airport"
)


def test_a_question_about_the_trucks_is_answered_from_the_world_graph(chat):
    assert chat(WHICH_TRUCKS, "quit")[1] == f"The fleet has 2 trucks: {BERTHA}; {ROCKET}."


def test_a_question_about_the_whole_fleet_lists_every_vehicle(chat):
    assert chat(WHOLE_FLEET, "quit")[1] == f"The fleet has 3 vehicles: {PELICAN}; {BERTHA}; {ROCKET}."


def test_a_truck_added_to_the_fleet_is_in_the_next_answer(chat):
    road_runner = "Road Runner (Ford F-750), a truck that carries up to 12,000 lb, at Denver Terminal"
    assert chat(ADD_TRUCK, TRUCK_DETAILS, AT_DENVER, YES, WHICH_TRUCKS, "quit")[5] == (
        f"The fleet has 3 trucks: {BERTHA}; {ROCKET}; {road_runner}."
    )


def test_a_question_about_the_state_the_agent_cannot_answer_is_said_so(chat):
    assert chat(WHO_DRIVES, "quit")[1] == ReactToInspectStateIntent.CANNOT_ANSWER


def test_small_talk_while_the_agent_is_asking_is_answered_and_the_question_asked_again(chat):
    assert chat(BOOKS_FROM_SEATTLE, HELLO, TO_DENVER, "quit") == [
        INTRODUCTION,
        DESTINATION_QUESTION,
        f"{ReactToGreetingIntent.REPLY} {DESTINATION_QUESTION}",
        BOOKS_CONFIRMATION,
        "Goodbye!",
    ]


def test_a_described_delivery_is_confirmed_and_then_priced(chat):
    assert chat(SHIPPING, PRICE, "quit") == [INTRODUCTION, BOOKS_CONFIRMATION, BOOKS_PRICE_REPLY, "Goodbye!"]


def test_with_trace_the_steps_of_the_reasoning_are_shown_among_the_replies(chat):
    replies = chat(SHIPPING, "quit", args=("--trace",))
    shown = "\n".join(replies)
    for step in ("· conductor: recognize_intent", "· conductor: react_to_shipping_request_intent",
                 "· fill_shipping_request: extract_parts",
                 "· fill_shipping_request: compose_asks",
                 "· conductor: reply", "· conductor: say_goodbye"):
        assert step in shown
    assert BOOKS_CONFIRMATION in shown


class RecordingConsole:
    """stands in for the console, recording the steps and progress shown"""

    def __init__(self) -> None:
        self.shown: list[tuple] = []

    def show_step(self, depth: int, source: str, action: str, kind: str = "decision") -> None:
        self.shown.append(("step", depth, source, action, kind))

    def show_progress(self, text: str) -> None:
        self.shown.append(("progress", text))


def test_the_planners_search_steps_are_counted_on_the_spinner_and_other_steps_shown_one_by_one():
    console = RecordingConsole()
    report = assistant.StepReporter(console)
    report(Step(0, "decision", "DecisionProcess", "react_to_estimate_price_intent"))
    report(Step(2, "decision", "BaseDecisionProcess", "search"))
    report(Step(2, "decision", "BaseDecisionProcess", "search"))
    report(Step(1, "planner", "planner", "search for the cheapest plan: 2 search steps, 1 states explored"))
    report(Step(0, "decision", "DecisionProcess", "describe_plan"))
    report(Step(1, "llm", "llm", "describe facts"))
    report(Step(2, "decision", "BaseDecisionProcess", "search"))
    assert console.shown == [
        ("step", 0, "conductor", "react_to_estimate_price_intent", "decision"),
        ("progress", "planning… 1 search steps"),
        ("progress", "planning… 2 search steps"),
        ("step", 1, "planner", "search for the cheapest plan: 2 search steps, 1 states explored", "planner"),
        ("step", 0, "conductor", "describe_plan", "decision"),
        ("step", 1, "llm", "describe facts", "llm"),
        ("progress", "planning… 1 search steps"),
    ]


def test_with_trace_the_planner_is_shown_but_not_the_search_step_by_step(chat):
    shown = "\n".join(chat(SHIPPING, PRICE, "quit", args=("--trace",)))
    assert re.search(
        r"· conductor: plan_delivery\n    · planner: search for the cheapest plan: \d+ search steps, "
        r"\d+ states explored\n",
        shown,
    )
    assert "BaseDecisionProcess" not in shown


def test_ok_gets_no_reply_and_thanks_is_answered(chat):
    assert chat("ok", "thanks!", "quit") == [INTRODUCTION, "You're welcome.", "Goodbye!"]


def test_every_intent_has_an_operator_that_handles_it():
    assert {intent for intent in HANDLERS.values() if isinstance(intent, HumanIntent)} == set(HumanIntent)


# === single operators


class RecordingIO:
    """stands in for the decision process's io container: io.i.interaction is the heard utterance,
    io.o.interaction records what the agent says"""

    def __init__(self, phrase: str | None = None) -> None:
        self.said: list[str] = []
        outer = self

        class I:
            interaction = Utterance(phrase=phrase)

        class O:
            @staticmethod
            def interaction(u: Utterance) -> None:
                outer.said.append(u.phrase)

        self.i, self.o = I, O


def heard(phrase: str, state: ConversationState | None = None) -> ConversationState:
    """a conversation state in which the phrase has been heard, and its intent, or NotUnderstood, pushed, as
    RecognizeIntent leaves it"""
    state = state or ConversationState()
    state.listening = False
    state.human_utterance = Utterance(phrase)
    state.push(NotUnderstood() if INTENTS[phrase] is None else INTENTS[phrase])
    return state


def working_on(*stack: HumanIntent | AgentIntent, state: ConversationState | None = None) -> ConversationState:
    """a conversation state between utterances, with the intents on the stack (the last on top)"""
    state = state or ConversationState()
    state.listening = False
    state.intent_stack = list(stack)
    return state


def react(op_type: type, phrase: str, state: ConversationState | None = None) -> tuple[ConversationState, str, object]:
    """has a fresh operator of the type handle the phrase; checks that it applies, that it pops the intent the phrase
    pushed, and that it notes one thing instead of speaking; returns the state, the note, and the operator"""
    state = heard(phrase, state)
    pushed = state.top
    op, io = op_type("op"), RecordingIO()
    assert op.can_perform(state, io)
    responses_before = len(state.partial_responses)
    op.perform(state, io)
    assert pushed not in state.intent_stack
    assert not io.said, "operators that handle an utterance note what to say, and the turn says it at the end"
    [note] = state.partial_responses[responses_before:]
    return state, note, op


GREETING = HumanIntent.GREETING
CAPABILITY = HumanIntent.ASK_CAPABILITIES
PRICE_INTENT = HumanIntent.ESTIMATE_PRICE
SHIPPING_INTENT = HumanIntent.SHIPPING_REQUEST
ASK_REQUEST = AskFor(ShippingRequest)
ASK_ORIGIN = AskFor(ShippingRequest, "origin")
ASK_DESTINATION = AskFor(ShippingRequest, "destination")
ASK_SHIPMENT = AskFor(ShippingRequest, "shipment")
ASK_KIND = AskForKind(VehicleType)
ASK_TRUCK = AskFor(Truck)
ASK_TRUCK_PLATE = AskFor(Truck, "license_plate")
ASK_PLACEMENT = AskFor(At)
ASK_LOCATION = AskFor(At, "entity2")
ASK_ROUTE = AskFor(RouteReport)
ASK_ROUTE_CITY2 = AskFor(RouteReport, "city2")
ASK_ROUTE_KIND = AskForKind(RouteType)
ROAD_RUNNER = Truck(name="truck3", display_name="Road Runner", model="Ford F-750", max_payload=12000,
                    license_plate="7XYZ123")


# --- conversation state and the intent stack


def test_planner_plans_over_the_reasoners_world_graph():
    state = ConversationState()
    assert state.planner.world is state.reasoner.world


def test_conversation_starts_listening_with_an_empty_stack():
    state = ConversationState()
    assert state.listening and state.intent_stack == [] and state.top is None and state.partial_responses == []
    assert state.planner.goal is None and state.planner.current is None


def test_push_puts_an_intent_on_top_and_moves_it_there_if_it_is_already_on_the_stack():
    state = ConversationState()
    for intent in (GREETING, CAPABILITY):
        state.push(intent)
    assert state.intent_stack == [GREETING, CAPABILITY] and state.top is CAPABILITY
    state.push(GREETING)
    assert state.intent_stack == [CAPABILITY, GREETING]


def test_pop_removes_only_the_intent_on_top():
    state = working_on(GREETING, CAPABILITY)
    with pytest.raises(AssertionError):
        state.pop(GREETING)
    state.pop(CAPABILITY)
    assert state.intent_stack == [GREETING]


def test_the_stack_holds_the_humans_intents_and_the_agents_asks_and_an_equal_ask_once():
    state = working_on(PRICE_INTENT, ASK_REQUEST)
    state.push(ASK_DESTINATION)
    state.push(AskFor(ShippingRequest))
    assert state.intent_stack == [PRICE_INTENT, ASK_DESTINATION, ASK_REQUEST]
    state.pop(AskFor(ShippingRequest))
    assert state.top == ASK_DESTINATION


def test_the_agent_is_asking_for_a_model_while_an_ask_for_it_or_its_attributes_is_on_the_stack():
    assert not working_on(PRICE_INTENT).asking_for(ShippingRequest)
    assert working_on(ASK_REQUEST, GREETING).asking_for(ShippingRequest)
    assert working_on(ASK_ORIGIN).asking_for(ShippingRequest)


def test_the_asks_for_a_shipping_request_are_satisfied_by_the_one_being_filled_in():
    state = with_chairs(ConversationState())
    assert state.model_of(ShippingRequest) is state.shipping_request


def test_planner_can_plan_once_it_has_current_facts_and_a_goal():
    state = ConversationState()
    find = state.reasoner.find_entity
    assert not state.planner.has_current() and not state.planner.has_goal() and not state.planner.can_plan()
    state.planner.set_current([At(entity1=find("shipment1"), entity2=find("terminal_seattle"))])
    assert state.planner.has_current() and not state.planner.has_goal() and not state.planner.can_plan()
    state.planner.set_goal([At(entity1=find("shipment1"), entity2=find("terminal_san_francisco"))])
    assert state.planner.has_goal() and state.planner.can_plan()


# --- exactly one operator applies at every step


ALL_OPERATORS = [
    Introduce, SayGoodbye, Acknowledge, RecognizeIntent, RespondUnknown,
    ReactToGreetingIntent, ReactToAskCapabilitiesIntent, ReactToShippingRequestIntent, ConfirmShippingRequest,
    PopSatisfiedAsk, AskQuestion, ReactToEstimatePriceIntent, PlanDelivery, DescribePlan, Reply,
    ReactToAddToFleetIntent, PlaceVehicle, ProposeAddition, ReactToObstructRouteIntent, ProposeClosure,
    ReactToInspectStateIntent, AskToConfirmChange, ReactToConfirmIntent, ReactToDenyIntent,
]


def introduced(state: ConversationState) -> ConversationState:
    state.introduced = True
    return state


def planned(state: ConversationState) -> ConversationState:
    """the state, with the delivery planned for the price question"""
    state.planned = True
    return state


@pytest.mark.parametrize("make_state, phrase, expected", [
    (ConversationState, None, Introduce),
    (lambda: introduced(ConversationState()), "ok bye", SayGoodbye),
    (lambda: introduced(ConversationState()), "thanks!", Acknowledge),
    (lambda: introduced(ConversationState()), HELLO, RecognizeIntent),
    (lambda: introduced(heard(GIBBERISH)), GIBBERISH, RespondUnknown),
    (lambda: introduced(heard(HELLO)), HELLO, ReactToGreetingIntent),
    (lambda: introduced(heard(CAPABILITIES)), CAPABILITIES, ReactToAskCapabilitiesIntent),
    (lambda: introduced(heard(SHIPPING)), SHIPPING, ReactToShippingRequestIntent),
    (lambda: introduced(heard(PRICE)), PRICE, ReactToEstimatePriceIntent),
    (lambda: introduced(heard(PRICE, with_chairs(ConversationState()))), PRICE, PlanDelivery),
    (lambda: introduced(planned(working_on(PRICE_INTENT, state=with_chairs(ConversationState())))), PRICE,
     DescribePlan),
    (lambda: introduced(working_on(PRICE_INTENT, ASK_REQUEST)), PRICE, AskQuestion),
    (lambda: introduced(working_on(PRICE_INTENT, ASK_REQUEST, state=with_chairs(ConversationState()))), PRICE,
     ConfirmShippingRequest),
    (lambda: introduced(working_on(ASK_REQUEST, ASK_ORIGIN, state=with_chairs(ConversationState()))), PRICE,
     PopSatisfiedAsk),
    (lambda: introduced(working_on()), HELLO, Reply),
])
def test_exactly_one_operator_applies_at_each_step(make_state, phrase, expected):
    state, io = make_state(), RecordingIO(phrase)
    assert [op_type for op_type in ALL_OPERATORS if op_type("op").can_perform(state, io)] == [expected]


# --- introduce and say goodbye


def test_introduce_applies_only_once():
    state, io = ConversationState(), RecordingIO()
    op = Introduce("introduce")
    assert op.can_perform(state, io)
    op.perform(state, io)
    assert io.said == [INTRODUCTION] and state.introduced and state.listening
    assert not op.can_perform(state, io)


@pytest.mark.parametrize("phrase", [None, "exiting the highway", "byeee", GIBBERISH])
def test_say_goodbye_ignores_phrases_without_a_quit_word(phrase):
    assert not SayGoodbye("say_goodbye").can_perform(ConversationState(), RecordingIO(phrase))


def test_say_goodbye_ends_the_conversation():
    state, io = ConversationState(), RecordingIO("ok bye")
    op = SayGoodbye("say_goodbye")
    assert op.can_perform(state, io)
    op.perform(state, io)
    assert io.said == ["Goodbye!"] and state.done


# --- acknowledge


@pytest.mark.parametrize("phrase", ["thanks", "Thank you so much!", "thx"])
def test_acknowledge_answers_thanks_and_keeps_listening(phrase):
    state, io = introduced(ConversationState()), RecordingIO(phrase)
    op = Acknowledge("acknowledge")
    assert op.can_perform(state, io)
    op.perform(state, io)
    assert io.said == ["You're welcome."] and state.listening and state.asked is None and state.intent_stack == []


@pytest.mark.parametrize("phrase", ["ok", "OK.", "got it", "sounds good!"])
def test_acknowledge_says_nothing_to_ok_and_keeps_the_question_pending(phrase):
    state, io = introduced(ConversationState()), RecordingIO(phrase)
    state.asked = QUESTION
    op = Acknowledge("acknowledge")
    assert op.can_perform(state, io)
    op.perform(state, io)
    assert io.said == [] and state.listening and state.asked == QUESTION


def test_acknowledge_asks_again_the_question_the_agent_last_asked_after_thanks():
    state, io = introduced(ConversationState()), RecordingIO("thanks")
    state.asked = QUESTION
    Acknowledge("acknowledge").perform(state, io)
    assert io.said == [f"You're welcome. {QUESTION}"] and state.asked == QUESTION


@pytest.mark.parametrize("phrase", [None, "ok, ship it from seattle", "thanks for the quote", "ok bye", HELLO])
def test_acknowledge_ignores_anything_that_says_more(phrase):
    assert not Acknowledge("acknowledge").can_perform(ConversationState(), RecordingIO(phrase))


def test_acknowledge_waits_until_the_agent_is_listening():
    assert not Acknowledge("acknowledge").can_perform(working_on(), RecordingIO("thanks"))


@pytest.mark.parametrize(("ask", "intent"), [
    (ASK_KIND, HumanIntent.ADD_TO_FLEET), (ASK_TRUCK, HumanIntent.ADD_TO_FLEET),
    (ASK_TRUCK_PLATE, HumanIntent.ADD_TO_FLEET), (ASK_PLACEMENT, HumanIntent.ADD_TO_FLEET),
    (ASK_LOCATION, HumanIntent.ADD_TO_FLEET), (ASK_ROUTE, HumanIntent.OBSTRUCT_ROUTE),
    (ASK_ROUTE_CITY2, HumanIntent.OBSTRUCT_ROUTE), (ASK_ROUTE_KIND, HumanIntent.OBSTRUCT_ROUTE),
    (ASK_ORIGIN, HumanIntent.SHIPPING_REQUEST), (ASK_DESTINATION, HumanIntent.SHIPPING_REQUEST),
])
def test_asked_about_is_what_the_human_wants_that_the_ask_on_top_is_part_of(ask, intent):
    assert working_on(GREETING, ask).asked_about() is intent


@pytest.mark.parametrize("stack", [(), (GREETING,), (ConfirmChange(),)])
def test_asked_about_is_nothing_without_an_ask_on_top(stack):
    assert working_on(*stack).asked_about() is None


# --- recognize intent


def test_recognize_intent_stores_the_utterance_stops_listening_and_pushes_the_intent():
    state, io = ConversationState(), RecordingIO(HELLO)
    op = RecognizeIntent("recognize_intent")
    assert op.can_perform(state, io)
    op.perform(state, io)
    assert state.human_utterance == Utterance(HELLO)
    assert state.intent_stack == [GREETING] and not state.listening
    assert not op.can_perform(state, io)


def test_recognize_intent_pushes_not_understood_for_an_unknown_intent():
    state = working_on(ASK_DESTINATION)
    state.listening = True
    RecognizeIntent("recognize_intent").perform(state, RecordingIO(GIBBERISH))
    assert state.human_utterance == Utterance(GIBBERISH)
    assert state.intent_stack == [ASK_DESTINATION, NotUnderstood()]


def test_recognize_intent_classifies_over_the_known_intents():
    op = RecognizeIntent("recognize_intent")
    op.perform(ConversationState(), RecordingIO(GIBBERISH))
    classifier = op._intent_classifier
    assert classifier.enum_type is HumanIntent
    assert classifier.task_desc == RecognizeIntent.TASK_DESC
    assert classifier.calls == [(GIBBERISH, conductor.DEFAULT_LLM, {"num_trials": 1})]


def test_recognize_intent_classifies_an_answer_together_with_the_question_asked():
    state = ConversationState()
    state.asked = QUESTION
    op = RecognizeIntent("recognize_intent")
    op.perform(state, RecordingIO(HELLO))
    [(text, _, _)] = op._intent_classifier.calls
    assert text == f'the agent asked: "{QUESTION}" and the human answered: "{HELLO}"'


def test_recognize_intent_classifies_an_answer_together_with_what_the_question_is_part_of():
    """after "Where is it now?" about a truck being added, "Boston terminal" is read as part of adding the truck, not
    as a delivery"""
    state = working_on(ASK_LOCATION)
    state.listen(ASK_LOCATION.question)
    op = RecognizeIntent("recognize_intent")
    op.perform(state, RecordingIO(AT_DENVER))
    [(text, _, _)] = op._intent_classifier.calls
    assert text == (
        f"while helping with what the human wants ({HumanIntent.ADD_TO_FLEET.__doc__}), the agent asked: "
        f'"{ASK_LOCATION.question}" and the human answered: "{AT_DENVER}"'
    )


@pytest.mark.parametrize("phrase", [None, "quit", "ok bye now", "EXIT please", "thanks", "ok"])
def test_recognize_intent_skips_silence_quitting_and_acknowledgements(phrase):
    op = RecognizeIntent("recognize_intent")
    assert not op.can_perform(ConversationState(), RecordingIO(phrase))


def test_recognize_intent_waits_until_the_agent_is_listening():
    op = RecognizeIntent("recognize_intent")
    assert not op.can_perform(heard(HELLO), RecordingIO(GIBBERISH))
    assert not op.can_perform(working_on(), RecordingIO(GIBBERISH))


# --- the top of the stack decides which operator applies


# the operator that handles each intent recognized in an utterance, when it is on top
HANDLERS = {
    ReactToGreetingIntent: HumanIntent.GREETING,
    ReactToAskCapabilitiesIntent: HumanIntent.ASK_CAPABILITIES,
    ReactToEstimatePriceIntent: HumanIntent.ESTIMATE_PRICE,
    ReactToShippingRequestIntent: HumanIntent.SHIPPING_REQUEST,
    ReactToAddToFleetIntent: HumanIntent.ADD_TO_FLEET,
    ReactToConfirmIntent: HumanIntent.CONFIRM,
    ReactToDenyIntent: HumanIntent.DENY,
    ReactToObstructRouteIntent: HumanIntent.OBSTRUCT_ROUTE,
    ReactToInspectStateIntent: HumanIntent.INSPECT_STATE,
    RespondUnknown: NotUnderstood(),
}
WORK_OPERATORS = [
    RespondUnknown, ReactToGreetingIntent, ReactToAskCapabilitiesIntent, ReactToShippingRequestIntent,
    ConfirmShippingRequest, PopSatisfiedAsk, AskQuestion, ReactToEstimatePriceIntent, PlanDelivery, DescribePlan, Reply,
    ReactToAddToFleetIntent, PlaceVehicle, ProposeAddition, ReactToObstructRouteIntent, ProposeClosure,
    ReactToInspectStateIntent, AskToConfirmChange, ReactToConfirmIntent, ReactToDenyIntent,
]


@pytest.mark.parametrize("op_type", list(HANDLERS))
@pytest.mark.parametrize("intent", [*HumanIntent, NotUnderstood(), ASK_REQUEST, ASK_DESTINATION])
def test_each_handler_applies_only_with_its_intent_on_top(op_type, intent):
    state = working_on(intent, state=ConversationState(human_utterance=Utterance(GIBBERISH)))
    assert op_type("op").can_perform(state, RecordingIO()) == (intent == HANDLERS[op_type])


@pytest.mark.parametrize("op_type", list(HANDLERS))
def test_each_handler_looks_only_at_the_top(op_type):
    state = working_on(HANDLERS[op_type], GREETING if op_type is not ReactToGreetingIntent else CAPABILITY)
    assert not op_type("op").can_perform(state, RecordingIO())


@pytest.mark.parametrize("op_type", WORK_OPERATORS)
def test_nothing_but_recognizing_the_intent_applies_while_the_agent_is_listening(op_type):
    state = with_chairs(working_on(HANDLERS.get(op_type, GREETING)))
    state.listening = True
    assert not op_type("op").can_perform(state, RecordingIO())


TOPS = [
    *HumanIntent, NotUnderstood(), ASK_REQUEST, ASK_ORIGIN, ASK_DESTINATION, ASK_SHIPMENT,
    ASK_KIND, ASK_TRUCK, ASK_TRUCK_PLATE, ASK_PLACEMENT, ASK_LOCATION,
    ASK_ROUTE, ASK_ROUTE_CITY2, ASK_ROUTE_KIND, ConfirmChange(), None,
]
REQUESTS = [None, ShippingRequest(), "books from seattle", "chairs"]
PROGRESS = ["nothing", "partial truck", "whole truck", "placed truck", "partial route", "whole route"]


def adding(state: ConversationState, progress: str) -> ConversationState:
    """the state, with a change to the world under way as far as the progress says: a truck being added to the fleet,
    as a draft with only its name and what it is called, the whole truck with no place yet, or the truck and where it
    is, waiting to be confirmed; or a route reported blocked, with one city known, or both and its kind"""
    if progress == "partial truck":
        state.vehicle_draft = draft_model(Truck)(name="truck3", display_name="Road Runner")
    elif progress in ("whole truck", "placed truck"):
        state.vehicle_draft = draft_model(Truck)(**dict(ROAD_RUNNER))
        state.placement_draft = draft_model(At)(entity1=ROAD_RUNNER)
        if progress == "placed truck":
            state.placement_draft.entity2 = state.reasoner.find_entity("terminal_denver")
            state.pending_change = VehicleAddition(build(state.placement_draft))
    elif progress == "partial route":
        state.route_report = RouteReport(city1=state.reasoner.find_entity("seattle"))
    elif progress == "whole route":
        find = state.reasoner.find_entity
        state.route_report = RouteReport(city1=find("seattle"), city2=find("san_francisco"))
        state.route_type = RouteType.ROAD
    return state


@pytest.mark.parametrize("top", TOPS)
@pytest.mark.parametrize("request_", REQUESTS)
@pytest.mark.parametrize("planned_", [False, True])
@pytest.mark.parametrize("progress", PROGRESS)
def test_whatever_is_on_top_exactly_one_operator_applies(top, request_, planned_, progress):
    state = working_on(*([] if top is None else [top]), state=ConversationState(human_utterance=Utterance(SHIPPING)))
    adding(state, progress)
    if request_ == "books from seattle":
        state.shipping_request = ShippingRequest(
            origin=Location(name="shipping-origin", city=state.reasoner.find_entity("seattle")),
            shipment=Shipment(name="shipment", details="box of books", weight=200),
        )
    elif request_ == "chairs":
        with_chairs(state)
    else:
        state.shipping_request = request_
    # a delivery is planned only for a complete request
    state.planned = planned_ and request_ == "chairs"
    assert len([op_type for op_type in WORK_OPERATORS if op_type("op").can_perform(state, RecordingIO())]) == 1


@pytest.mark.parametrize("op_type, phrase", [
    (ReactToGreetingIntent, HELLO), (ReactToAskCapabilitiesIntent, CAPABILITIES),
])
def test_small_talk_pops_its_intent_and_leaves_the_rest_of_the_stack(op_type, phrase):
    other = CAPABILITY if op_type is ReactToGreetingIntent else GREETING
    state, _, _ = react(op_type, phrase, working_on(other))
    assert state.intent_stack == [other]


# --- respond unknown


def test_respond_unknown_notes_an_apology_pops_not_understood_and_leaves_the_rest_of_the_stack():
    state, note, _ = react(RespondUnknown, GIBBERISH, working_on(GREETING))
    assert note == RespondUnknown.REPLY == "Sorry, I am not designed to respond to that."
    assert state.intent_stack == [GREETING]


def with_chairs(state: ConversationState) -> ConversationState:
    """the state, with a pallet of office chairs to deliver from Seattle to San Francisco as its shipping request"""
    find = state.reasoner.find_entity
    state.shipping_request = ShippingRequest(
        origin=Location(name="shipping-origin", city=find("seattle")),
        destination=Location(name="shipping-destination", city=find("san_francisco")),
        shipment=Shipment(name="shipment", details="pallet of office chairs", weight=800),
    )
    return state


# --- taking shipping requests


SEATTLE = Location(name="pickup1", city=City(name="seattle", display_name="Seattle"))
DENVER = Location(name="dropoff1", city=City(name="denver", display_name="Denver"))
BOOKS = Shipment(name="shipment2", details="box of books", weight=200)
BOOKS_REQUEST = ShippingRequest(origin=SEATTLE, destination=DENVER, shipment=BOOKS)
BOOKS_FROM_SEATTLE_REQUEST = ShippingRequest(origin=SEATTLE, shipment=BOOKS)


class FakeShippingRequestFiller:
    """stands in for the ShippingRequestFiller sub-process: returns a canned final state, or None as if it failed;
    records how often it was restarted, and what it was called with"""

    result: ShippingRequestState | None = None

    def __init__(self) -> None:
        self.reinits = 0
        self.calls: list[dict] = []

    def reinit(self) -> "FakeShippingRequestFiller":
        self.reinits += 1
        return self

    def __call__(self, **args) -> ShippingRequestState | None:
        self.calls.append(args)
        return self.result


@pytest.fixture
def fake_filler(monkeypatch):
    """replaces the ShippingRequestFiller sub-process the operator sets up; returns its class, whose result the test
    sets"""
    FakeShippingRequestFiller.result = None
    monkeypatch.setattr(conductor, "ShippingRequestFiller", FakeShippingRequestFiller)
    return FakeShippingRequestFiller


def filled(request: ShippingRequest, *asks: AskFor) -> ShippingRequestState:
    """the sub-process's final state: the request, and the asks for what is missing from it"""
    return ShippingRequestState(request=request, asks=list(asks))


def take(state: ConversationState) -> ConversationState:
    """has a fresh ReactToShippingRequestIntent handle the delivery; checks that it applies, that it pops its intent,
    and that it neither notes nor says anything, leaving that to the asks; returns the state"""
    op, io = ReactToShippingRequestIntent("op"), RecordingIO(SHIPPING)
    assert op.can_perform(state, io)
    op.perform(state, io)
    assert SHIPPING_INTENT not in state.intent_stack and io.said == [] and state.partial_responses == []
    return state


def test_shipping_request_sets_up_its_sub_process_once():
    assert isinstance(ReactToShippingRequestIntent("op")._shipping_request_filler, ShippingRequestFiller)


def test_shipping_request_runs_a_fresh_sub_process_over_a_new_request_when_the_agent_is_not_asking_for_one(
    fake_filler
):
    fake_filler.result = filled(BOOKS_REQUEST)
    state = heard(SHIPPING, with_chairs(ConversationState()))
    state.asked = QUESTION
    op = ReactToShippingRequestIntent("op")
    op.perform(state, RecordingIO(SHIPPING))
    sub_process = op._shipping_request_filler
    assert sub_process.reinits == 1
    assert sub_process.calls == [{
        "utterance": Utterance(SHIPPING), "answering": None, "reasoner": state.reasoner, "request": ShippingRequest(),
        "suppress_errors": False,
    }]


@pytest.mark.parametrize("ask", [ASK_REQUEST, ASK_DESTINATION])
def test_shipping_request_fills_in_the_request_being_asked_for_answering_the_ask_just_asked(fake_filler, ask):
    fake_filler.result = filled(BOOKS_REQUEST)
    state = heard(SHIPPING, working_on(ask))
    state.shipping_request = BOOKS_FROM_SEATTLE_REQUEST
    op = ReactToShippingRequestIntent("op")
    op.perform(state, RecordingIO(SHIPPING))
    [call] = op._shipping_request_filler.calls
    assert call["request"] is BOOKS_FROM_SEATTLE_REQUEST and call["answering"] == ask


@pytest.mark.parametrize("stack, answered", [
    ([SHIPPING_INTENT], None),
    ([ASK_REQUEST, SHIPPING_INTENT], ASK_REQUEST),
    ([ASK_REQUEST, ASK_ORIGIN, SHIPPING_INTENT], ASK_ORIGIN),
    ([ASK_ORIGIN, PRICE_INTENT, SHIPPING_INTENT], None),
])
def test_the_ask_answered_is_the_one_right_under_the_intent_the_utterance_pushed(stack, answered):
    assert working_on(*stack).answered_ask() == answered


def test_shipping_request_takes_a_complete_request_and_asks_for_it_so_it_is_confirmed(fake_filler):
    fake_filler.result = filled(BOOKS_REQUEST)
    state = take(heard(SHIPPING, working_on(GREETING)))
    assert state.shipping_request is BOOKS_REQUEST and state.intent_stack == [GREETING, ASK_REQUEST]


def test_shipping_request_takes_a_partial_request_and_pushes_the_asks_with_the_first_on_top(fake_filler):
    fake_filler.result = filled(ShippingRequest(origin=SEATTLE), ASK_DESTINATION, ASK_SHIPMENT)
    state = take(heard(SHIPPING, working_on(PRICE_INTENT, ASK_REQUEST)))
    assert state.shipping_request == ShippingRequest(origin=SEATTLE)
    assert state.intent_stack == [PRICE_INTENT, ASK_REQUEST, ASK_SHIPMENT, ASK_DESTINATION]


def test_shipping_request_moves_the_asks_already_on_the_stack_to_the_top(fake_filler):
    fake_filler.result = filled(ShippingRequest(origin=SEATTLE), ASK_DESTINATION)
    state = heard(SHIPPING, working_on(ASK_REQUEST, ASK_DESTINATION, GREETING))
    state.shipping_request = ShippingRequest()
    assert take(state).intent_stack == [GREETING, ASK_REQUEST, ASK_DESTINATION]


def test_a_new_delivery_replaces_the_shipping_request(fake_filler):
    fake_filler.result = filled(BOOKS_REQUEST)
    assert take(heard(SHIPPING, with_chairs(ConversationState()))).shipping_request is BOOKS_REQUEST


# --- confirming the shipping request


def confirm(state: ConversationState) -> str:
    """has ConfirmShippingRequest confirm the request; checks that it applies, pops its ask, and notes one thing;
    returns the note"""
    op, io = ConfirmShippingRequest("op"), RecordingIO()
    assert op.can_perform(state, io)
    op.perform(state, io)
    assert ASK_REQUEST not in state.intent_stack and not io.said
    [note] = state.partial_responses
    return note


def test_confirm_shipping_request_notes_the_request_and_invites_the_price_question():
    state = working_on(GREETING, ASK_REQUEST, state=ConversationState(shipping_request=BOOKS_REQUEST))
    assert confirm(state) == BOOKS_CONFIRMATION and state.intent_stack == [GREETING]


def test_confirm_shipping_request_does_not_invite_the_price_question_once_the_human_has_asked_it():
    state = working_on(PRICE_INTENT, ASK_REQUEST, state=ConversationState(shipping_request=BOOKS_REQUEST))
    assert confirm(state) == BOOKS_CONFIRMATION.removesuffix(" Ask me what it costs whenever you are ready.")
    assert state.intent_stack == [PRICE_INTENT]


@pytest.mark.parametrize("make_state", [
    lambda: working_on(ASK_REQUEST, state=ConversationState(shipping_request=BOOKS_FROM_SEATTLE_REQUEST)),
    lambda: working_on(ASK_REQUEST, ASK_ORIGIN, state=ConversationState(shipping_request=BOOKS_REQUEST)),
])
def test_confirm_shipping_request_waits_for_its_ask_on_top_with_the_request_complete(make_state):
    assert not ConfirmShippingRequest("op").can_perform(make_state(), RecordingIO())


def test_a_second_delivery_is_confirmed_and_priced_in_place_of_the_first(chat):
    assert chat(SHIPPING, PRICE, SHIPPING, PRICE, "quit") == [
        INTRODUCTION, BOOKS_CONFIRMATION, BOOKS_PRICE_REPLY, BOOKS_CONFIRMATION, BOOKS_PRICE_REPLY, "Goodbye!",
    ]


# --- asking for what the agent needs


def test_an_ask_takes_its_question_from_the_model():
    assert ASK_REQUEST.question == ShippingRequest.__doc__
    assert ASK_DESTINATION.question == ShippingRequest.model_fields["destination"].description


def test_an_ask_can_only_be_for_an_attribute_of_its_model():
    with pytest.raises(ValueError):
        AskFor(ShippingRequest, "weight")


@pytest.mark.parametrize("request_, satisfied", [
    (None, set()),
    (ShippingRequest(), set()),
    (BOOKS_FROM_SEATTLE_REQUEST, {ASK_ORIGIN, ASK_SHIPMENT}),
    (BOOKS_REQUEST, {ASK_REQUEST, ASK_ORIGIN, ASK_DESTINATION, ASK_SHIPMENT}),
])
def test_an_ask_is_satisfied_once_the_request_has_its_attribute_or_for_the_whole_request_every_attribute(
    request_, satisfied
):
    asks = {ASK_REQUEST, ASK_ORIGIN, ASK_DESTINATION, ASK_SHIPMENT}
    assert {ask for ask in asks if ask.satisfied(request_)} == satisfied


def test_pop_satisfied_ask_pops_the_asks_for_attributes_on_top_once_they_are_satisfied():
    state = working_on(PRICE_INTENT, ASK_REQUEST, ASK_ORIGIN, ASK_DESTINATION)
    state.shipping_request = BOOKS_REQUEST
    op = PopSatisfiedAsk("op")
    for remaining in ([PRICE_INTENT, ASK_REQUEST, ASK_ORIGIN], [PRICE_INTENT, ASK_REQUEST]):
        assert op.can_perform(state, RecordingIO())
        op.perform(state, RecordingIO())
        assert state.intent_stack == remaining
    assert not op.can_perform(state, RecordingIO()), "the ask for the whole request is left to be confirmed"


@pytest.mark.parametrize("make_state", [
    lambda: working_on(ASK_DESTINATION),
    lambda: working_on(ASK_REQUEST, GREETING, state=with_chairs(ConversationState())),
])
def test_pop_satisfied_ask_waits_for_a_satisfied_ask_on_top(make_state):
    assert not PopSatisfiedAsk("op").can_perform(make_state(), RecordingIO())


def test_ask_question_ends_the_turn_with_the_question_and_keeps_the_ask():
    state, io = working_on(PRICE_INTENT, ASK_REQUEST, ASK_DESTINATION), RecordingIO()
    state.shipping_request = BOOKS_FROM_SEATTLE_REQUEST
    state.partial_responses = [ReactToGreetingIntent.REPLY]
    op = AskQuestion("op")
    assert op.can_perform(state, io)
    op.perform(state, io)
    assert io.said == [f"{ReactToGreetingIntent.REPLY} {DESTINATION_QUESTION}"]
    assert state.listening and state.asked == DESTINATION_QUESTION and state.partial_responses == []
    assert state.top == ASK_DESTINATION


@pytest.mark.parametrize("make_state", [
    lambda: working_on(ASK_REQUEST, state=with_chairs(ConversationState())),
    lambda: working_on(ASK_REQUEST, GREETING),
])
def test_ask_question_waits_for_an_unsatisfied_ask_on_top(make_state):
    assert not AskQuestion("op").can_perform(make_state(), RecordingIO())


# --- adding vehicles to the fleet


def add(phrase: str, state: ConversationState | None = None) -> ConversationState:
    """has a fresh ReactToAddToFleetIntent handle the phrase; checks that it applies, pops its intent, and neither
    notes nor says anything, leaving that to the asks; returns the state"""
    state = heard(phrase, state)
    op, io = ReactToAddToFleetIntent("op"), RecordingIO(phrase)
    assert op.can_perform(state, io)
    op.perform(state, io)
    assert HumanIntent.ADD_TO_FLEET not in state.intent_stack and not io.said and not state.partial_responses
    return state


def test_add_to_fleet_starts_a_vehicle_and_asks_for_its_missing_fields_the_first_on_top():
    state = add(ADD_TRUCK, working_on(GREETING))
    assert state.vehicle_draft.display_name == "Road Runner" and state.vehicle_draft.name == "truck3"
    assert state.intent_stack == [
        GREETING, ASK_TRUCK, ASK_TRUCK_PLATE, AskFor(Truck, "max_payload"), AskFor(Truck, "model"),
    ]


def test_add_to_fleet_asks_for_the_kind_when_the_human_does_not_say():
    state = add(ADD_VEHICLE)
    assert state.vehicle_draft is None and state.intent_stack == [ASK_KIND]


def test_add_to_fleet_continues_the_vehicle_being_asked_for():
    state = add(TRUCK_DETAILS, adding(working_on(ASK_TRUCK, AskFor(Truck, "model")), "partial truck"))
    assert build(state.vehicle_draft) == ROAD_RUNNER


def test_add_to_fleet_fills_in_where_the_vehicle_is_when_the_utterance_answers_that():
    state = add(AT_DENVER, adding(working_on(ASK_PLACEMENT, ASK_LOCATION), "whole truck"))
    assert state.placement_draft.entity2.name == "terminal_denver" and state.top == ASK_PLACEMENT


def test_a_new_vehicle_drops_the_one_that_was_not_finished_with_every_ask_for_it():
    stale = adding(working_on(ASK_PLACEMENT, ASK_LOCATION, PRICE_INTENT), "whole truck")
    state = add(ADD_VEHICLE, stale)
    assert state.placement_draft is None and state.intent_stack == [PRICE_INTENT, ASK_KIND]


def test_place_vehicle_builds_the_vehicle_and_asks_where_it_is():
    state = adding(working_on(GREETING, ASK_TRUCK), "partial truck")
    state.vehicle_draft = draft_model(Truck)(**dict(ROAD_RUNNER))
    op = PlaceVehicle("op")
    assert op.can_perform(state, RecordingIO())
    op.perform(state, RecordingIO())
    assert state.placement_draft.entity1 == ROAD_RUNNER and state.placement_draft.entity2 is None
    assert state.intent_stack == [GREETING, ASK_PLACEMENT, ASK_LOCATION]


def test_place_vehicle_waits_for_the_whole_vehicle():
    assert not PlaceVehicle("op").can_perform(adding(working_on(ASK_TRUCK), "partial truck"), RecordingIO())


def test_propose_addition_swaps_the_placement_for_the_confirmation_once_the_vehicle_is_placed():
    state = adding(working_on(GREETING, ASK_PLACEMENT), "placed truck")
    op = ProposeAddition("op")
    assert op.can_perform(state, RecordingIO())
    op.perform(state, RecordingIO())
    assert state.intent_stack == [GREETING, ConfirmChange()]
    assert not ProposeAddition("op").can_perform(adding(working_on(ASK_PLACEMENT), "whole truck"), RecordingIO())


def test_ask_to_confirm_addition_asks_whether_to_add_the_vehicle_there_and_waits():
    state, io = adding(working_on(ConfirmChange()), "placed truck"), RecordingIO()
    op = AskToConfirmChange("op")
    assert op.can_perform(state, io)
    op.perform(state, io)
    assert io.said == [ROAD_RUNNER_QUESTION] and state.listening and state.top == ConfirmChange()


def test_confirm_adds_the_vehicle_where_it_is_so_the_planner_can_use_it():
    state = adding(heard(YES, working_on(GREETING, ConfirmChange())), "placed truck")
    op = ReactToConfirmIntent("op")
    assert op.can_perform(state, RecordingIO())
    op.perform(state, RecordingIO())
    assert state.partial_responses == [ROAD_RUNNER_ADDED] and state.intent_stack == [GREETING]
    assert state.vehicle_draft is None and state.placement_draft is None
    [at] = state.reasoner.find_relations(At, entity1="truck3")
    assert at.entity2.name == "terminal_denver" and state.reasoner.find_entity("Road Runner").name == "truck3"


def test_deny_drops_the_vehicle_without_adding_it():
    state = adding(heard(NO, working_on(ConfirmChange())), "placed truck")
    ReactToDenyIntent("op").perform(state, RecordingIO())
    assert state.partial_responses == ["OK, I did not add Road Runner to the fleet."] and state.intent_stack == []
    assert state.reasoner.find_entity("Road Runner") is None and state.vehicle_draft is None


@pytest.mark.parametrize("op_type, phrase", [(ReactToConfirmIntent, YES), (ReactToDenyIntent, NO)])
def test_a_yes_or_no_to_nothing_is_let_go(op_type, phrase):
    state = heard(phrase, working_on(ASK_DESTINATION))
    op_type("op").perform(state, RecordingIO())
    assert state.intent_stack == [ASK_DESTINATION] and state.partial_responses == []


def test_ok_is_an_answer_while_the_agent_waits_for_a_yes_or_no_and_otherwise_only_an_acknowledgement():
    confirming = working_on(ConfirmChange())
    confirming.listening = True
    assert not Acknowledge("op").can_perform(confirming, RecordingIO("ok"))
    assert RecognizeIntent("op").can_perform(confirming, RecordingIO("ok"))
    assert Acknowledge("op").can_perform(ConversationState(), RecordingIO("ok"))
    assert not RecognizeIntent("op").can_perform(ConversationState(), RecordingIO("ok"))


# --- closing blocked routes


def report(phrase: str, state: ConversationState | None = None) -> ConversationState:
    """has a fresh ReactToObstructRouteIntent handle the phrase; checks that it applies, pops its intent, and neither
    notes nor says anything, leaving that to the asks; returns the state"""
    state = heard(phrase, state)
    op, io = ReactToObstructRouteIntent("op"), RecordingIO(phrase)
    assert op.can_perform(state, io)
    op.perform(state, io)
    assert HumanIntent.OBSTRUCT_ROUTE not in state.intent_stack and not io.said and not state.partial_responses
    return state


def test_obstruct_route_fills_in_the_route_and_asks_for_it_whole():
    state = report(SNOWED_IN, working_on(GREETING))
    assert (state.route_report.city1.name, state.route_report.city2.name) == ("seattle", "san_francisco")
    assert state.route_type is RouteType.ROAD and state.intent_stack == [GREETING, ASK_ROUTE]


def test_obstruct_route_asks_for_the_kind_when_the_cause_does_not_tell():
    state = report(BLOCKED)
    assert state.route_type is None and state.intent_stack == [ASK_ROUTE, ASK_ROUTE_KIND]


def test_obstruct_route_continues_the_route_being_asked_for():
    state = adding(working_on(ASK_ROUTE, ASK_ROUTE_KIND), "partial route")
    state.route_report = report(BLOCKED).route_report
    state = report(THE_ROAD, state)
    assert state.route_report.city2.name == "denver" and state.route_type is RouteType.ROAD


def test_a_report_while_the_agent_is_not_asking_about_a_route_starts_over():
    stale = adding(working_on(PRICE_INTENT), "whole route")
    stale.route_type = RouteType.AIR
    stale.route_report.city2 = stale.reasoner.find_entity("denver")
    state = report(SNOWED_IN, stale)
    assert state.route_report.city2.name == "san_francisco" and state.route_type is RouteType.ROAD
    assert state.intent_stack == [PRICE_INTENT, ASK_ROUTE]


def propose(state: ConversationState) -> ConversationState:
    """has a fresh ProposeClosure handle the route reported; checks that it applies; returns the state"""
    op = ProposeClosure("op")
    assert op.can_perform(state, RecordingIO())
    op.perform(state, RecordingIO())
    return state


@pytest.mark.parametrize("kind, model, ends", [
    (RouteType.ROAD, TruckRoute, {"seattle", "san_francisco"}),
    (RouteType.AIR, AirRoute, {"SEA", "SFO"}),
])
def test_propose_closure_finds_the_route_between_the_cities_and_asks_to_close_it(kind, model, ends):
    assert kind.model is model
    state = adding(working_on(GREETING, ASK_ROUTE), "whole route")
    state.route_type = kind
    propose(state)
    change = state.pending_change
    assert {change.route.entity1.name, change.route.entity2.name} == ends and change.kind is kind
    assert state.intent_stack == [GREETING, ConfirmChange()]
    what = "road" if kind is RouteType.ROAD else "flights"
    assert change.question == f"Shall I close the {what} between Seattle and San Francisco?"


def test_propose_closure_finds_a_route_declared_the_other_way_round():
    state = adding(working_on(ASK_ROUTE), "whole route")
    find = state.reasoner.find_entity
    state.route_report = RouteReport(city1=find("san_francisco"), city2=find("seattle"))
    assert propose(state).pending_change.route.entity1.name == "seattle"


def test_propose_closure_says_so_and_drops_the_report_when_there_is_no_such_route():
    state = adding(working_on(GREETING, ASK_ROUTE), "whole route")
    state.route_report.city2 = state.reasoner.find_entity("seattle")
    propose(state)
    assert state.partial_responses == [
        "I don't know of the road between Seattle and Seattle, so there is nothing to close."
    ]
    assert state.intent_stack == [GREETING] and state.route_report is None and state.pending_change is None


def test_propose_closure_waits_for_both_cities_and_the_kind():
    assert not ProposeClosure("op").can_perform(adding(working_on(ASK_ROUTE), "partial route"), RecordingIO())
    state = adding(working_on(ASK_ROUTE), "whole route")
    state.route_type = None
    assert not ProposeClosure("op").can_perform(state, RecordingIO())


def test_confirming_closes_the_road_and_the_planner_plans_around_it():
    state = propose(adding(working_on(ASK_ROUTE), "whole route"))
    state = heard(YES, state)
    ReactToConfirmIntent("op").perform(state, RecordingIO())
    assert state.partial_responses == [f"Closed {SEATTLE_SF_ROAD}; I will plan deliveries around it."]
    assert state.intent_stack == [] and state.route_report is None and state.pending_change is None
    assert state.reasoner.find_relations(TruckRoute, "seattle", "san_francisco") == []
    state.partial_responses.clear()
    assert estimate(state).price == 3210.0, "the pallet of chairs now goes by way of Denver, not straight for $1,810"


def test_saying_no_leaves_the_road_open():
    state = heard(NO, propose(adding(working_on(ASK_ROUTE), "whole route")))
    ReactToDenyIntent("op").perform(state, RecordingIO())
    assert state.partial_responses == [f"OK, I left {SEATTLE_SF_ROAD} open."] and state.pending_change is None
    assert state.reasoner.find_relations(TruckRoute, "seattle", "san_francisco") != []


# --- answering questions about the state of the world


def test_inspect_state_notes_the_answer_and_pops_its_intent():
    state, note, _ = react(ReactToInspectStateIntent, WHICH_TRUCKS, working_on(GREETING))
    assert note == f"The fleet has 2 trucks: {BERTHA}; {ROCKET}." and state.intent_stack == [GREETING]


def test_inspect_state_reads_the_question_with_the_ask_it_answers(monkeypatch):
    seen = {}
    real = conductor.StateInspector.__call__

    def spy(self, **args):
        seen.update(args)
        return real(self, **args)

    monkeypatch.setattr(conductor.StateInspector, "__call__", spy)
    react(ReactToInspectStateIntent, WHICH_TRUCKS, working_on(ASK_DESTINATION))
    assert seen["answering"] == ASK_DESTINATION


def test_the_fleet_answer_says_so_when_there_is_no_vehicle_of_the_kind():
    assert conductor.fleet_answer(VehicleType.AIRPLANE, []) == "The fleet has no airplanes."
    assert conductor.fleet_answer(None, []) == "The fleet has no vehicles."


@pytest.mark.parametrize("noun, expected", [("truck", "a truck"), ("airplane", "an airplane")])
def test_with_article_puts_a_or_an_in_front_as_the_noun_sounds(noun, expected):
    assert conductor.with_article(noun) == expected


# --- estimating prices


def test_conversation_starts_with_no_shipping_request_and_no_price():
    state = ConversationState()
    assert state.shipping_request is None and not state.planned and state.plan is None and state.price is None


@pytest.mark.parametrize("request_, asks", [
    (None, [ASK_REQUEST]),
    (ShippingRequest(), [ASK_REQUEST, ASK_SHIPMENT, ASK_DESTINATION, ASK_ORIGIN]),
    (BOOKS_FROM_SEATTLE_REQUEST, [ASK_REQUEST, ASK_DESTINATION]),
])
def test_estimate_price_asks_for_the_request_and_its_missing_parts_and_keeps_its_intent_under_them(request_, asks):
    state = heard(PRICE, working_on(GREETING))
    state.shipping_request = request_
    op, io = ReactToEstimatePriceIntent("op"), RecordingIO()
    assert op.can_perform(state, io)
    op.perform(state, io)
    assert not io.said and not state.partial_responses
    assert state.intent_stack == [GREETING, PRICE_INTENT, *asks]


def test_estimate_price_asked_again_while_the_agent_is_asking_moves_the_asks_back_above_it():
    state = heard(PRICE, working_on(PRICE_INTENT, ASK_REQUEST, ASK_DESTINATION))
    state.shipping_request = BOOKS_FROM_SEATTLE_REQUEST
    ReactToEstimatePriceIntent("op").perform(state, RecordingIO())
    assert state.intent_stack == [PRICE_INTENT, ASK_REQUEST, ASK_DESTINATION]


def test_estimate_price_does_not_ask_once_the_request_is_complete():
    assert not ReactToEstimatePriceIntent("op").can_perform(heard(PRICE, with_chairs(ConversationState())),
                                                            RecordingIO())


def estimate(state: ConversationState | None = None) -> ConversationState:
    """has PlanDelivery handle a price question about the pallet of chairs; checks that it applies without noting or
    saying anything, and that the intent stays on the stack for DescribePlan"""
    state = heard(PRICE, with_chairs(state or ConversationState()))
    op, io = PlanDelivery("op"), RecordingIO()
    assert op.can_perform(state, io)
    op.perform(state, io)
    assert not io.said and not state.partial_responses
    assert state.top is PRICE_INTENT and state.planned
    return state


def no_plan():
    raise RuntimeError("Plan not available")


def test_plan_delivery_records_the_plan_and_its_cost_as_the_price():
    state = estimate()
    assert state.price == 1810.0 and sum(action.cost for action in state.plan) == state.price
    assert isinstance(state.plan[-1], UnloadAction) and state.plan[-1].location.name == "shipping-destination"


def test_plan_delivery_leaves_the_world_graph_and_the_planner_as_they_were():
    state = ConversationState()
    before = state.reasoner.world.snapshot
    estimate(state)
    assert state.reasoner.world.snapshot == before
    assert not state.planner.can_plan()


def test_plan_delivery_records_no_plan_when_no_delivery_can_be_planned(monkeypatch):
    state = ConversationState()
    before = state.reasoner.world.snapshot
    monkeypatch.setattr(state.planner, "plan", no_plan)
    estimate(state)
    assert state.plan is None and state.price is None
    assert state.reasoner.world.snapshot == before


@pytest.mark.parametrize("make_state", [
    lambda: working_on(PRICE_INTENT, state=ConversationState(shipping_request=BOOKS_FROM_SEATTLE_REQUEST)),
    lambda: planned(working_on(PRICE_INTENT, state=with_chairs(ConversationState()))),
    lambda: working_on(PRICE_INTENT, ASK_REQUEST, state=with_chairs(ConversationState())),
])
def test_plan_delivery_waits_for_the_price_question_on_top_with_a_complete_request_not_planned_yet(make_state):
    assert not PlanDelivery("op").can_perform(make_state(), RecordingIO())


def describe(state: ConversationState) -> str:
    """has DescribePlan describe the plan; checks that it applies and notes one thing; returns the note"""
    op, io = DescribePlan("op"), RecordingIO()
    assert op.can_perform(state, io)
    op.perform(state, io)
    assert not io.said and not state.planned
    [note] = state.partial_responses
    return note


def test_describe_plan_notes_the_price_and_the_description_of_the_plan_and_pops_its_intent():
    state = estimate(working_on(GREETING))
    assert describe(state) == PRICE_REPLY
    assert state.intent_stack == [GREETING]


def test_describe_plan_describes_the_plan_without_the_language_model(fake_language):
    describe(estimate())
    assert fake_language.calls == []


def test_describe_plan_says_so_when_no_delivery_could_be_planned(monkeypatch):
    state = ConversationState()
    monkeypatch.setattr(state.planner, "plan", no_plan)
    note = describe(estimate(state))
    assert note == (
        "Sorry, I could not find a way to deliver the pallet of office chairs from Seattle to San Francisco."
    )
    assert state.intent_stack == []


def test_describe_plan_waits_until_the_delivery_has_been_planned():
    assert not DescribePlan("op").can_perform(heard(PRICE), RecordingIO())
    assert not DescribePlan("op").can_perform(working_on(PRICE_INTENT, state=with_chairs(ConversationState())),
                                              RecordingIO())


# --- ending the turn


def test_reply_says_the_partial_responses_and_starts_listening_once_the_stack_is_empty():
    state, io = working_on(), RecordingIO()
    state.partial_responses = ["Hello!", "Glad to help."]
    op = Reply("op")
    assert op.can_perform(state, io)
    op.perform(state, io)
    assert io.said == ["Hello! Glad to help."]
    assert state.partial_responses == [] and state.listening and state.asked is None


@pytest.mark.parametrize("make_state", [
    lambda: working_on(GREETING),
    lambda: heard(GIBBERISH),
    lambda: ConversationState(),
])
def test_reply_waits_while_there_is_work_left_or_the_agent_is_listening(make_state):
    assert not Reply("op").can_perform(make_state(), RecordingIO())
