"""offline tests of the logistics support agent's conversation

a fake EnumClassifier stands in for the LLM (intents are keyed by phrase), and a fake describe_facts names the actions
it is given; the human's lines are fed to input() from a script, so the whole perceive-decide-act loop runs deterministically with no API key. When the script runs out,
input() raises EOFError, which the agent treats as quitting.
"""

import re
from collections import deque

import pytest

from logistics_agent import assistant, conductor, extract_shipping_request
from logistics_agent.communicator import HumanIntent, Utterance
from logistics_agent.conductor import (
    Acknowledge,
    AskForShippingRequest,
    INTRODUCTION,
    ConversationState,
    DescribePlan,
    Introduce,
    ReactToAskCapabilitiesIntent,
    ReactToEstimatePriceIntent,
    ReactToGreetingIntent,
    ReactToShippingRequestIntent,
    RecognizeIntent,
    Reply,
    RespondUnknown,
    SayGoodbye,
)
from logistics_agent.domain import At, City, Location, Shipment, ShippingRequest, UnloadAction
from logistics_agent.extract_shipping_request import (
    ExtractDestination,
    ExtractOrigin,
    ExtractShippingRequest,
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
    "Delivering the box of books from Seattle to Denver costs $1,112.00. "
    "DriveLocalAction, LoadAction, DriveLongDistanceAction, UnloadAction"
)
PRICE_REPLY = (
    "Delivering the pallet of office chairs from Seattle to San Francisco "
    "costs $802.00. DriveLocalAction, LoadAction, DriveLongDistanceAction, UnloadAction"
)
QUESTION = "Where is the shipment now?"

INTENTS = {
    GIBBERISH: None,
    HELLO: HumanIntent.GREETING,
    CAPABILITIES: HumanIntent.ASK_CAPABILITIES,
    PRICE: HumanIntent.ESTIMATE_PRICE,
    SHIPPING: HumanIntent.SHIPPING_REQUEST,
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
    """stands in for ModelPopulator in the ExtractShippingRequest sub-process: whatever the human said, the delivery
    is a box of books, 200 pounds, from Seattle to Denver"""

    CITIES = {ExtractOrigin.TASK_DESC: "Seattle", ExtractDestination.TASK_DESC: "Denver"}

    def __init__(self, model_type, task_desc: str) -> None:
        self.model_type, self.task_desc = model_type, task_desc

    def __call__(self, utterance: str, llm, *extra: str, **kwargs):
        if self.model_type is Shipment:
            return Shipment(name="books", details="box of books", weight=200)
        return self.model_type.model_validate({"city": self.CITIES[self.task_desc]})


@pytest.fixture(autouse=True)
def fake_language(monkeypatch):
    """replaces the LLM-backed EnumClassifier and describe_facts used by the operators; returns the fake
    describe_facts"""
    monkeypatch.setattr(conductor, "EnumClassifier", FakeEnumClassifier)
    fake_describe_facts = FakeDescribeFacts()
    monkeypatch.setattr(conductor, "describe_facts", fake_describe_facts)
    monkeypatch.setattr(extract_shipping_request, "ModelPopulator", FakeModelPopulator)
    return fake_describe_facts


@pytest.fixture
def chat(monkeypatch, capsys):
    """runs the agent through a script of human lines; returns the agent's replies, in order"""

    def run(*lines: str) -> list[str]:
        script = deque(lines)

        def scripted_input(_prompt: str = "") -> str:
            if not script:
                raise EOFError
            return script.popleft()

        monkeypatch.setattr("builtins.input", scripted_input)
        assistant.main()
        out = capsys.readouterr().out
        replies = out.split("cogent> ")[1:]
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
    assert chat(PRICE, "quit") == [INTRODUCTION, AskForShippingRequest.REPLY, "Goodbye!"]


def test_a_described_delivery_is_confirmed_and_then_priced(chat):
    assert chat(SHIPPING, PRICE, "quit") == [INTRODUCTION, BOOKS_CONFIRMATION, BOOKS_PRICE_REPLY, "Goodbye!"]


def test_ok_gets_no_reply_and_thanks_is_answered(chat):
    assert chat("ok", "thanks!", "quit") == [INTRODUCTION, "You're welcome.", "Goodbye!"]


def test_every_intent_has_an_operator_that_handles_it():
    assert set().union(*HANDLERS.values()) - {None} == set(HumanIntent)


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
    """a conversation state in which the phrase has been heard, its intent recognized and pushed, as RecognizeIntent
    leaves it"""
    state = state or ConversationState()
    state.listening = False
    state.human_utterance, state.human_intent = Utterance(phrase), INTENTS[phrase]
    if state.human_intent is not None:
        state.push(state.human_intent)
    return state


def working_on(*stack: HumanIntent, state: ConversationState | None = None) -> ConversationState:
    """a conversation state between utterances, with the intents on the stack (the last on top)"""
    state = state or ConversationState()
    state.listening = False
    state.intent_stack = list(stack)
    return state


def react(op_type: type, phrase: str, state: ConversationState | None = None) -> tuple[ConversationState, str, object]:
    """has a fresh operator of the type handle the phrase; checks that it applies, that it handles the utterance, and
    that it notes one thing instead of speaking; returns the state, the note, and the operator"""
    state = heard(phrase, state)
    op, io = op_type("op"), RecordingIO()
    assert op.can_perform(state, io)
    responses_before = len(state.partial_responses)
    op.perform(state, io)
    assert state.human_utterance is None and state.human_intent is None
    assert not io.said, "operators that handle an utterance note what to say, and the turn says it at the end"
    [note] = state.partial_responses[responses_before:]
    return state, note, op


GREETING = HumanIntent.GREETING
CAPABILITY = HumanIntent.ASK_CAPABILITIES
PRICE_INTENT = HumanIntent.ESTIMATE_PRICE


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
    ReactToGreetingIntent, ReactToAskCapabilitiesIntent, ReactToShippingRequestIntent,
    AskForShippingRequest, ReactToEstimatePriceIntent, DescribePlan, Reply,
]


def introduced(state: ConversationState) -> ConversationState:
    state.introduced = True
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
    (lambda: introduced(heard(PRICE)), PRICE, AskForShippingRequest),
    (lambda: introduced(heard(PRICE, with_chairs(ConversationState()))), PRICE, ReactToEstimatePriceIntent),
    (lambda: introduced(working_on(PRICE_INTENT)), PRICE, DescribePlan),
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


# --- recognize intent


def test_recognize_intent_stores_the_utterance_stops_listening_and_pushes_the_intent():
    state, io = ConversationState(), RecordingIO(HELLO)
    op = RecognizeIntent("recognize_intent")
    assert op.can_perform(state, io)
    op.perform(state, io)
    assert state.human_utterance == Utterance(HELLO)
    assert state.human_intent is GREETING
    assert state.intent_stack == [GREETING] and not state.listening
    assert not op.can_perform(state, io)


def test_recognize_intent_pushes_nothing_for_an_unknown_intent():
    state = ConversationState()
    RecognizeIntent("recognize_intent").perform(state, RecordingIO(GIBBERISH))
    assert state.human_utterance == Utterance(GIBBERISH) and state.human_intent is None
    assert state.intent_stack == []


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


@pytest.mark.parametrize("phrase", [None, "quit", "ok bye now", "EXIT please", "thanks", "ok"])
def test_recognize_intent_skips_silence_quitting_and_acknowledgements(phrase):
    op = RecognizeIntent("recognize_intent")
    assert not op.can_perform(ConversationState(), RecordingIO(phrase))


def test_recognize_intent_waits_until_the_agent_is_listening():
    op = RecognizeIntent("recognize_intent")
    assert not op.can_perform(heard(HELLO), RecordingIO(GIBBERISH))
    assert not op.can_perform(working_on(), RecordingIO(GIBBERISH))


# --- operators that handle an utterance apply only to its intent


HANDLERS = {
    ReactToGreetingIntent: {HumanIntent.GREETING},
    ReactToAskCapabilitiesIntent: {HumanIntent.ASK_CAPABILITIES},
    ReactToEstimatePriceIntent: {HumanIntent.ESTIMATE_PRICE},
    AskForShippingRequest: {HumanIntent.ESTIMATE_PRICE},
    ReactToShippingRequestIntent: {HumanIntent.SHIPPING_REQUEST},
    RespondUnknown: {None},
}


@pytest.mark.parametrize("op_type", list(HANDLERS))
@pytest.mark.parametrize("intent", [*HumanIntent, None])
def test_each_handler_applies_only_to_its_intent(op_type, intent):
    state = working_on()
    if op_type is ReactToEstimatePriceIntent:
        with_chairs(state)
    state.human_utterance, state.human_intent = Utterance(GIBBERISH), intent
    assert op_type("op").can_perform(state, RecordingIO()) == (intent in HANDLERS[op_type])


@pytest.mark.parametrize("op_type", list(HANDLERS))
def test_handlers_wait_for_an_utterance(op_type):
    assert not op_type("op").can_perform(working_on(GREETING), RecordingIO())


@pytest.mark.parametrize("op_type, phrase", [
    (ReactToGreetingIntent, HELLO), (ReactToAskCapabilitiesIntent, CAPABILITIES),
])
def test_small_talk_pops_its_intent_and_leaves_the_rest_of_the_stack(op_type, phrase):
    other = CAPABILITY if op_type is ReactToGreetingIntent else GREETING
    state, _, _ = react(op_type, phrase, working_on(other))
    assert state.intent_stack == [other]


# --- respond unknown


def test_respond_unknown_notes_an_apology_and_leaves_the_stack():
    state, note, _ = react(RespondUnknown, GIBBERISH, working_on(GREETING))
    assert note == "Sorry, I am not designed to respond to that."
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


class FakeExtractShippingRequest:
    """stands in for the ExtractShippingRequest sub-process: returns a canned final state, or None as if it did not
    finish; records how often it was restarted, the output channels it was linked to, and what it was called with"""

    result: ShippingRequestState | None = None

    def __init__(self) -> None:
        self.reinits = 0
        self.outputs: dict = {}
        self.calls: list[dict] = []

    def reinit(self) -> "FakeExtractShippingRequest":
        self.reinits += 1
        return self

    def set_output_channel(self, key: str, channel) -> "FakeExtractShippingRequest":
        self.outputs[key] = channel
        return self

    def __call__(self, **args) -> ShippingRequestState | None:
        self.calls.append(args)
        return self.result


@pytest.fixture
def fake_extraction(monkeypatch):
    """replaces the ExtractShippingRequest sub-process the operator sets up; returns its class, whose result the test
    sets"""
    FakeExtractShippingRequest.result = None
    monkeypatch.setattr(conductor, "ExtractShippingRequest", FakeExtractShippingRequest)
    return FakeExtractShippingRequest


def test_shipping_request_sets_up_its_sub_process_once():
    assert isinstance(ReactToShippingRequestIntent("op")._extract_shipping_request, ExtractShippingRequest)


def test_shipping_request_runs_a_fresh_sub_process_with_the_humans_utterance_and_the_reasoner(fake_extraction):
    state = heard(SHIPPING)
    op, io = ReactToShippingRequestIntent("op"), RecordingIO(SHIPPING)
    op.perform(state, io)
    sub_process = op._extract_shipping_request
    assert sub_process.reinits == 1
    assert sub_process.calls == [{"utterance": Utterance(SHIPPING), "reasoner": state.reasoner}]


def test_shipping_request_takes_the_shipping_request_once_all_of_it_is_populated_and_confirms_it(fake_extraction):
    fake_extraction.result = ShippingRequestState(origin=SEATTLE, destination=DENVER, shipment=BOOKS)
    state = heard(SHIPPING, working_on(GREETING))
    op, io = ReactToShippingRequestIntent("op"), RecordingIO(SHIPPING)
    assert op.can_perform(state, io)
    op.perform(state, io)
    request = state.shipping_request
    assert (request.origin, request.destination, request.shipment) == (SEATTLE, DENVER, BOOKS)
    assert state.partial_responses == [BOOKS_CONFIRMATION] and io.said == []
    assert state.intent_stack == [GREETING] and state.human_utterance is None


@pytest.mark.parametrize("result", [None, ShippingRequestState(origin=SEATTLE, shipment=BOOKS)])
def test_shipping_request_keeps_the_shipping_request_and_says_so_when_not_all_of_it_is_populated(
    fake_extraction, result
):
    fake_extraction.result = result
    state = ConversationState()
    before = state.shipping_request
    state, note, _ = react(ReactToShippingRequestIntent, SHIPPING, state)
    assert state.shipping_request is before and note == ReactToShippingRequestIntent.NOT_UNDERSTOOD
    assert state.intent_stack == []


def test_a_new_delivery_replaces_the_shipping_request(fake_extraction):
    fake_extraction.result = ShippingRequestState(origin=SEATTLE, destination=DENVER, shipment=BOOKS)
    state, note, _ = react(ReactToShippingRequestIntent, SHIPPING, with_chairs(ConversationState()))
    assert state.shipping_request.shipment == BOOKS and note == BOOKS_CONFIRMATION


def test_a_second_delivery_is_confirmed_and_priced_in_place_of_the_first(chat):
    assert chat(SHIPPING, PRICE, SHIPPING, PRICE, "quit") == [
        INTRODUCTION, BOOKS_CONFIRMATION, BOOKS_PRICE_REPLY, BOOKS_CONFIRMATION, BOOKS_PRICE_REPLY, "Goodbye!",
    ]


# --- estimating prices


def test_conversation_starts_with_no_shipping_request_and_no_price():
    state = ConversationState()
    assert state.shipping_request is None and state.plan is None and state.price is None


def test_ask_for_shipping_request_notes_what_the_agent_needs_and_pops_its_intent():
    state, note, _ = react(AskForShippingRequest, PRICE, working_on(GREETING))
    assert note == AskForShippingRequest.REPLY and state.intent_stack == [GREETING]


def test_ask_for_shipping_request_does_not_apply_once_there_is_a_shipping_request():
    state = heard(PRICE, with_chairs(ConversationState()))
    assert not AskForShippingRequest("op").can_perform(state, RecordingIO(PRICE))


def estimate(state: ConversationState | None = None) -> ConversationState:
    """has ReactToEstimatePriceIntent handle a price question about the pallet of chairs; checks that it applies, that
    it handles the utterance without noting or saying anything, and that the intent stays on the stack for
    DescribePlan"""
    state = heard(PRICE, with_chairs(state or ConversationState()))
    op, io = ReactToEstimatePriceIntent("op"), RecordingIO()
    assert op.can_perform(state, io)
    op.perform(state, io)
    assert state.human_utterance is None and state.human_intent is None
    assert not io.said and not state.partial_responses
    assert state.top is PRICE_INTENT
    return state


def no_plan():
    raise RuntimeError("Plan not available")


def test_estimate_price_records_the_plan_and_its_cost_as_the_price():
    state = estimate()
    assert state.price == 802.0 and sum(action.cost for action in state.plan) == state.price
    assert isinstance(state.plan[-1], UnloadAction) and state.plan[-1].location.name == "shipping-destination"


def test_estimate_price_leaves_the_world_graph_and_the_planner_as_they_were():
    state = ConversationState()
    before = state.reasoner.world.snapshot
    estimate(state)
    assert state.reasoner.world.snapshot == before
    assert not state.planner.can_plan()


def test_estimate_price_records_no_plan_when_no_delivery_can_be_planned(monkeypatch):
    state = ConversationState()
    before = state.reasoner.world.snapshot
    monkeypatch.setattr(state.planner, "plan", no_plan)
    estimate(state)
    assert state.plan is None and state.price is None
    assert state.reasoner.world.snapshot == before


def describe(state: ConversationState) -> str:
    """has DescribePlan describe the plan; checks that it applies and notes one thing; returns the note"""
    op, io = DescribePlan("op"), RecordingIO()
    assert op.can_perform(state, io)
    op.perform(state, io)
    assert not io.said
    [note] = state.partial_responses
    return note


def test_describe_plan_notes_the_price_and_the_description_of_the_plan_and_pops_its_intent():
    state = estimate(working_on(GREETING))
    assert describe(state) == PRICE_REPLY
    assert state.intent_stack == [GREETING]


def test_describe_plan_has_describe_facts_describe_the_plan_with_the_default_llm(fake_language):
    state = estimate()
    describe(state)
    [(facts, task_desc, llm)] = fake_language.calls
    assert facts == tuple(state.plan) and task_desc == DescribePlan.TASK_DESC and llm is conductor.DEFAULT_LLM


def test_describe_plan_does_not_describe_facts_when_no_delivery_could_be_planned(monkeypatch, fake_language):
    state = ConversationState()
    monkeypatch.setattr(state.planner, "plan", no_plan)
    describe(estimate(state))
    assert fake_language.calls == []


def test_describe_plan_says_so_when_no_delivery_could_be_planned(monkeypatch):
    state = ConversationState()
    monkeypatch.setattr(state.planner, "plan", no_plan)
    note = describe(estimate(state))
    assert note == (
        "Sorry, I could not find a way to deliver the pallet of office chairs from Seattle to San Francisco."
    )
    assert state.intent_stack == []


def test_describe_plan_waits_until_the_price_question_has_been_handled():
    assert not DescribePlan("op").can_perform(heard(PRICE), RecordingIO())


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
