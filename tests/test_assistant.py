"""offline tests of the logistics support agent's conversation

a fake EnumClassifier, a fake ModelPopulator and a fake describe_facts stand in for the LLM (intents are keyed by phrase,
populated models by model type and phrase, and facts are described by how many there are), and the human's lines are
fed to input() from a script, so the whole perceive-decide-act loop runs deterministically with no
API key. When the script runs out, input() raises EOFError, which the agent treats as quitting.
"""

from collections import deque
from types import SimpleNamespace

import pytest
from pydantic_ai.exceptions import UnexpectedModelBehavior

from logistics_agent import assistant, conductor, extract_and_set_current, extract_and_set_goal
from logistics_agent.communicator import (
    AtFact,
    CurrentLocation,
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
from logistics_agent.conductor import (
    INTRODUCTION,
    AskWhereThePackageGoes,
    AskWhereThePackageIs,
    ConversationState,
    Introduce,
    ReactToAddEntityIntent,
    ReactToAskAboutEntitiesIntent,
    ReactToAskCurrentStateIntent,
    ReactToCurrentLocationIntent,
    ReactToDesiredOutcomeIntent,
    ReactToEstimateCostIntent,
    ReactToUpdateEntityIntent,
    ReadPriceRequest,
    RecognizeIntent,
    RespondUnknown,
    SayGoodbye,
)
from logistics_agent.domain import At, DriveLongDistanceAction, GasPrice, Package, Truck, UnloadAction
from logistics_agent.extract_and_set_current import (
    ASK_WHERE,
    AskWhere,
    CurrentState,
    ExtractAndSetCurrent,
    ExtractCurrentLocation,
    ResolveCurrent,
    SetCurrent,
)
from logistics_agent.extract_and_set_goal import (
    ASK_AGAIN,
    AskAgain,
    ExtractAndSetGoal,
    ExtractOutcome,
    GoalState,
    ResolveGoal,
    SetGoal,
)
from logistics_agent.planner import LogisticsPlanner
from logistics_agent.reasoner import Reasoner

PRICE = "how much to ship a package from seattle to san francisco?"
PRICE_AGAIN = "how much would that cost?"
GIBBERISH = "what is the capital of france?"
HELLO = "hello there"
CAPABILITIES = "what can you do?"

OUTCOME = "I want my crate of machine parts to end up in denver"
TWO_OUTCOMES = "the crate should go to the san francisco office, and the pelican should end at SFO"
VAGUE_OUTCOME = "I want my crate delivered somewhere nice"
UNKNOWN_OUTCOME = "send the crate to narnia"
IMPOSSIBLE_OUTCOME = "put the denver office inside big bertha"
GARBLED_OUTCOME = "crate... uh... the place with the mountains"

CRATE_IN_SEATTLE = "the crate is at the seattle office"
CRATE_SOMEWHERE = "the crate is somewhere around"
CRATE_IN_NARNIA = "the crate is in narnia"
GARBLED_LOCATION = "the crate... it's, you know, there"

ASK_TRUCKS = "which trucks do we have?"
ASK_VEHICLES_AND_TRUCKS = "which vehicles and trucks do we have?"
ASK_AIRPLANES = "which airplanes do we have?"
ASK_UNICORNS = "which unicorns do we have?"

WHERE_BERTHA = "where is big bertha currently?"
WHAT_AT_SEA = "what is at the seattle airport?"
WHAT_IN_PELICAN = "what is in the pelican?"
WHERE_BOOKS = "where is the box of books?"
WHERE_VAGUE = "where is it?"

ADD_BOOKS = "add a 200 lb package of books at the boston office"
ADD_OFFICE = "add an office in chicago"
ADD_SOMETHING = "add something"
ADD_NOWHERE = "add a 200 lb package of books"
ADD_NARNIA = "add a 200 lb package of books in narnia"
ADD_VAGUE = "add a package at the boston office"
ADD_TRUCK_AT_GAS_PRICE = "add a truck at the gas price"

UPDATE_GAS = "set the gas price to 0.95 per mile"
UPDATE_GAS_SAME = "set the gas price to 0.8 per mile"
UPDATE_CRATE = "the crate now weighs 500 lb"
UPDATE_BERTHA = "big bertha can now carry 12000 lb"
UPDATE_SOMETHING = "change something"
UPDATE_UNICORN = "make the unicorn faster"
UPDATE_VAGUE = "change the gas price"

BOOKS = Package(name="made_up_by_the_llm", display_name="box of books", weight=200)

INTENTS = {
    PRICE: HumanIntent.GET_PRICE_ESTIMATE,
    PRICE_AGAIN: HumanIntent.GET_PRICE_ESTIMATE,
    **dict.fromkeys((CRATE_IN_SEATTLE, CRATE_SOMEWHERE, CRATE_IN_NARNIA, GARBLED_LOCATION),
                    HumanIntent.STATE_CURRENT_LOCATION),
    GIBBERISH: None,
    HELLO: HumanIntent.GREETING,
    CAPABILITIES: HumanIntent.ASK_CAPABILITIES,
    **dict.fromkeys((OUTCOME, TWO_OUTCOMES, VAGUE_OUTCOME, UNKNOWN_OUTCOME, IMPOSSIBLE_OUTCOME, GARBLED_OUTCOME),
                    HumanIntent.STATE_DESIRED_OUTCOME),
    **dict.fromkeys((ASK_TRUCKS, ASK_VEHICLES_AND_TRUCKS, ASK_AIRPLANES, ASK_UNICORNS), HumanIntent.ASK_ABOUT_ENTITIES),
    **dict.fromkeys((WHERE_BERTHA, WHAT_AT_SEA, WHAT_IN_PELICAN, WHERE_BOOKS, WHERE_VAGUE), HumanIntent.ASK_CURRENT_STATE),
    **dict.fromkeys((ADD_BOOKS, ADD_OFFICE, ADD_SOMETHING, ADD_NOWHERE, ADD_NARNIA, ADD_VAGUE, ADD_TRUCK_AT_GAS_PRICE),
                    HumanIntent.ADD_ENTITY),
    **dict.fromkeys((UPDATE_GAS, UPDATE_GAS_SAME, UPDATE_CRATE, UPDATE_BERTHA, UPDATE_SOMETHING, UPDATE_UNICORN,
                     UPDATE_VAGUE),
                    HumanIntent.UPDATE_ENTITY),
}

# what each populator returns, keyed by (model type, phrase); an exception is raised instead of returned
POPULATED = {
    (CurrentLocation, PRICE): CurrentLocation(facts=[AtFact(entity1="package1", entity2="office_seattle")]),
    (DesiredOutcome, PRICE): DesiredOutcome(facts=[AtFact(entity1="package1", entity2="office_san_francisco")]),
    (CurrentLocation, PRICE_AGAIN): CurrentLocation(facts=[]),
    (DesiredOutcome, PRICE_AGAIN): DesiredOutcome(facts=[]),

    (CurrentLocation, CRATE_IN_SEATTLE): CurrentLocation(facts=[AtFact(entity1="package1", entity2="office_seattle")]),
    (CurrentLocation, CRATE_SOMEWHERE): CurrentLocation(facts=[]),
    (CurrentLocation, CRATE_IN_NARNIA): CurrentLocation(facts=[AtFact(entity1="package1", entity2="narnia")]),
    (CurrentLocation, GARBLED_LOCATION): UnexpectedModelBehavior("could not fill in the location"),

    (DesiredOutcome, OUTCOME): DesiredOutcome(facts=[AtFact(entity1="package1", entity2="office_denver")]),
    (DesiredOutcome, TWO_OUTCOMES): DesiredOutcome(facts=[
        AtFact(entity1="package1", entity2="office_san_francisco"), AtFact(entity1="air1", entity2="SFO"),
    ]),
    (DesiredOutcome, VAGUE_OUTCOME): DesiredOutcome(facts=[]),
    (DesiredOutcome, UNKNOWN_OUTCOME): DesiredOutcome(facts=[AtFact(entity1="package1", entity2="narnia")]),
    (DesiredOutcome, IMPOSSIBLE_OUTCOME): DesiredOutcome(facts=[AtFact(entity1="office_denver", entity2="truck1")]),
    (DesiredOutcome, GARBLED_OUTCOME): UnexpectedModelBehavior("could not fill in the outcome"),

    (EntityQuery, ASK_TRUCKS): EntityQuery(kinds=[EntityKind.TRUCK]),
    (EntityQuery, ASK_VEHICLES_AND_TRUCKS): EntityQuery(kinds=[EntityKind.VEHICLE, EntityKind.TRUCK]),
    (EntityQuery, ASK_AIRPLANES): EntityQuery(kinds=[EntityKind.AIRPLANE]),
    (EntityQuery, ASK_UNICORNS): EntityQuery(kinds=[]),

    (StateQuery, WHERE_BERTHA): StateQuery(entity1="Big Bertha", entity2=None),
    (StateQuery, WHAT_AT_SEA): StateQuery(entity1=None, entity2="SEA"),
    (StateQuery, WHAT_IN_PELICAN): StateQuery(entity1=None, entity2="air1"),
    (StateQuery, WHERE_BOOKS): StateQuery(entity1="box of books", entity2=None),
    (StateQuery, WHERE_VAGUE): StateQuery(entity1=None, entity2=None),

    (NewEntity, ADD_BOOKS): NewEntity(kind=EntityKind.PACKAGE, location="office_boston"),
    (Package, ADD_BOOKS): BOOKS,
    (NewEntity, ADD_OFFICE): NewEntity(kind=EntityKind.OFFICE, location=None),
    (NewEntity, ADD_SOMETHING): NewEntity(kind=None, location=None),
    (NewEntity, ADD_NOWHERE): NewEntity(kind=EntityKind.PACKAGE, location=None),
    (NewEntity, ADD_NARNIA): NewEntity(kind=EntityKind.PACKAGE, location="narnia"),
    (Package, ADD_NARNIA): BOOKS,
    (NewEntity, ADD_VAGUE): NewEntity(kind=EntityKind.PACKAGE, location="office_boston"),
    (Package, ADD_VAGUE): UnexpectedModelBehavior("no weight given"),
    (NewEntity, ADD_TRUCK_AT_GAS_PRICE): NewEntity(kind=EntityKind.TRUCK, location="gas_price"),
    (Truck, ADD_TRUCK_AT_GAS_PRICE): Truck(name="t", display_name="Tiny", model="Ford Transit", max_payload=3000,
                                           license_plate="X1"),

    (EntityChange, UPDATE_GAS): EntityChange(entity="gas_price"),
    (GasPrice, UPDATE_GAS): GasPrice(name="made_up_by_the_llm", per_mile=0.95),
    (EntityChange, UPDATE_GAS_SAME): EntityChange(entity="gas_price"),
    (GasPrice, UPDATE_GAS_SAME): GasPrice(name="gas_price", per_mile=0.8),
    (EntityChange, UPDATE_CRATE): EntityChange(entity="crate of machine parts"),
    (Package, UPDATE_CRATE): Package(name="package1", display_name="crate of machine parts", weight=500),
    (EntityChange, UPDATE_BERTHA): EntityChange(entity="Big Bertha"),
    (Truck, UPDATE_BERTHA): Truck(name="truck1", display_name="Big Bertha", model="Freightliner M2 106 26ft box truck",
                                  max_payload=12000, license_plate="C12345A"),
    (EntityChange, UPDATE_SOMETHING): EntityChange(entity=None),
    (EntityChange, UPDATE_UNICORN): EntityChange(entity="unicorn"),
    (EntityChange, UPDATE_VAGUE): EntityChange(entity="gas_price"),
    (GasPrice, UPDATE_VAGUE): UnexpectedModelBehavior("no new value given"),
}


class FakeEnumClassifier:
    """stands in for EnumClassifier: canned intents, keyed by phrase; records how it was built and called"""

    def __init__(self, enum_type, task_desc: str) -> None:
        self.enum_type, self.task_desc = enum_type, task_desc
        self.calls: list[tuple] = []

    def __call__(self, phrase: str, llm, **kwargs) -> tuple[HumanIntent | None, float]:
        self.calls.append((phrase, llm, kwargs))
        return INTENTS[phrase], 1.0


class FakeModelPopulator:
    """stands in for ModelPopulator: canned models, keyed by model type and phrase; records how it was built and
    called"""

    def __init__(self, model_type, task_desc: str) -> None:
        self.model_type, self.task_desc = model_type, task_desc
        self.calls: list[tuple] = []

    def __call__(self, phrase: str, llm, *extra: str, **kwargs):
        self.calls.append((phrase, llm, extra))
        result = POPULATED[self.model_type, phrase]
        if isinstance(result, Exception):
            raise result
        return result


DESCRIBED: list[tuple[list, str]] = []


def fake_describe_facts(facts, task_desc, _llm) -> str:
    """describes facts by how many there are and the task description; records every set of facts and task
    description it was given"""
    DESCRIBED.append((list(facts), task_desc))
    return f"described {len(DESCRIBED[-1][0])} facts: {task_desc}"


@pytest.fixture(autouse=True)
def fake_language(monkeypatch):
    """replaces the LLM-backed EnumClassifier, ModelPopulator and describe_facts used by the operators"""
    DESCRIBED.clear()
    monkeypatch.setattr(conductor, "EnumClassifier", FakeEnumClassifier)
    monkeypatch.setattr(conductor, "ModelPopulator", FakeModelPopulator)
    monkeypatch.setattr(extract_and_set_goal, "ModelPopulator", FakeModelPopulator)
    monkeypatch.setattr(extract_and_set_current, "ModelPopulator", FakeModelPopulator)
    monkeypatch.setattr(conductor, "describe_facts", fake_describe_facts)


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


def test_price_estimate_describes_the_plan_and_cost(chat):
    _, reply, goodbye = chat(PRICE, "quit")
    [(actions, task_desc)] = DESCRIBED
    assert len(actions) == 4
    [drive] = [a for a in actions if isinstance(a, DriveLongDistanceAction)]
    assert (drive.from_location.name, drive.to_location.name) == ("office_seattle", "office_san_francisco")
    assert "costs 830.0 in total" in task_desc
    assert reply == f"described 4 facts: {task_desc}"
    assert goodbye == "Goodbye!"


def test_keeps_answering_after_each_reply(chat):
    replies = chat(GIBBERISH, PRICE, GIBBERISH, PRICE, "quit")
    assert replies[1] == replies[3] == "Sorry, I am not designed to respond to that."
    assert replies[2] == replies[4]
    assert "costs 830.0 in total" in replies[2]
    assert replies[-1] == "Goodbye!"


def test_empty_lines_are_ignored(chat):
    assert chat("", "   ", GIBBERISH, "quit") == [
        INTRODUCTION, "Sorry, I am not designed to respond to that.", "Goodbye!"
    ]


def test_price_request_naming_where_from_and_where_to_is_answered_in_one_turn(chat):
    replies = chat(PRICE, "quit")
    assert "costs 830.0 in total" in replies[1]


def test_price_request_asks_for_where_from_then_where_to_then_estimates(chat):
    """seattle office to denver office: truck to the office 50, load 60, drive 1300 miles 1040, unload 50"""
    replies = chat(PRICE_AGAIN, CRATE_IN_SEATTLE, PRICE_AGAIN, OUTCOME, PRICE_AGAIN, "quit")
    assert replies[1] == "Where is the package now? Tell me, and then ask me for the estimate again."
    assert replies[2] == ("Got it: the crate of machine parts is at the Seattle Office. "
                          "Ask me for a price estimate whenever you are ready.")
    assert replies[3] == "Where should the package go? Tell me, and then ask me for the estimate again."
    assert replies[4] == ("Got it: the crate of machine parts at the Denver Office. "
                          "Ask me for a price estimate whenever you are ready.")
    assert "costs 1200.0 in total" in replies[5]


def test_price_request_without_a_destination_does_not_plan(chat):
    replies = chat(CRATE_IN_SEATTLE, PRICE_AGAIN, "quit")
    assert replies[2] == "Where should the package go? Tell me, and then ask me for the estimate again."
    assert not DESCRIBED


def test_desired_outcome_is_confirmed_and_used_by_the_estimate(chat):
    _, _, confirmation, _, _ = chat(CRATE_IN_SEATTLE, OUTCOME, PRICE_AGAIN, "quit")
    assert confirmation == ("Got it: the crate of machine parts at the Denver Office. "
                            "Ask me for a price estimate whenever you are ready.")
    [(actions, _)] = DESCRIBED
    unload = actions[-1]
    assert isinstance(unload, UnloadAction)
    assert (unload.package.name, unload.location.name) == ("package1", "office_denver")


def test_vague_outcome_keeps_the_goal_the_estimate_uses(chat):
    chat(PRICE, VAGUE_OUTCOME, PRICE_AGAIN, "quit")
    [(_, _), (actions, _)] = DESCRIBED
    assert actions[-1].location.name == "office_san_francisco"


def test_a_follow_up_price_request_keeps_where_from_and_where_to(chat):
    replies = chat(PRICE, PRICE_AGAIN, "quit")
    assert replies[1] == replies[2]


def test_an_added_package_can_be_asked_about(chat):
    replies = chat(ADD_BOOKS, WHERE_BOOKS, "quit")
    assert replies[1] == "Added the box of books (package) at the Boston Office."
    [(facts, _)] = DESCRIBED
    assert [(f.entity1.name, f.entity2.name) for f in facts] == [("package2", "office_boston")]


def test_an_updated_gas_price_changes_the_estimate(chat):
    """at 0.95 per mile the 800 mile drive costs 760, so driving (950 in total) loses to flying (879)"""
    replies = chat(PRICE, UPDATE_GAS, PRICE, "quit")
    assert "costs 830.0 in total" in replies[1]
    assert replies[2] == "Updated the gas price: per_mile is now 0.95."
    assert "costs 879.0 in total" in replies[3]


@pytest.mark.xfail(strict=True, reason="no operator answers greetings or capability questions yet")
@pytest.mark.parametrize("phrase", [HELLO, CAPABILITIES])
def test_known_intents_without_an_operator_still_get_a_reply(chat, phrase):
    replies = chat(phrase, GIBBERISH, "quit")
    assert replies[2] == "Sorry, I am not designed to respond to that."


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


def heard(phrase: str) -> ConversationState:
    """a conversation state in which the phrase has been heard and its intent recognized"""
    return ConversationState(human_utterance=Utterance(phrase), human_intent=INTENTS[phrase])


def react(op_type: type, phrase: str, state: ConversationState | None = None) -> tuple[ConversationState, str, object]:
    """has a fresh operator of the type react to the phrase; checks that it applies and that it clears the utterance,
    and returns the state, the one reply, and the operator"""
    state = state or heard(phrase)
    state.human_utterance, state.human_intent = Utterance(phrase), INTENTS[phrase]
    op, io = op_type("op"), RecordingIO()
    assert op.can_perform(state, io)
    op.perform(state, io)
    assert state.human_utterance is None and state.human_intent is None
    [reply] = io.said
    return state, reply, op


def current_of(state: ConversationState) -> list[tuple[str, str]] | None:
    if state.planner.current is None:
        return None
    return [(fact.entity1.name, fact.entity2.name) for fact in state.planner.current]


def package_in_seattle(state: ConversationState) -> ConversationState:
    """sets the planner's current facts to package1 at the Seattle office"""
    find = state.reasoner.find_entity
    state.planner.set_current([At(entity1=find("package1"), entity2=find("office_seattle"))])
    return state


def to_san_francisco(state: ConversationState) -> ConversationState:
    """sets the planner's goal to package1 at the San Francisco office"""
    find = state.reasoner.find_entity
    state.planner.set_goal([At(entity1=find("package1"), entity2=find("office_san_francisco"))])
    return state


def read_price_request(phrase: str) -> ConversationState:
    """a conversation state in which the price request has been heard and read, for the package going from Seattle
    to San Francisco"""
    state = to_san_francisco(package_in_seattle(heard(phrase)))
    state.price_request_read = True
    return state


def goal_of(state: ConversationState) -> list[tuple[str, str]]:
    return [(fact.entity1.name, fact.entity2.name) for fact in state.planner.goal]


def location_of(state: ConversationState, name: str) -> str:
    [at] = state.reasoner.find_relations(At, name)
    return at.entity2.name


# --- conversation state


def test_planner_plans_over_the_reasoners_world_graph():
    state = ConversationState()
    assert state.planner.world is state.reasoner.world


def test_planner_starts_with_no_goal_and_no_current_facts():
    state = ConversationState()
    assert state.planner.goal is None and state.planner.current is None


# --- introduce and say goodbye


def test_introduce_applies_only_once():
    state, io = ConversationState(), RecordingIO()
    op = Introduce("introduce")
    assert op.can_perform(state, io)
    op.perform(state, io)
    assert io.said == [INTRODUCTION] and state.introduced
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


# --- recognize intent


def test_recognize_intent_stores_utterance_and_intent():
    state, io = ConversationState(), RecordingIO(PRICE)
    op = RecognizeIntent("recognize_intent")
    assert op.can_perform(state, io)
    op.perform(state, io)
    assert state.human_utterance == Utterance(PRICE)
    assert state.human_intent is HumanIntent.GET_PRICE_ESTIMATE


def test_recognize_intent_classifies_over_the_known_intents():
    op = RecognizeIntent("recognize_intent")
    op.perform(ConversationState(), RecordingIO(GIBBERISH))
    classifier = op._intent_classifier
    assert classifier.enum_type is HumanIntent
    assert classifier.task_desc == RecognizeIntent.TASK_DESC
    assert classifier.calls == [(GIBBERISH, conductor.DEFAULT_LLM, {"num_trials": 1})]


@pytest.mark.parametrize("phrase", [None, "quit", "ok bye now", "EXIT please"])
def test_recognize_intent_skips_silence_and_quitting(phrase):
    op = RecognizeIntent("recognize_intent")
    assert not op.can_perform(ConversationState(), RecordingIO(phrase))


def test_recognize_intent_waits_until_the_current_utterance_is_answered():
    op = RecognizeIntent("recognize_intent")
    assert not op.can_perform(heard(PRICE), RecordingIO(GIBBERISH))


# --- operators apply only to their own intent


REACTIONS = {
    ReadPriceRequest: HumanIntent.GET_PRICE_ESTIMATE,
    ReactToEstimateCostIntent: HumanIntent.GET_PRICE_ESTIMATE,
    ReactToDesiredOutcomeIntent: HumanIntent.STATE_DESIRED_OUTCOME,
    ReactToCurrentLocationIntent: HumanIntent.STATE_CURRENT_LOCATION,
    ReactToAskAboutEntitiesIntent: HumanIntent.ASK_ABOUT_ENTITIES,
    ReactToAskCurrentStateIntent: HumanIntent.ASK_CURRENT_STATE,
    ReactToAddEntityIntent: HumanIntent.ADD_ENTITY,
    ReactToUpdateEntityIntent: HumanIntent.UPDATE_ENTITY,
}


@pytest.mark.parametrize("op_type", list(REACTIONS))
@pytest.mark.parametrize("intent", [*HumanIntent, None])
def test_each_reaction_applies_only_to_its_intent(op_type, intent):
    """with the delivery known and the price request read (for every operator but the one that reads it), only the
    intent decides whether an operator applies"""
    state = to_san_francisco(package_in_seattle(ConversationState(human_utterance=Utterance(GIBBERISH),
                                                                  human_intent=intent)))
    state.price_request_read = op_type is not ReadPriceRequest
    assert op_type("op").can_perform(state, RecordingIO()) == (REACTIONS[op_type] is intent)


# --- respond unknown


@pytest.mark.parametrize("intent", list(HumanIntent))
def test_respond_unknown_ignores_known_intents(intent):
    state = ConversationState(human_utterance=Utterance(PRICE), human_intent=intent)
    assert not RespondUnknown("respond_unknown").can_perform(state, RecordingIO())


def test_respond_unknown_waits_for_an_utterance():
    assert not RespondUnknown("respond_unknown").can_perform(ConversationState(), RecordingIO())


def test_respond_unknown_clears_the_utterance():
    _, reply, _ = react(RespondUnknown, GIBBERISH)
    assert reply == "Sorry, I am not designed to respond to that."


# --- estimate cost


def test_read_price_request_sets_where_from_and_where_to_and_says_nothing():
    state, io = heard(PRICE), RecordingIO()
    op = ReadPriceRequest("op")
    assert op.can_perform(state, io)
    op.perform(state, io)
    assert current_of(state) == [("package1", "office_seattle")]
    assert goal_of(state) == [("package1", "office_san_francisco")]
    assert state.price_request_read and not io.said
    assert state.human_utterance == Utterance(PRICE) and state.human_intent is HumanIntent.GET_PRICE_ESTIMATE
    assert not op.can_perform(state, io)


def test_read_price_request_keeps_what_the_request_does_not_mention():
    state = package_in_seattle(heard(PRICE_AGAIN))
    state.planner.set_goal([At(entity1=state.reasoner.find_entity("package1"),
                               entity2=state.reasoner.find_entity("office_denver"))])
    ReadPriceRequest("op").perform(state, RecordingIO())
    assert current_of(state) == [("package1", "office_seattle")]
    assert goal_of(state) == [("package1", "office_denver")]


def test_read_price_request_sets_up_its_sub_processes_once():
    op = ReadPriceRequest("op")
    current, goal = op._extract_and_set_current, op._extract_and_set_goal
    assert isinstance(current, ExtractAndSetCurrent) and isinstance(goal, ExtractAndSetGoal)
    op.perform(heard(PRICE), RecordingIO())
    op.perform(heard(PRICE_AGAIN), RecordingIO())
    assert op._extract_and_set_current is current and op._extract_and_set_goal is goal


def test_estimate_waits_until_the_request_is_read():
    state = read_price_request(PRICE)
    state.price_request_read = False
    assert not ReactToEstimateCostIntent("op").can_perform(state, RecordingIO())


def test_estimate_waits_for_the_planners_goal():
    state = read_price_request(PRICE)
    state.planner.goal = None
    assert not ReactToEstimateCostIntent("op").can_perform(state, RecordingIO())


def test_estimate_waits_for_the_planners_current_facts():
    state = read_price_request(PRICE)
    state.planner.current = None
    assert not ReactToEstimateCostIntent("op").can_perform(state, RecordingIO())


def test_estimate_plans_from_current_toward_the_goal_without_changing_them():
    state, reply, _ = react(ReactToEstimateCostIntent, PRICE, read_price_request(PRICE))
    [(actions, task_desc)] = DESCRIBED
    assert actions[-1].location.name == "office_san_francisco"
    assert reply == f"described 4 facts: {task_desc}"
    assert goal_of(state) == [("package1", "office_san_francisco")]
    assert current_of(state) == [("package1", "office_seattle")]
    assert not state.price_request_read


def test_estimate_without_a_plan_says_so():
    state = read_price_request(PRICE)
    for vehicle in ("truck1", "truck2", "air1"):
        state.reasoner.world.remove_node(vehicle)
    state, reply, _ = react(ReactToEstimateCostIntent, PRICE, state)
    assert reply == "Sorry, I could not find a way to deliver that with the vehicles we have."
    assert not DESCRIBED and not state.price_request_read


# --- ask where the package is


def test_ask_where_the_package_is_applies_only_to_a_read_request_without_current_facts():
    op, state = AskWhereThePackageIs("op"), read_price_request(PRICE)
    assert not op.can_perform(state, RecordingIO())
    state.planner.current = None
    assert op.can_perform(state, RecordingIO())
    state.price_request_read = False
    assert not op.can_perform(state, RecordingIO())


def test_ask_where_the_package_is_asks_and_clears_the_request():
    state = read_price_request(PRICE)
    state.planner.current = None
    state, reply, _ = react(AskWhereThePackageIs, PRICE, state)
    assert reply == "Where is the package now? Tell me, and then ask me for the estimate again."
    assert not state.price_request_read


# --- ask where the package goes


def test_ask_where_the_package_goes_applies_only_once_where_it_is_is_known_and_where_it_goes_is_not():
    op, state = AskWhereThePackageGoes("op"), read_price_request(PRICE)
    assert not op.can_perform(state, RecordingIO())
    state.planner.goal = None
    assert op.can_perform(state, RecordingIO())
    state.planner.current = None
    assert not op.can_perform(state, RecordingIO())
    assert AskWhereThePackageIs("op").can_perform(state, RecordingIO())


def test_ask_where_the_package_goes_asks_and_clears_the_request():
    state = read_price_request(PRICE)
    state.planner.goal = None
    state, reply, _ = react(AskWhereThePackageGoes, PRICE, state)
    assert reply == "Where should the package go? Tell me, and then ask me for the estimate again."
    assert not state.price_request_read


# --- current location


def test_current_location_sets_the_planners_current_facts():
    state, reply, _ = react(ReactToCurrentLocationIntent, CRATE_IN_SEATTLE)
    assert current_of(state) == [("package1", "office_seattle")]
    assert reply == ("Got it: the crate of machine parts is at the Seattle Office. "
                     "Ask me for a price estimate whenever you are ready.")


@pytest.mark.parametrize("phrase", [CRATE_SOMEWHERE, CRATE_IN_NARNIA, GARBLED_LOCATION])
def test_current_location_it_cannot_use_asks_where_and_keeps_the_current_facts(phrase):
    state, reply, _ = react(ReactToCurrentLocationIntent, phrase, package_in_seattle(heard(phrase)))
    assert reply == ASK_WHERE
    assert current_of(state) == [("package1", "office_seattle")]


def test_current_location_sets_up_its_sub_process_once():
    op = ReactToCurrentLocationIntent("op")
    sub_process = op._extract_and_set_current
    assert isinstance(sub_process, ExtractAndSetCurrent)
    op.perform(heard(CRATE_IN_SEATTLE), RecordingIO())
    assert op._extract_and_set_current is sub_process


# --- desired outcome


def test_desired_outcome_sets_up_its_sub_process_once_and_restarts_it_for_every_outcome():
    op = ReactToDesiredOutcomeIntent("op")
    sub_process = op._extract_and_set_goal
    assert isinstance(sub_process, ExtractAndSetGoal)
    state = heard(OUTCOME)
    op.perform(state, RecordingIO())
    state.human_utterance, state.human_intent = Utterance(VAGUE_OUTCOME), INTENTS[VAGUE_OUTCOME]
    op.perform(state, RecordingIO())
    assert op._extract_and_set_goal is sub_process
    assert sub_process.state.reply == ASK_AGAIN and sub_process.state.desired_outcome.facts == []


def test_desired_outcome_the_sub_process_cannot_finish_asks_again_and_keeps_the_goal():
    state, reply, _ = react(ReactToDesiredOutcomeIntent, GARBLED_OUTCOME, to_san_francisco(heard(GARBLED_OUTCOME)))
    assert reply == ASK_AGAIN
    assert goal_of(state) == [("package1", "office_san_francisco")]


def test_desired_outcome_sets_the_planners_goal():
    state, reply, _ = react(ReactToDesiredOutcomeIntent, OUTCOME)
    assert goal_of(state) == [("package1", "office_denver")]
    assert reply == ("Got it: the crate of machine parts at the Denver Office. "
                     "Ask me for a price estimate whenever you are ready.")


def test_desired_outcome_with_several_facts_sets_them_all():
    state, reply, _ = react(ReactToDesiredOutcomeIntent, TWO_OUTCOMES)
    assert goal_of(state) == [("package1", "office_san_francisco"), ("air1", "SFO")]
    assert reply.startswith("Got it: the crate of machine parts at the San Francisco Office; "
                            "the The Pelican at the San Francisco International Airport.")


@pytest.mark.parametrize("phrase", [VAGUE_OUTCOME, UNKNOWN_OUTCOME, IMPOSSIBLE_OUTCOME])
def test_desired_outcome_it_cannot_use_asks_again_and_keeps_the_goal(phrase):
    state, reply, _ = react(ReactToDesiredOutcomeIntent, phrase, to_san_francisco(heard(phrase)))
    assert reply == "Sorry, I did not catch where things should end up. Where should the package go?"
    assert goal_of(state) == [("package1", "office_san_francisco")]


# --- ask about entities


def test_ask_about_entities_describes_the_entities_of_the_kinds_asked_about():
    _, reply, op = react(ReactToAskAboutEntitiesIntent, ASK_TRUCKS)
    [(entities, task_desc)] = DESCRIBED
    assert {e.name for e in entities} == {"truck1", "truck2"}
    assert ASK_TRUCKS in task_desc
    assert reply == f"described 2 facts: {task_desc}"
    assert op._query_populator.model_type is EntityQuery


def test_ask_about_overlapping_kinds_describes_each_entity_once():
    react(ReactToAskAboutEntitiesIntent, ASK_VEHICLES_AND_TRUCKS)
    [(entities, _)] = DESCRIBED
    assert sorted(e.name for e in entities) == ["air1", "truck1", "truck2"]


def test_ask_about_entities_of_a_kind_with_none_says_so():
    state = heard(ASK_AIRPLANES)
    state.reasoner.world.remove_node("air1")
    _, reply, _ = react(ReactToAskAboutEntitiesIntent, ASK_AIRPLANES, state)
    assert reply == "We do not have any of those right now."
    assert not DESCRIBED


def test_ask_about_no_known_kind_lists_what_it_can_answer():
    _, reply, _ = react(ReactToAskAboutEntitiesIntent, ASK_UNICORNS)
    assert reply.startswith("Sorry, I did not catch what you are asking about.")
    assert not DESCRIBED


# --- ask current state


def test_ask_where_something_is_describes_its_at_relation():
    state, reply, op = react(ReactToAskCurrentStateIntent, WHERE_BERTHA)
    [(facts, task_desc)] = DESCRIBED
    assert [(f.entity1.name, f.entity2.name) for f in facts] == [("truck1", "SEA")]
    assert WHERE_BERTHA in task_desc
    assert reply == f"described 1 facts: {task_desc}"
    assert op._query_populator.calls[0][2] == tuple(known_entities(state.reasoner.world))


def test_ask_what_is_at_a_place_describes_everything_there():
    react(ReactToAskCurrentStateIntent, WHAT_AT_SEA)
    [(facts, _)] = DESCRIBED
    assert {(f.entity1.name, f.entity2.name) for f in facts} == {("truck1", "SEA"), ("air1", "SEA")}


def test_ask_about_an_empty_place_says_nothing_matches():
    _, reply, _ = react(ReactToAskCurrentStateIntent, WHAT_IN_PELICAN)
    assert reply == "Nothing matches that right now."
    assert not DESCRIBED


def test_ask_current_state_without_naming_anything_asks_again():
    _, reply, _ = react(ReactToAskCurrentStateIntent, WHERE_VAGUE)
    assert reply == "Sorry, I did not catch which package, vehicle, or place you are asking about."
    assert not DESCRIBED


# --- add entity


def test_add_entity_adds_it_under_a_fresh_name_at_the_location():
    state, reply, op = react(ReactToAddEntityIntent, ADD_BOOKS)
    assert reply == "Added the box of books (package) at the Boston Office."
    books = state.reasoner.find_entity("package2")
    assert (books.display_name, books.weight) == ("box of books", 200)
    assert location_of(state, "package2") == "office_boston"
    assert state.reasoner.find_entity("made_up_by_the_llm") is None
    assert op._details_populators[EntityKind.PACKAGE].model_type is Package


def test_add_entity_reuses_the_details_populator_for_a_kind():
    state = heard(ADD_BOOKS)
    op, io = ReactToAddEntityIntent("op"), RecordingIO()
    op.perform(state, io)
    populator = op._details_populators[EntityKind.PACKAGE]
    state.human_utterance, state.human_intent = Utterance(ADD_BOOKS), INTENTS[ADD_BOOKS]
    op.perform(state, io)
    assert op._details_populators[EntityKind.PACKAGE] is populator and len(populator.calls) == 2
    assert {location_of(state, name) for name in ("package2", "package3")} == {"office_boston"}


@pytest.mark.parametrize("phrase", [ADD_OFFICE, ADD_SOMETHING])
def test_add_entity_of_a_kind_it_cannot_add_says_so(phrase):
    state, reply, _ = react(ReactToAddEntityIntent, phrase)
    assert reply == "Sorry, I can only add packages, trucks, and airplanes."


def test_add_entity_without_a_location_asks_for_one():
    state, reply, _ = react(ReactToAddEntityIntent, ADD_NOWHERE)
    assert reply == ("Where is the new package? Please tell me again, including the office, airport, or vehicle "
                     "it is at.")
    assert state.reasoner.find_entity("package2") is None


def test_add_entity_at_an_unknown_location_says_so():
    state, reply, _ = react(ReactToAddEntityIntent, ADD_NARNIA)
    assert reply == "Sorry, I do not know of a place called narnia."
    assert state.reasoner.find_entity("package2") is None


@pytest.mark.parametrize("phrase, kind", [(ADD_VAGUE, "package"), (ADD_TRUCK_AT_GAS_PRICE, "truck")])
def test_add_entity_it_cannot_fill_in_or_place_asks_for_the_details(phrase, kind):
    state, reply, _ = react(ReactToAddEntityIntent, phrase)
    assert reply.startswith(f"Sorry, I could not add that {kind}. Please tell me its display_name, ")
    assert state.reasoner.find_entity(f"{kind}3" if kind == "truck" else "package2") is None


# --- update entity


def test_update_entity_changes_it_and_says_what_changed():
    state, reply, op = react(ReactToUpdateEntityIntent, UPDATE_GAS)
    assert reply == "Updated the gas price: per_mile is now 0.95."
    assert state.reasoner.find_entity("gas_price").per_mile == 0.95
    assert state.reasoner.find_entity("made_up_by_the_llm") is None
    populator = op._details_populators[GasPrice]
    [(_, _, extra)] = populator.calls
    assert extra == ('current values: {"name":"gas_price","per_mile":0.8}',)


def test_update_entity_reports_the_changed_field():
    state, reply, _ = react(ReactToUpdateEntityIntent, UPDATE_CRATE)
    assert reply == "Updated the crate of machine parts: weight is now 500.0."
    assert state.reasoner.find_entity("package1").weight == 500


def test_update_entity_keeps_its_location():
    state, reply, _ = react(ReactToUpdateEntityIntent, UPDATE_BERTHA)
    assert reply == "Updated the Big Bertha: max_payload is now 12000.0."
    assert location_of(state, "truck1") == "SEA"


def test_update_entity_with_the_same_values_says_nothing_changed():
    _, reply, _ = react(ReactToUpdateEntityIntent, UPDATE_GAS_SAME)
    assert reply == "The gas price already has those values; nothing changed."


@pytest.mark.parametrize("phrase", [UPDATE_SOMETHING, UPDATE_UNICORN])
def test_update_entity_it_cannot_find_asks_again(phrase):
    _, reply, _ = react(ReactToUpdateEntityIntent, phrase)
    assert reply == "Sorry, I did not catch what you want to change."


def test_update_entity_it_cannot_fill_in_leaves_it_unchanged():
    state, reply, _ = react(ReactToUpdateEntityIntent, UPDATE_VAGUE)
    assert reply == "Sorry, I could not work out the new values for the gas price."
    assert state.reasoner.find_entity("gas_price").per_mile == 0.8


# --- extract and set goal sub-process


def sub_process_io(phrase: str | None = None) -> SimpleNamespace:
    """stands in for the sub-process's io container: io.a holds the arguments the parent operator supplies"""
    reasoner = Reasoner()
    return SimpleNamespace(a=SimpleNamespace(
        outcome=Utterance(phrase), reasoner=reasoner, planner=LogisticsPlanner(reasoner.world)
    ))


def run_sub_process(phrase: str) -> tuple[GoalState | None, LogisticsPlanner]:
    """runs a fresh ExtractAndSetGoal on the phrase, as the parent operator does; returns its final state and the
    planner it was given"""
    reasoner = Reasoner()
    planner = LogisticsPlanner(reasoner.world)
    return ExtractAndSetGoal()(outcome=Utterance(phrase), reasoner=reasoner, planner=planner), planner


def test_sub_process_sets_the_goal_and_words_the_confirmation():
    result, planner = run_sub_process(OUTCOME)
    assert result.desired_outcome == POPULATED[DesiredOutcome, OUTCOME]
    assert [(f.entity1.name, f.entity2.name) for f in result.goal] == [("package1", "office_denver")]
    assert planner.goal == tuple(result.goal)
    assert result.reply == ("Got it: the crate of machine parts at the Denver Office. "
                            "Ask me for a price estimate whenever you are ready.")


@pytest.mark.parametrize("phrase", [VAGUE_OUTCOME, UNKNOWN_OUTCOME, IMPOSSIBLE_OUTCOME])
def test_sub_process_asks_again_and_leaves_the_goal_unset_when_the_outcome_does_not_resolve(phrase):
    result, planner = run_sub_process(phrase)
    assert result.goal == [] and result.reply == ASK_AGAIN
    assert planner.goal is None


def test_sub_process_returns_none_when_an_operator_fails():
    result, _ = run_sub_process(GARBLED_OUTCOME)
    assert result is None


def test_sub_process_can_be_restarted_for_another_outcome():
    sub_process = ExtractAndSetGoal()
    reasoner = Reasoner()
    planner = LogisticsPlanner(reasoner.world)
    sub_process(outcome=Utterance(OUTCOME), reasoner=reasoner, planner=planner)
    sub_process.reinit()
    result = sub_process(outcome=Utterance(TWO_OUTCOMES), reasoner=reasoner, planner=planner)
    assert [(f.entity1.name, f.entity2.name) for f in planner.goal] == [("package1", "office_san_francisco"), ("air1", "SFO")]
    assert result.reply.startswith("Got it: the crate of machine parts at the San Francisco Office; ")


def test_extract_outcome_populates_over_the_outcome_model_with_the_known_entities():
    op, io = ExtractOutcome("extract_outcome"), sub_process_io(OUTCOME)
    state = GoalState()
    assert op.can_perform(state, io)
    op.perform(state, io)
    populator = op._outcome_populator
    assert populator.model_type is DesiredOutcome and populator.task_desc == ExtractOutcome.TASK_DESC
    assert populator.calls == [(OUTCOME, conductor.DEFAULT_LLM, tuple(known_entities(io.a.reasoner.world)))]
    assert not op.can_perform(state, io)


@pytest.mark.parametrize("state, applies", [
    (GoalState(), {ExtractOutcome}),
    (GoalState(desired_outcome=DesiredOutcome(facts=[])), {ResolveGoal}),
    (GoalState(desired_outcome=DesiredOutcome(facts=[]), goal=[]), {AskAgain}),
    (GoalState(desired_outcome=DesiredOutcome(facts=[]), goal=["a goal fact"]), {SetGoal}),
    (GoalState(desired_outcome=DesiredOutcome(facts=[]), goal=[], reply=ASK_AGAIN), set()),
])
def test_exactly_one_sub_process_operator_applies_at_each_step(state, applies):
    io = sub_process_io(OUTCOME)
    ops = {ExtractOutcome: ExtractOutcome("op"), ResolveGoal: ResolveGoal("op"), SetGoal: SetGoal("op"),
           AskAgain: AskAgain("op")}
    assert {op_type for op_type, op in ops.items() if op.can_perform(state, io)} == applies


# --- extract and set current sub-process


def location_io(phrase: str | None = None) -> SimpleNamespace:
    """stands in for the sub-process's io container: io.a holds the arguments the parent operator supplies"""
    reasoner = Reasoner()
    return SimpleNamespace(a=SimpleNamespace(
        location=Utterance(phrase), reasoner=reasoner, planner=LogisticsPlanner(reasoner.world)
    ))


def run_current_sub_process(phrase: str) -> tuple[CurrentState | None, LogisticsPlanner]:
    """runs a fresh ExtractAndSetCurrent on the phrase, as a parent operator does; returns its final state and the
    planner it was given"""
    reasoner = Reasoner()
    planner = LogisticsPlanner(reasoner.world)
    return ExtractAndSetCurrent()(location=Utterance(phrase), reasoner=reasoner, planner=planner), planner


def test_current_sub_process_sets_the_current_facts_and_words_the_confirmation():
    result, planner = run_current_sub_process(CRATE_IN_SEATTLE)
    assert result.current_location == POPULATED[CurrentLocation, CRATE_IN_SEATTLE]
    assert [(f.entity1.name, f.entity2.name) for f in result.current] == [("package1", "office_seattle")]
    assert planner.current == tuple(result.current)
    assert result.reply == ("Got it: the crate of machine parts is at the Seattle Office. "
                            "Ask me for a price estimate whenever you are ready.")


@pytest.mark.parametrize("phrase", [CRATE_SOMEWHERE, CRATE_IN_NARNIA])
def test_current_sub_process_asks_where_and_leaves_the_current_facts_unset(phrase):
    result, planner = run_current_sub_process(phrase)
    assert result.current == [] and result.reply == ASK_WHERE
    assert planner.current is None


def test_current_sub_process_returns_none_when_an_operator_fails():
    result, _ = run_current_sub_process(GARBLED_LOCATION)
    assert result is None


def test_extract_current_location_populates_over_the_location_model_with_the_known_entities():
    op, io = ExtractCurrentLocation("extract_current_location"), location_io(CRATE_IN_SEATTLE)
    state = CurrentState()
    assert op.can_perform(state, io)
    op.perform(state, io)
    populator = op._location_populator
    assert populator.model_type is CurrentLocation and populator.task_desc == ExtractCurrentLocation.TASK_DESC
    assert populator.calls == [(CRATE_IN_SEATTLE, conductor.DEFAULT_LLM, tuple(known_entities(io.a.reasoner.world)))]
    assert not op.can_perform(state, io)


@pytest.mark.parametrize("state, applies", [
    (CurrentState(), {ExtractCurrentLocation}),
    (CurrentState(current_location=CurrentLocation(facts=[])), {ResolveCurrent}),
    (CurrentState(current_location=CurrentLocation(facts=[]), current=[]), {AskWhere}),
    (CurrentState(current_location=CurrentLocation(facts=[]), current=["a fact"]), {SetCurrent}),
    (CurrentState(current_location=CurrentLocation(facts=[]), current=[], reply=ASK_WHERE), set()),
])
def test_exactly_one_current_sub_process_operator_applies_at_each_step(state, applies):
    io = location_io(CRATE_IN_SEATTLE)
    ops = {ExtractCurrentLocation: ExtractCurrentLocation("op"), ResolveCurrent: ResolveCurrent("op"),
           SetCurrent: SetCurrent("op"), AskWhere: AskWhere("op")}
    assert {op_type for op_type, op in ops.items() if op.can_perform(state, io)} == applies

