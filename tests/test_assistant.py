"""offline tests of the pricing support agent's conversation

a fake EnumClassifier, a fake ModelPopulator and a fake describe_facts stand in for the LLM (intents are keyed by phrase,
populated models by model type and phrase, and facts are described by how many there are), and the human's lines are
fed to input() from a script, so the whole perceive-decide-act loop runs deterministically with no
API key. When the script runs out, input() raises EOFError, which the agent treats as quitting.
"""

from collections import deque

import pytest
from pydantic_ai.exceptions import UnexpectedModelBehavior

from logistics_agent import assistant
from logistics_agent import conductor
from logistics_agent.communicator import (
    AtFact,
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
    ConversationState,
    Introduce,
    ReactToAddEntityIntent,
    ReactToAskAboutEntitiesIntent,
    ReactToAskCurrentStateIntent,
    ReactToDesiredOutcomeIntent,
    ReactToEstimateCostIntent,
    ReactToUpdateEntityIntent,
    RecognizeIntent,
    RespondUnknown,
    SayGoodbye,
)
from logistics_agent.domain import At, DriveLongDistanceAction, GasPrice, Package, Truck, UnloadAction

PRICE = "how much to ship a package from seattle to san francisco?"
GIBBERISH = "what is the capital of france?"
HELLO = "hello there"
CAPABILITIES = "what can you do?"

OUTCOME = "I want my crate of machine parts to end up in denver"
TWO_OUTCOMES = "the crate should go to the san francisco office, and the pelican should end at SFO"
VAGUE_OUTCOME = "I want my crate delivered somewhere nice"
UNKNOWN_OUTCOME = "send the crate to narnia"
IMPOSSIBLE_OUTCOME = "put the denver office inside big bertha"

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
UPDATE_SOMETHING = "change something"
UPDATE_UNICORN = "make the unicorn faster"
UPDATE_VAGUE = "change the gas price"

BOOKS = Package(name="made_up_by_the_llm", display_name="box of books", weight=200)

INTENTS = {
    PRICE: HumanIntent.GET_PRICE_ESTIMATE,
    GIBBERISH: None,
    HELLO: HumanIntent.GREETING,
    CAPABILITIES: HumanIntent.ASK_CAPABILITIES,
    **dict.fromkeys((OUTCOME, TWO_OUTCOMES, VAGUE_OUTCOME, UNKNOWN_OUTCOME, IMPOSSIBLE_OUTCOME),
                    HumanIntent.STATE_DESIRED_OUTCOME),
    **dict.fromkeys((ASK_TRUCKS, ASK_VEHICLES_AND_TRUCKS, ASK_AIRPLANES, ASK_UNICORNS), HumanIntent.ASK_ABOUT_ENTITIES),
    **dict.fromkeys((WHERE_BERTHA, WHAT_AT_SEA, WHAT_IN_PELICAN, WHERE_BOOKS, WHERE_VAGUE), HumanIntent.ASK_CURRENT_STATE),
    **dict.fromkeys((ADD_BOOKS, ADD_OFFICE, ADD_SOMETHING, ADD_NOWHERE, ADD_NARNIA, ADD_VAGUE, ADD_TRUCK_AT_GAS_PRICE),
                    HumanIntent.ADD_ENTITY),
    **dict.fromkeys((UPDATE_GAS, UPDATE_GAS_SAME, UPDATE_CRATE, UPDATE_SOMETHING, UPDATE_UNICORN, UPDATE_VAGUE),
                    HumanIntent.UPDATE_ENTITY),
}

# what each populator returns, keyed by (model type, phrase); an exception is raised instead of returned
POPULATED = {
    (DesiredOutcome, OUTCOME): DesiredOutcome(facts=[AtFact(entity1="package1", entity2="office_denver")]),
    (DesiredOutcome, TWO_OUTCOMES): DesiredOutcome(facts=[
        AtFact(entity1="package1", entity2="office_san_francisco"), AtFact(entity1="air1", entity2="SFO"),
    ]),
    (DesiredOutcome, VAGUE_OUTCOME): DesiredOutcome(facts=[]),
    (DesiredOutcome, UNKNOWN_OUTCOME): DesiredOutcome(facts=[AtFact(entity1="package1", entity2="narnia")]),
    (DesiredOutcome, IMPOSSIBLE_OUTCOME): DesiredOutcome(facts=[AtFact(entity1="office_denver", entity2="truck1")]),

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


def test_estimate_without_a_desired_outcome_delivers_to_san_francisco(chat):
    chat(PRICE, "quit")
    [(actions, _)] = DESCRIBED
    assert actions[-1].location.name == "office_san_francisco"


def test_desired_outcome_is_confirmed_and_used_by_the_estimate(chat):
    _, confirmation, _, _ = chat(OUTCOME, PRICE, "quit")
    assert confirmation == ("Got it: the crate of machine parts at the Denver Office. "
                            "Ask me for a price estimate whenever you are ready.")
    [(actions, _)] = DESCRIBED
    unload = actions[-1]
    assert isinstance(unload, UnloadAction)
    assert (unload.package.name, unload.location.name) == ("package1", "office_denver")


def test_vague_outcome_keeps_the_goal_the_estimate_uses(chat):
    chat(VAGUE_OUTCOME, PRICE, "quit")
    [(actions, _)] = DESCRIBED
    assert actions[-1].location.name == "office_san_francisco"


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


def goal_of(state: ConversationState) -> list[tuple[str, str]]:
    return [(fact.entity1.name, fact.entity2.name) for fact in state.planner.goal]


def location_of(state: ConversationState, name: str) -> str:
    [at] = state.reasoner.find_relations(At, name)
    return at.entity2.name


# --- conversation state


def test_planner_plans_over_the_reasoners_world_graph():
    state = ConversationState()
    assert state.planner.world is state.reasoner.world


def test_planner_starts_with_package1_at_the_san_francisco_office():
    assert goal_of(ConversationState()) == [("package1", "office_san_francisco")]


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
    ReactToEstimateCostIntent: HumanIntent.GET_PRICE_ESTIMATE,
    ReactToDesiredOutcomeIntent: HumanIntent.STATE_DESIRED_OUTCOME,
    ReactToAskAboutEntitiesIntent: HumanIntent.ASK_ABOUT_ENTITIES,
    ReactToAskCurrentStateIntent: HumanIntent.ASK_CURRENT_STATE,
    ReactToAddEntityIntent: HumanIntent.ADD_ENTITY,
    ReactToUpdateEntityIntent: HumanIntent.UPDATE_ENTITY,
}


@pytest.mark.parametrize("op_type", list(REACTIONS))
@pytest.mark.parametrize("intent", [*HumanIntent, None])
def test_each_reaction_applies_only_to_its_intent(op_type, intent):
    state = ConversationState(human_utterance=Utterance(GIBBERISH), human_intent=intent)
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


def test_estimate_plans_toward_the_planners_goal_without_changing_it():
    state, reply, _ = react(ReactToEstimateCostIntent, PRICE)
    [(actions, task_desc)] = DESCRIBED
    assert actions[-1].location.name == "office_san_francisco"
    assert reply == f"described 4 facts: {task_desc}"
    assert goal_of(state) == [("package1", "office_san_francisco")]


# --- desired outcome


def test_desired_outcome_populates_over_the_outcome_model_with_the_known_entities():
    state, _, op = react(ReactToDesiredOutcomeIntent, OUTCOME)
    populator = op._outcome_populator
    assert populator.model_type is DesiredOutcome
    assert populator.task_desc == ReactToDesiredOutcomeIntent.TASK_DESC
    assert populator.calls == [(OUTCOME, conductor.DEFAULT_LLM, tuple(known_entities(state.reasoner.world)))]


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
    state, reply, _ = react(ReactToDesiredOutcomeIntent, phrase)
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


def test_update_entity_keeps_its_location():
    state, reply, _ = react(ReactToUpdateEntityIntent, UPDATE_CRATE)
    assert reply == "Updated the crate of machine parts: weight is now 500.0."
    assert location_of(state, "package1") == "office_seattle"


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
