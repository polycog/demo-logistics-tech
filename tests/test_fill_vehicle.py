"""offline tests of the VehicleFiller and PlacementFiller sub-processes, which fill in a vehicle to add to the fleet and
where it is

a fake EnumClassifier and a fake ModelPopulator stand in for the LLM, answering with the kind and the fields they are
given, or failing, so the sub-processes run deterministically with no API key
"""

import pytest
from pydantic_ai.exceptions import UnexpectedModelBehavior

from logistics_agent import fill_placement, fill_vehicle
from logistics_agent.communicator import AskFor, AskForKind, Utterance, asked_context, in_context
from logistics_agent.domain import Airplane, At, Truck, VehicleType
from logistics_agent.drafts import draft_model, drafted
from logistics_agent.fill_placement import PlacementFiller, location_named, places_for
from logistics_agent.fill_vehicle import ChooseKind, VehicleFiller, fields_model, valid
from logistics_agent.reasoner import Reasoner

PHRASE = "add a truck called Road Runner, a Ford F-750 that carries 12000 pounds, plate 7XYZ123"
ROAD_RUNNER = {"display_name": "Road Runner", "model": "Ford F-750", "max_payload": 12000, "license_plate": "7XYZ123"}
ASK_KIND = AskForKind(VehicleType)


class FakeEnumClassifier:
    """stands in for EnumClassifier: answers with the VehicleType whose value is the kind in `kind`, or None for any
    other kind; records how it was built, and every text it classified"""

    kind: str | None = "truck"
    built: list[tuple] = []
    calls: list[str] = []

    def __init__(self, enum_type, task_desc: str) -> None:
        self.enum_type = enum_type
        FakeEnumClassifier.built.append((enum_type, task_desc))

    def __call__(self, text: str, llm, **kwargs):
        FakeEnumClassifier.calls.append(text)
        return next((kind for kind in self.enum_type if kind.value == FakeEnumClassifier.kind), None), 1.0


class FakeModelPopulator:
    """stands in for ModelPopulator: answers with the fields in `given` that the model it was built for has, or raises
    the error in `given`; records the models it was built for, and every call"""

    given: dict | Exception = {}
    built: list[type] = []
    calls: list[tuple] = []

    def __init__(self, model_type, task_desc: str) -> None:
        self.model_type = model_type
        FakeModelPopulator.built.append(model_type)

    def __call__(self, utterance: str, llm, *extra: str, **kwargs):
        FakeModelPopulator.calls.append((utterance, extra))
        if isinstance(FakeModelPopulator.given, Exception):
            raise FakeModelPopulator.given
        fields = self.model_type.model_fields
        return self.model_type.model_validate({k: v for k, v in FakeModelPopulator.given.items() if k in fields})


@pytest.fixture(autouse=True)
def fake_populator(monkeypatch):
    """replaces the LLM-backed EnumClassifier and ModelPopulator of both sub-processes; returns the fake populator,
    whose answer the test sets"""
    FakeEnumClassifier.kind, FakeEnumClassifier.built, FakeEnumClassifier.calls = "truck", [], []
    monkeypatch.setattr(fill_vehicle, "EnumClassifier", FakeEnumClassifier)
    FakeModelPopulator.given = dict(ROAD_RUNNER)
    FakeModelPopulator.built, FakeModelPopulator.calls = [], []
    monkeypatch.setattr(fill_vehicle, "ModelPopulator", FakeModelPopulator)
    monkeypatch.setattr(fill_placement, "ModelPopulator", FakeModelPopulator)
    return FakeModelPopulator


def fill(draft=None, answering: AskFor | None = None, reasoner: Reasoner | None = None):
    """runs a fresh VehicleFiller over the phrase, as the conductor does"""
    return VehicleFiller()(
        utterance=Utterance(PHRASE), answering=answering, reasoner=reasoner or Reasoner(), draft=draft,
        suppress_errors=False,
    )


def truck(**fields):
    """a draft of a truck named truck3 with the fields"""
    return draft_model(Truck)(name="truck3", **fields)


# === VehicleFiller

# --- the kind


def test_the_kind_is_classified_among_the_vehicle_types():
    ChooseKind("op")
    assert FakeEnumClassifier.built == [(VehicleType, ChooseKind.TASK_DESC)]


@pytest.mark.parametrize("kind, model", [(VehicleType.TRUCK, Truck), (VehicleType.AIRPLANE, Airplane)])
def test_each_vehicle_type_knows_its_model(kind, model):
    assert kind.model is model


def test_the_kind_starts_a_draft_named_with_a_name_no_other_entity_has():
    result = fill()
    assert drafted(result.draft) is Truck and result.draft.name == "truck3"


def test_an_airplane_draft_is_named_after_the_airplanes():
    FakeEnumClassifier.kind = "airplane"
    assert fill().draft.name == "air2"


@pytest.mark.parametrize("kind", ["boat", None])
def test_without_a_kind_there_is_no_draft_and_the_kind_is_asked_for(fake_populator, kind):
    FakeEnumClassifier.kind = kind
    result = fill()
    assert result.draft is None and result.asks == [ASK_KIND] and fake_populator.calls == []


def test_the_kind_is_classified_as_the_answer_to_the_question_asked():
    fill(answering=ASK_KIND)
    assert FakeEnumClassifier.calls == [in_context(ASK_KIND.question, PHRASE)]


# --- the fields


def test_the_fields_are_filled_in_together_in_one_call_after_the_kind(fake_populator):
    result = fill()
    assert result.asks == [] and {name: getattr(result.draft, name) for name in ROAD_RUNNER} == ROAD_RUNNER
    [fields] = fake_populator.built
    assert set(fields.model_fields) == set(ROAD_RUNNER) and FakeEnumClassifier.calls == [PHRASE]


def test_only_the_fields_missing_from_the_draft_are_filled_in(fake_populator):
    result = fill(truck(display_name="Big Wheel", model="Mack MD"))
    assert (result.draft.display_name, result.draft.model) == ("Big Wheel", "Mack MD")
    [fields] = fake_populator.built
    assert set(fields.model_fields) == {"max_payload", "license_plate"}


def test_a_draft_given_is_left_as_it_was():
    draft = truck(display_name="Big Wheel")
    fill(draft)
    assert draft == truck(display_name="Big Wheel")


def test_the_fields_still_missing_are_asked_for_in_order():
    FakeModelPopulator.given = {"display_name": "Road Runner"}
    assert fill().asks == [AskFor(Truck, "model"), AskFor(Truck, "max_payload"), AskFor(Truck, "license_plate")]


@pytest.mark.parametrize("value, ok", [("Ford", True), ("  ", False), (12000, True), (0, False), (-5, False),
                                       (None, False)])
def test_a_value_is_kept_only_when_it_is_text_that_is_not_blank_or_a_number_above_zero(value, ok):
    assert valid(value) is ok


def test_blank_text_and_a_payload_not_above_zero_are_asked_for_again():
    FakeModelPopulator.given = {**ROAD_RUNNER, "model": " ", "max_payload": 0}
    assert fill().asks == [AskFor(Truck, "model"), AskFor(Truck, "max_payload")]


def test_the_fields_model_uses_the_vehicles_questions():
    model = fields_model(truck(), ["license_plate"])
    assert Truck.model_fields["license_plate"].description in model.model_fields["license_plate"].description


def test_a_complete_draft_asks_for_nothing_without_calling_the_language_model(fake_populator):
    result = fill(truck(**ROAD_RUNNER))
    assert result.asks == [] and fake_populator.calls == []


# === PlacementFiller

DENVER_TERMINAL = "Denver Terminal"


def place(vehicle, given: dict, answering: AskFor | None = None):
    """runs a fresh PlacementFiller for the vehicle, with the language model giving the fields"""
    FakeModelPopulator.given = given
    return PlacementFiller()(
        utterance=Utterance("it is at the Denver terminal"), answering=answering, reasoner=Reasoner(),
        draft=draft_model(At)(entity1=vehicle), suppress_errors=False,
    )


def road_runner() -> Truck:
    return Truck(name="truck3", **ROAD_RUNNER)


def pelican_two() -> Airplane:
    return Airplane(name="air2", display_name="Pelican Two", model="Cessna", max_payload=5000, tail_number="N1")


def test_places_for_a_truck_are_every_terminal_and_airport_and_for_an_airplane_only_the_airports():
    reasoner = Reasoner()
    assert len(places_for(road_runner(), reasoner)) == 10
    assert {place.name for place in places_for(pelican_two(), reasoner)} == {"SEA", "SFO", "BOS", "DEN", "DTW"}


def test_location_named_finds_a_place_by_its_name_or_display_name():
    places = Reasoner().find_entities(fill_placement.Location)
    assert location_named("denver terminal", places).name == "terminal_denver"
    assert location_named("DEN", places).name == "DEN"
    assert location_named("Union Station", places) is None and location_named(None, places) is None


def test_the_location_is_filled_in_and_nothing_more_is_asked(fake_populator):
    result = place(road_runner(), {"entity2": DENVER_TERMINAL}, answering=AskFor(At, "entity2"))
    assert result.draft.entity2.name == "terminal_denver" and result.asks == []
    assert fake_populator.calls[0][1] == tuple(asked_context(AskFor(At, "entity2").question))


@pytest.mark.parametrize("given", [{"entity2": "Union Station"}, {}, UnexpectedModelBehavior("no answer")])
def test_without_a_known_location_it_is_asked_for(given):
    result = place(road_runner(), given)
    assert result.draft.entity2 is None and result.asks == [AskFor(At, "entity2")]


def test_an_airplane_can_only_be_placed_at_an_airport():
    assert place(pelican_two(), {"entity2": DENVER_TERMINAL}).draft.entity2 is None
    assert place(pelican_two(), {"entity2": "Denver International Airport"}).draft.entity2.name == "DEN"
