"""offline tests of drafts, the models filled in over several turns, and of the fleet models they are used for"""

import pytest
from pydantic import ValidationError

from logistics_agent.communicator import AskFor, AskForKind
from logistics_agent.domain import Airplane, At, Tag, Truck, Vehicle, VehicleType
from logistics_agent.drafts import build, draft_model, drafted, missing
from logistics_agent.reasoner import Reasoner

ROAD_RUNNER = {"name": "truck3", "display_name": "Road Runner", "model": "Ford F-750", "max_payload": 12000,
               "license_plate": "7XYZ123"}


# --- the fleet


def test_every_vehicle_is_part_of_the_fleet():
    assert Tag.FLEET in Vehicle.tags and Truck.tags == Airplane.tags == Vehicle.tags


def test_the_tags_and_name_prefixes_are_not_fields():
    assert "tags" not in Truck.model_fields and "name_prefix" not in Truck.model_fields
    assert (Truck.name_prefix, Airplane.name_prefix) == ("truck", "air")


def test_the_vehicle_types_are_the_vehicles_the_human_can_add():
    assert [kind.model for kind in VehicleType] == [Truck, Airplane]
    assert all(kind.__doc__ for kind in VehicleType)


@pytest.mark.parametrize("kind", list(VehicleType))
def test_every_field_the_human_gives_of_a_vehicle_type_is_asked_with_a_question(kind):
    for name, info in kind.model.model_fields.items():
        if name != "name":
            assert info.description and info.description.endswith("?"), f"{kind.model.__name__}.{name}"


def test_the_kind_and_where_a_vehicle_is_are_asked_with_questions():
    question = AskForKind().question
    assert question == VehicleType.__doc__ and question.endswith("?")
    assert all(kind.value in question for kind in VehicleType)
    assert AskFor(At, "entity2").question.endswith("?")


# --- drafts


def test_a_draft_has_the_models_fields_each_optional_with_the_same_descriptions():
    draft = draft_model(Truck)()
    assert list(type(draft).model_fields) == list(Truck.model_fields)
    assert all(getattr(draft, name) is None for name in Truck.model_fields)
    assert type(draft).model_fields["license_plate"].description == Truck.model_fields["license_plate"].description


def test_a_draft_model_is_made_once_per_model_and_knows_what_it_drafts():
    assert draft_model(Truck) is draft_model(Truck) is not draft_model(Airplane)
    assert drafted(draft_model(Truck)()) is Truck


def test_missing_lists_the_fields_not_known_yet_in_order():
    draft = draft_model(Truck)(name="truck3", display_name="Road Runner")
    assert missing(draft) == ["model", "max_payload", "license_plate"]
    draft.license_plate = "7XYZ123"
    assert missing(draft) == ["model", "max_payload"]


def test_asks_for_a_model_are_satisfied_by_its_draft():
    draft = draft_model(Truck)(name="truck3", display_name="Road Runner")
    assert AskFor(Truck, "display_name").satisfied(draft) and not AskFor(Truck, "model").satisfied(draft)
    assert not AskFor(Truck).satisfied(draft)
    assert AskFor(Truck).satisfied(draft_model(Truck)(**ROAD_RUNNER))


def test_build_makes_the_model_once_nothing_is_missing():
    truck = build(draft_model(Truck)(**ROAD_RUNNER))
    assert truck == Truck(**ROAD_RUNNER)
    with pytest.raises(AssertionError):
        build(draft_model(Truck)(name="truck3"))


def test_build_makes_a_relation_from_the_entities_in_its_draft():
    truck, denver = Truck(**ROAD_RUNNER), Reasoner().find_entity("terminal_denver")
    at = build(draft_model(At)(entity1=truck, entity2=denver))
    assert isinstance(at, At) and at.entity1 == truck and at.entity2.name == "terminal_denver"


def test_build_runs_the_models_validation():
    with pytest.raises(ValidationError):
        build(draft_model(At)(entity1=Truck(**ROAD_RUNNER), entity2=Reasoner().find_entity("denver")))
