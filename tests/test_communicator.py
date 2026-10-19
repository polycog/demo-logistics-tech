"""offline tests of the communicator's language models: what they turn into, and what they tell the LLM"""

import pytest
from pydantic import ValidationError

from communicator import AtFact, DesiredOutcome, EntityKind, HumanIntent, known_entities
from domain import Airplane, Airport, At, GasPrice, LaborCost, Office, Package, Truck, Vehicle
from reasoner import Reasoner


def test_every_intent_has_a_description():
    assert all(intent.__doc__ for intent in HumanIntent)


@pytest.mark.parametrize("kind, entity_type", [
    (EntityKind.OFFICE, Office),
    (EntityKind.AIRPORT, Airport),
    (EntityKind.PACKAGE, Package),
    (EntityKind.VEHICLE, Vehicle),
    (EntityKind.TRUCK, Truck),
    (EntityKind.AIRPLANE, Airplane),
    (EntityKind.GAS_PRICE, GasPrice),
    (EntityKind.LABOR_COST, LaborCost),
])
def test_every_entity_kind_maps_to_its_domain_type(kind, entity_type):
    assert kind.entity_type is entity_type and kind.__doc__


def test_desired_outcome_resolves_to_at_relations():
    world = Reasoner().world
    outcome = DesiredOutcome(facts=[
        AtFact(entity1="package1", entity2="office_denver"),
        AtFact(entity1="air1", entity2="SFO"),
    ])
    relations = outcome.to_relations(world)
    assert all(isinstance(r, At) for r in relations)
    assert [(r.entity1.name, r.entity2.name) for r in relations] == [("package1", "office_denver"), ("air1", "SFO")]


def test_empty_desired_outcome_resolves_to_no_relations():
    assert DesiredOutcome(facts=[]).to_relations(Reasoner().world) == []


def test_desired_outcome_with_an_unknown_name_raises_key_error():
    with pytest.raises(KeyError):
        DesiredOutcome(facts=[AtFact(entity1="package1", entity2="narnia")]).to_relations(Reasoner().world)


def test_desired_outcome_that_cannot_hold_raises_validation_error():
    with pytest.raises(ValidationError):
        DesiredOutcome(facts=[AtFact(entity1="office_denver", entity2="truck1")]).to_relations(Reasoner().world)


def test_known_entities_names_everything_the_human_can_refer_to():
    lines = known_entities(Reasoner().world)
    for expected in (
        "package package1: crate of machine parts",
        "truck truck1: Big Bertha",
        "airplane air1: The Pelican",
        "office office_boston: Boston Office in boston",
        "airport SEA: Seattle-Tacoma International Airport in seattle",
        "gas price gas_price",
        "labor cost labor_cost_denver: labor cost in denver",
    ):
        assert expected in lines
    assert len(lines) == 1 + 2 + 1 + 5 + 5 + 1 + 5


def test_known_entities_follows_changes_to_the_world():
    reasoner = Reasoner()
    reasoner.add_entity(Package(name="package2", display_name="box of books", weight=200), "office_boston")
    assert "package package2: box of books" in known_entities(reasoner.world)
