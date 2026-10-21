"""offline tests of the reasoner's queries over the world graph and its updates to it"""

import pytest
from pydantic import ValidationError

from logistics_agent.domain import Airplane, At, GasPrice, Terminal, Shipment, Truck, Vehicle
from logistics_agent.reasoner import Reasoner


def pairs(relations) -> set[tuple[str, str]]:
    return {(r.entity1.name, r.entity2.name) for r in relations}


def location_of(reasoner: Reasoner, name: str) -> str:
    [at] = reasoner.find_relations(At, name)
    return at.entity2.name


# === entity queries


@pytest.mark.parametrize("entity_type, names", [
    (Truck, {"truck1", "truck2"}),
    (Airplane, {"air1"}),
    (Vehicle, {"truck1", "truck2", "air1"}),
    (Shipment, {"shipment1"}),
    (GasPrice, {"gas_price"}),
])
def test_find_entities_returns_every_entity_of_the_type_and_its_subtypes(entity_type, names):
    assert {e.name for e in Reasoner().find_entities(entity_type)} == names


def test_find_entities_returns_nothing_for_a_type_with_no_entities():
    reasoner = Reasoner()
    reasoner.world.remove_node("air1")
    assert reasoner.find_entities(Airplane) == []


@pytest.mark.parametrize("name", ["truck1", "Big Bertha", "big bertha", "TRUCK1"])
def test_find_entity_matches_name_or_display_name_ignoring_case(name):
    entity = Reasoner().find_entity(name)
    assert isinstance(entity, Truck) and entity.name == "truck1"


@pytest.mark.parametrize("name", ["gas price", "Gas Price", "gas_price"])
def test_find_entity_treats_underscores_as_spaces(name):
    assert Reasoner().find_entity(name).name == "gas_price"


def test_find_entity_returns_the_graphs_entity_not_a_frozen_copy():
    assert type(Reasoner().find_entity("terminal_boston")) is Terminal


def test_find_entity_returns_none_for_an_unknown_name():
    assert Reasoner().find_entity("unicorn") is None


# === relation queries


def test_find_relations_constrains_entity1():
    assert pairs(Reasoner().find_relations(At, "truck1")) == {("truck1", "terminal_seattle")}


def test_find_relations_constrains_entity2():
    assert pairs(Reasoner().find_relations(At, entity2="SEA")) == {("air1", "SEA")}


def test_find_relations_constrains_both_sides():
    reasoner = Reasoner()
    assert pairs(reasoner.find_relations(At, "air1", "SEA")) == {("air1", "SEA")}
    assert reasoner.find_relations(At, "air1", "SFO") == []


def test_find_relations_without_constraints_returns_every_relation_of_the_type():
    assert pairs(Reasoner().find_relations(At)) == {
        ("truck1", "terminal_seattle"), ("truck2", "terminal_san_francisco"), ("air1", "SEA")
    }


def test_find_relations_matches_display_names_ignoring_case():
    assert pairs(Reasoner().find_relations(At, "big bertha", "seattle terminal")) == {
        ("truck1", "terminal_seattle")
    }


def test_find_relations_returns_nothing_for_an_unknown_name():
    assert Reasoner().find_relations(At, "unicorn") == []


# === adding entities


def test_fresh_name_skips_names_in_use():
    reasoner = Reasoner()
    assert reasoner.fresh_name("shipment") == "shipment2"
    assert reasoner.fresh_name("truck") == "truck3"
    assert reasoner.fresh_name("airplane") == "airplane1"


def test_add_entity_places_it_at_the_location():
    reasoner = Reasoner()
    reasoner.add_entity(Shipment(name="shipment2", details="box of books", weight=200), "Boston Terminal")
    assert reasoner.find_entity("shipment2").weight == 200
    assert location_of(reasoner, "shipment2") == "terminal_boston"


def test_add_entity_without_a_location_adds_no_relation():
    reasoner = Reasoner()
    reasoner.add_entity(Shipment(name="shipment2", details="box of books", weight=200))
    assert reasoner.find_entity("shipment2") is not None
    assert reasoner.find_relations(At, "shipment2") == []


def test_add_entity_refuses_a_name_in_use():
    reasoner = Reasoner()
    with pytest.raises(ValueError, match="shipment1"):
        reasoner.add_entity(Shipment(name="shipment1", details="impostor", weight=1))
    assert reasoner.find_entity("shipment1").details == "crate of machine parts"


def test_add_entity_refuses_an_unknown_location_and_adds_nothing():
    reasoner = Reasoner()
    with pytest.raises(KeyError):
        reasoner.add_entity(Shipment(name="shipment2", details="box of books", weight=200), "narnia")
    assert reasoner.find_entity("shipment2") is None


def test_add_entity_refuses_a_location_it_cannot_be_at_and_adds_nothing():
    reasoner = Reasoner()
    with pytest.raises(ValidationError):
        reasoner.add_entity(Shipment(name="shipment2", details="box of books", weight=200), "gas_price")
    assert reasoner.find_entity("shipment2") is None


# === updating entities


def test_update_entity_replaces_its_values():
    reasoner = Reasoner()
    reasoner.update_entity(GasPrice(name="gas_price", per_mile=0.95))
    assert reasoner.find_entity("gas_price").per_mile == 0.95


def test_update_entity_keeps_its_relations():
    reasoner = Reasoner()
    reasoner.update_entity(Truck(name="truck1", display_name="Big Bertha", model="Freightliner M2 106 26ft box truck",
                                 max_payload=12000, license_plate="C12345A"))
    assert reasoner.find_entity("truck1").max_payload == 12000
    assert location_of(reasoner, "truck1") == "terminal_seattle"


def test_update_entity_refuses_an_unknown_name():
    with pytest.raises(KeyError):
        Reasoner().update_entity(GasPrice(name="unicorn", per_mile=1))


def test_update_entity_refuses_a_change_of_type():
    reasoner = Reasoner()
    with pytest.raises(TypeError, match="truck1"):
        reasoner.update_entity(Shipment(name="truck1", details="not a truck", weight=1))
    assert isinstance(reasoner.find_entity("truck1"), Truck)


def test_updates_are_seen_by_the_planner():
    """the planner plans over the same world graph, so a pricier gas price makes the drive cost more"""
    from logistics_agent.planner import LogisticsPlanner

    reasoner = Reasoner()
    lp = LogisticsPlanner(reasoner.world)
    lp.set_current([At(entity1=reasoner.find_entity("shipment1"), entity2=reasoner.find_entity("terminal_seattle"))])
    lp.set_goal([At(entity1=reasoner.find_entity("shipment1"), entity2=reasoner.find_entity("terminal_san_francisco"))])
    _, before = lp.plan()
    reasoner.update_entity(GasPrice(name="gas_price", per_mile=1.0))
    _, after = lp.plan()
    assert before == pytest.approx(780)
    assert after > before
