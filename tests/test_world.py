"""offline tests of the world model: the static world graph, each evolver, and the logistics planner"""

from typing import Any

import pytest
from cognition import WorldGraph, WorldSnapshot
from pydantic import ValidationError

from logistics_agent.domain import (
    AirRoute, Airplane, Airport, At, City, DriveLocalAction, DriveLongDistanceAction, FlyAction, GasPrice, LaborCost, LoadAction, Terminal,
    Location, Shipment, ShippingQuery, ShippingRequest, Truck, TruckRoute, UnloadAction,
)
from logistics_agent.reasoner import Reasoner
from logistics_agent.planner import (
    EvolverDriveTruckLocal,
    EvolverDriveTruckLongDistance,
    EvolverFlyAirplane,
    EvolverLoadShipment,
    EvolverUnloadShipment,
    LogisticsPlanner,
)


def entity(world: WorldGraph, name: str) -> Any:
    return world.get_entity(name, linked=False)


def world_with(*placements: tuple[str, str]) -> WorldGraph:
    """the static world graph, with each (thing, location) placement added as an At fact"""
    world = Reasoner.configure_world_graph()
    for thing, location in placements:
        world.add_relation(At(entity1=entity(world, thing), entity2=entity(world, location)))
    return world


def label(option: Any) -> str:
    """a readable name for an evolver option, built from the things it acts on

    action models compare by value, and the snapshot holds frozen copies of entities, so tests find
    options by name instead
    """
    match option:
        case EvolverDriveTruckLocal():
            return f"drive {option.truck.name} locally from {option.from_location.name} to {option.to_location.name}"
        case EvolverDriveTruckLongDistance():
            return f"drive {option.truck.name} from {option.from_location.name} to {option.to_location.name}"
        case EvolverFlyAirplane():
            return f"fly {option.airplane.name} from {option.from_airport.name} to {option.to_airport.name}"
        case EvolverLoadShipment():
            return f"load {option.shipment.name} into {option.vehicle.name} at {option.location.name}"
        case EvolverUnloadShipment():
            return f"unload {option.shipment.name} from {option.vehicle.name} at {option.location.name}"
    raise TypeError(option)


def moves(evolver: type, snapshot: WorldSnapshot) -> dict[str, Any]:
    """every option the evolver proposes in the snapshot, keyed by its label"""
    return {label(option): option for option in evolver.when(snapshot)}


def location_of(snapshot: WorldSnapshot, name: str) -> str:
    return next(iter(snapshot.by(At, lambda a: a.entity1.name == name))).entity2.name


# === static world graph


def test_static_world_has_every_entity():
    snapshot = Reasoner.configure_world_graph().snapshot
    assert {o.name for o in snapshot.by(Terminal)} == {
        "terminal_seattle", "terminal_san_francisco", "terminal_boston", "terminal_denver", "terminal_detroit"
    }
    assert {a.name for a in snapshot.by(Airport)} == {"SEA", "SFO", "BOS", "DEN", "DTW"}
    assert {t.name for t in snapshot.by(Truck)} == {"truck1", "truck2"}
    assert {a.name for a in snapshot.by(Airplane)} == {"air1"}
    assert {p.name for p in snapshot.by(Shipment)} == {"shipment1"}
    assert snapshot.find_first(GasPrice).per_mile == 0.8


def test_every_city_has_one_labor_cost():
    snapshot = Reasoner.configure_world_graph().snapshot
    costs = list(snapshot.by(LaborCost))
    assert sorted(c.city.name for c in costs) == sorted(c.name for c in snapshot.by(City))
    assert all(c.by_weight > 0 for c in costs)


def test_every_entity_has_a_human_readable_name():
    snapshot = Reasoner.configure_world_graph().snapshot
    for kind in (Terminal, Airport, Truck, Airplane):
        for e in snapshot.by(kind):
            assert e.display_name and e.display_name != e.name
    for shipment in snapshot.by(Shipment):
        assert shipment.details and shipment.details != shipment.name


def test_vehicles_can_carry_the_shipment():
    snapshot = Reasoner.configure_world_graph().snapshot
    shipment = snapshot.find_first(Shipment)
    for vehicle in (*snapshot.by(Truck), *snapshot.by(Airplane)):
        assert vehicle.max_payload >= shipment.weight


def test_static_world_connects_every_pair_of_cities_once():
    snapshot = Reasoner.configure_world_graph().snapshot
    air_pairs = [frozenset((r.entity1.name, r.entity2.name)) for r in snapshot.by(AirRoute)]
    truck_pairs = [frozenset((r.entity1.name, r.entity2.name)) for r in snapshot.by(TruckRoute)]
    assert len(air_pairs) == len(set(air_pairs)) == 10
    assert len(truck_pairs) == len(set(truck_pairs)) == 10


def test_static_world_has_five_cities_and_truck_routes_connect_cities():
    snapshot = Reasoner.configure_world_graph().snapshot
    assert {(c.name, c.display_name) for c in snapshot.by(City)} == {
        ("seattle", "Seattle"), ("san_francisco", "San Francisco"), ("boston", "Boston"), ("denver", "Denver"),
        ("detroit", "Detroit"),
    }
    assert all(isinstance(r.entity1, City) and isinstance(r.entity2, City) for r in snapshot.by(TruckRoute))


def test_static_world_has_no_locations():
    """where shipments and vehicles are is dynamic, so it is not part of the static configuration"""
    assert not list(Reasoner.configure_world_graph().snapshot.by(At))


# === evolvers


def test_local_drive_stays_within_the_city():
    snapshot = world_with(("truck1", "SEA")).snapshot
    assert set(moves(EvolverDriveTruckLocal, snapshot)) == {"drive truck1 locally from SEA to terminal_seattle"}


def test_local_drive_moves_the_truck():
    snapshot = world_with(("truck1", "SEA")).snapshot
    new, cost = moves(EvolverDriveTruckLocal, snapshot)["drive truck1 locally from SEA to terminal_seattle"].then(snapshot)
    assert location_of(new, "truck1") == "terminal_seattle"
    assert cost == 50


def test_long_distance_drive_goes_both_ways_along_truck_routes_to_every_location_in_the_other_city():
    snapshot = world_with(("truck1", "terminal_boston")).snapshot
    assert set(moves(EvolverDriveTruckLongDistance, snapshot)) == {
        f"drive truck1 from terminal_boston to {place}"
        for place in ("terminal_seattle", "terminal_san_francisco", "terminal_denver", "terminal_detroit",
                      "SEA", "SFO", "DEN", "DTW")
    }


def test_long_distance_drive_can_start_at_any_location_in_a_city_with_a_terminal():
    drives = moves(EvolverDriveTruckLongDistance, world_with(("truck1", "SEA")).snapshot)
    assert "drive truck1 from SEA to terminal_san_francisco" in drives and len(drives) == 8


def test_long_distance_drive_costs_gas_price_times_distance():
    snapshot = world_with(("truck1", "terminal_seattle")).snapshot
    drive = moves(EvolverDriveTruckLongDistance, snapshot)["drive truck1 from terminal_seattle to terminal_san_francisco"]
    new, cost = drive.then(snapshot)
    assert location_of(new, "truck1") == "terminal_san_francisco"
    assert cost == pytest.approx(0.8 * 800)


def test_long_distance_drive_needs_a_terminal_in_the_city_it_leaves():
    world = world_with(("truck1", "SEA"))
    world.remove_node("terminal_seattle")
    assert not moves(EvolverDriveTruckLongDistance, world.snapshot)


def test_long_distance_drive_needs_a_terminal_in_the_city_it_goes_to():
    world = world_with(("truck1", "SEA"))
    world.remove_node("terminal_denver")
    drives = moves(EvolverDriveTruckLongDistance, world.snapshot)
    assert not any(drive.endswith(("terminal_denver", "DEN")) for drive in drives) and len(drives) == 6


def test_long_distance_drive_reaches_a_new_location_in_a_city_with_a_terminal():
    world = world_with(("truck1", "terminal_seattle"))
    world.add_entity(Location(name="dropoff_san_francisco", city=entity(world, "san_francisco")))
    drive = moves(EvolverDriveTruckLongDistance, world.snapshot)["drive truck1 from terminal_seattle to dropoff_san_francisco"]
    assert drive.then(world.snapshot)[1] == pytest.approx(0.8 * 800)


def test_flight_costs_the_route_price_in_either_direction():
    snapshot = world_with(("air1", "DTW")).snapshot
    flights = moves(EvolverFlyAirplane, snapshot)
    assert set(flights) == {f"fly air1 from DTW to {code}" for code in ("SEA", "SFO", "BOS", "DEN")}
    new, cost = flights["fly air1 from DTW to SEA"].then(snapshot)
    assert location_of(new, "air1") == "SEA"
    assert cost == 364


def test_load_and_unload_move_the_shipment_through_the_vehicle():
    snapshot = world_with(("truck1", "terminal_seattle"), ("shipment1", "terminal_seattle")).snapshot
    loaded, _ = moves(EvolverLoadShipment, snapshot)["load shipment1 into truck1 at terminal_seattle"].then(snapshot)
    assert location_of(loaded, "shipment1") == "truck1"
    assert not moves(EvolverLoadShipment, loaded)

    unloaded, cost = moves(EvolverUnloadShipment, loaded)["unload shipment1 from truck1 at terminal_seattle"].then(loaded)
    assert location_of(unloaded, "shipment1") == "terminal_seattle"
    assert cost == pytest.approx(1000 * 0.06)


def test_handling_costs_the_shipment_weight_times_the_city_labor_rate():
    """the 1000 lb shipment costs 60 to handle in Seattle (0.06 per lb) and 80 in San Francisco (0.08 per lb)"""
    seattle = world_with(("truck1", "terminal_seattle"), ("shipment1", "terminal_seattle")).snapshot
    [load] = EvolverLoadShipment.when(seattle)
    assert load.then(seattle)[1] == pytest.approx(60)

    san_francisco = world_with(("truck1", "SFO"), ("shipment1", "truck1")).snapshot
    [unload] = EvolverUnloadShipment.when(san_francisco)
    assert unload.then(san_francisco)[1] == pytest.approx(80)


def test_shipment_cannot_be_loaded_into_a_vehicle_elsewhere():
    snapshot = world_with(("truck1", "SEA"), ("shipment1", "terminal_seattle")).snapshot
    assert not moves(EvolverLoadShipment, snapshot)


# === action models


def test_each_evolver_carries_its_action_model():
    snapshot = world_with(("truck1", "terminal_seattle"), ("air1", "SEA"), ("shipment1", "terminal_seattle")).snapshot
    local = moves(EvolverDriveTruckLocal, snapshot)["drive truck1 locally from terminal_seattle to SEA"].action
    assert isinstance(local, DriveLocalAction)
    assert (local.truck.name, local.from_location.name, local.to_location.name, local.cost) == (
        "truck1", "terminal_seattle", "SEA", 50
    )
    drive = moves(EvolverDriveTruckLongDistance, snapshot)["drive truck1 from terminal_seattle to terminal_boston"].action
    assert isinstance(drive, DriveLongDistanceAction) and drive.truck.name == "truck1"
    assert drive.cost == pytest.approx(0.8 * 3000)
    flight = moves(EvolverFlyAirplane, snapshot)["fly air1 from SEA to BOS"].action
    assert isinstance(flight, FlyAction)
    assert (flight.airplane.name, flight.to_location.name, flight.cost) == ("air1", "BOS", 809)
    load = moves(EvolverLoadShipment, snapshot)["load shipment1 into truck1 at terminal_seattle"].action
    assert isinstance(load, LoadAction)
    assert (load.shipment.name, load.vehicle.name, load.location.name, load.cost) == (
        "shipment1", "truck1", "terminal_seattle", 60
    )


def test_action_cost_matches_the_cost_of_the_move():
    snapshot = world_with(("truck1", "terminal_seattle"), ("shipment1", "truck1")).snapshot
    for evolver in (EvolverDriveTruckLocal, EvolverDriveTruckLongDistance, EvolverUnloadShipment):
        for option in evolver.when(snapshot):
            assert option.then(snapshot)[1] == option.action.cost
    unload = next(iter(EvolverUnloadShipment.when(snapshot))).action
    assert isinstance(unload, UnloadAction)
    assert (unload.shipment.name, unload.vehicle.name, unload.location.name, unload.cost) == (
        "shipment1", "truck1", "terminal_seattle", 60
    )


# === reasoner


def test_reasoner_world_is_the_static_configuration_plus_the_initial_state():
    static = Reasoner.configure_world_graph().snapshot
    world = Reasoner().world.snapshot
    assert static <= world
    assert world.items - static.items == set(world.by(At))


def test_reasoner_starts_with_every_vehicle_placed_and_the_shipment_nowhere():
    snapshot = Reasoner().world.snapshot
    assert {thing: location_of(snapshot, thing) for thing in ("truck1", "truck2", "air1")} == {
        "truck1": "terminal_seattle", "truck2": "terminal_san_francisco", "air1": "SEA",
    }
    assert not list(snapshot.by(At, lambda a: a.entity1.name == "shipment1"))


def test_each_reasoner_maintains_its_own_world_graph():
    assert Reasoner().world is not Reasoner().world


def test_planner_plans_over_the_world_it_is_given():
    world = Reasoner().world
    assert LogisticsPlanner(world).world is world


# === planner


@pytest.fixture
def planned() -> tuple[list[Any], float]:
    """a plan for the shipment from the Seattle terminal to the San Francisco terminal"""
    plan, cost = to_san_francisco(shipment_in_seattle(LogisticsPlanner(Reasoner().world))).plan()
    return list(plan), cost


def to_san_francisco(lp: LogisticsPlanner) -> LogisticsPlanner:
    """sets the planner's goal to shipment1 at the San Francisco terminal"""
    lp.set_goal([At(entity1=entity(lp.world, "shipment1"), entity2=entity(lp.world, "terminal_san_francisco"))])
    return lp


def shipment_in_seattle(lp: LogisticsPlanner) -> LogisticsPlanner:
    """sets the planner's current facts to shipment1 at the Seattle terminal"""
    lp.set_current([At(entity1=entity(lp.world, "shipment1"), entity2=entity(lp.world, "terminal_seattle"))])
    return lp


def test_planner_starts_with_no_current_facts():
    assert LogisticsPlanner(Reasoner().world).current is None


def test_planning_starts_from_the_world_graph_without_current_facts():
    world = Reasoner().world
    assert LogisticsPlanner(world).start() == world.snapshot


def test_current_facts_are_added_to_the_world_graph_and_planning_starts_from_them():
    world = Reasoner().world
    lp = shipment_in_seattle(LogisticsPlanner(world))
    assert location_of(world.snapshot, "shipment1") == "terminal_seattle"
    assert location_of(lp.start(), "shipment1") == "terminal_seattle"


def test_without_knowing_where_the_shipment_is_there_is_no_plan():
    with pytest.raises(RuntimeError, match="Plan not available"):
        to_san_francisco(LogisticsPlanner(Reasoner().world)).plan()


def test_current_facts_change_where_the_plan_starts():
    """with the shipment at the Denver terminal, the first load happens there, and the plan costs more than from
    Seattle because a truck must first drive to Denver"""
    world = Reasoner().world
    lp = to_san_francisco(LogisticsPlanner(world))
    lp.set_current([At(entity1=entity(world, "shipment1"), entity2=entity(world, "terminal_denver"))])
    plan, cost = lp.plan()
    [first_load, *_] = [a for a in plan if isinstance(a, LoadAction)]
    assert first_load.location.name == "terminal_denver"
    assert isinstance(plan[-1], UnloadAction) and plan[-1].location.name == "terminal_san_francisco"
    assert cost > 780


def test_planner_starts_with_no_goal():
    assert LogisticsPlanner(Reasoner().world).goal is None


def test_planning_without_a_goal_fails():
    lp = shipment_in_seattle(LogisticsPlanner(Reasoner().world))
    with pytest.raises(AssertionError, match="set_goal"):
        lp.plan()


def test_set_goal_sets_where_the_plan_ends():
    world = Reasoner().world
    lp = shipment_in_seattle(LogisticsPlanner(world))
    lp.set_goal([At(entity1=entity(world, "shipment1"), entity2=entity(world, "terminal_denver"))])
    plan, _ = lp.plan()
    assert isinstance(plan[-1], UnloadAction) and plan[-1].location.name == "terminal_denver"


def test_planner_returns_action_models(planned):
    plan, _ = planned
    assert [type(a) for a in plan] == [LoadAction, DriveLongDistanceAction, UnloadAction]


def test_planner_finds_the_cheapest_plan(planned):
    """with handling at 60-80 per move of the 1000 lb shipment, driving straight to San Francisco from the Seattle
    terminal, where the truck starts (load 60, drive 800 miles 640, unload 80), beats flying, which needs local
    drives and handles the shipment 3 more times"""
    plan, cost = planned
    assert cost == pytest.approx(780)
    [drive] = [a for a in plan if isinstance(a, DriveLongDistanceAction)]
    assert (drive.from_location.name, drive.to_location.name) == ("terminal_seattle", "terminal_san_francisco")


def test_planner_cost_is_the_sum_of_its_actions(planned):
    plan, cost = planned
    assert sum(a.cost for a in plan) == cost


def test_planner_ends_at_the_san_francisco_terminal(planned):
    plan, _ = planned
    assert isinstance(plan[-1], UnloadAction)
    assert plan[-1].location.name == "terminal_san_francisco"


def test_goal_check_needs_every_goal_fact():
    world = Reasoner().world
    lp = LogisticsPlanner(world)
    lp.set_goal([
        At(entity1=entity(world, "truck1"), entity2=entity(world, "terminal_seattle")),
        At(entity1=entity(world, "shipment1"), entity2=entity(world, "terminal_san_francisco")),
    ])
    assert not lp.create_goal_check()(world.snapshot)

    lp.set_goal([At(entity1=entity(world, "truck1"), entity2=entity(world, "terminal_seattle"))])
    assert lp.create_goal_check()(world.snapshot)


def test_planner_satisfies_every_goal_fact():
    """the shipment must reach the San Francisco terminal and the airplane must end at SFO, so flying the shipment
    along (879) beats driving it and flying the airplane empty (830 + 309)"""
    world = Reasoner().world
    lp = shipment_in_seattle(LogisticsPlanner(world))
    lp.set_goal([
        At(entity1=entity(world, "shipment1"), entity2=entity(world, "terminal_san_francisco")),
        At(entity1=entity(world, "air1"), entity2=entity(world, "SFO")),
    ])
    plan, cost = lp.plan()
    [flight] = [a for a in plan if isinstance(a, FlyAction)]
    assert (flight.from_location.name, flight.to_location.name) == ("SEA", "SFO")
    assert isinstance(plan[-1], UnloadAction) and plan[-1].location.name == "terminal_san_francisco"
    assert cost == pytest.approx(879)


def test_the_starting_world_has_no_shipments(starting_world):
    assert not list(starting_world.snapshot.by(Shipment))


def test_a_shipping_query_names_an_origin_terminal_a_destination_terminal_and_a_shipment():
    reasoner = Reasoner()
    seattle, denver = reasoner.find_entity("terminal_seattle"), reasoner.find_entity("terminal_denver")
    crate = reasoner.find_entity("shipment1")
    query = ShippingQuery(origin=seattle, destination=denver, shipment=crate)
    assert (query.origin.name, query.destination.name, query.shipment.name) == ("terminal_seattle", "terminal_denver", "shipment1")


def test_a_shipping_query_needs_another_terminal_as_its_destination():
    reasoner = Reasoner()
    seattle = reasoner.find_entity("terminal_seattle")
    with pytest.raises(ValidationError, match="another terminal than the origin"):
        ShippingQuery(origin=seattle, destination=seattle, shipment=reasoner.find_entity("shipment1"))


@pytest.mark.parametrize("field, name", [("origin", "SEA"), ("destination", "SFO"), ("shipment", "truck1")])
def test_a_shipping_query_takes_only_terminals_and_a_shipment(field, name):
    reasoner = Reasoner()
    parts = {"origin": reasoner.find_entity("terminal_seattle"), "destination": reasoner.find_entity("terminal_denver"),
             "shipment": reasoner.find_entity("shipment1")}
    with pytest.raises(ValidationError):
        ShippingQuery(**{**parts, field: reasoner.find_entity(name)})


def test_terminals_and_airports_are_locations_with_a_city(starting_world):
    locations = list(starting_world.snapshot.by(Location))
    assert len(locations) == 10 and all(isinstance(location, Terminal | Airport) for location in locations)
    assert {location.name for location in locations if isinstance(location, Airport)} == {"SEA", "SFO", "BOS", "DEN", "DTW"}
    assert all(location.city for location in locations)
    cities = {city.name for city in starting_world.snapshot.by(City)}
    assert sorted(location.city.name for location in locations if isinstance(location, Terminal)) == sorted(cities)
    assert sorted(location.city.name for location in locations if isinstance(location, Airport)) == sorted(cities)


def test_location_fields_are_described():
    assert set(Location.model_fields) == {"name", "city"}
    assert issubclass(Terminal, Location) and issubclass(Airport, Location)
    assert Location.model_fields["city"].description


def test_a_shipping_request_goes_between_any_two_locations():
    find = Reasoner().find_entity
    request = ShippingRequest(origin=find("terminal_seattle"), destination=find("SFO"), shipment=find("shipment1"))
    assert (request.origin.name, request.destination.name, request.shipment.name) == ("terminal_seattle", "SFO", "shipment1")
    assert set(ShippingRequest.model_fields) == {"origin", "destination", "shipment"}


@pytest.mark.parametrize("field, name", [("origin", "truck1"), ("destination", "seattle"), ("shipment", "air1")])
def test_a_shipping_request_takes_only_locations_and_a_shipment(field, name):
    find = Reasoner().find_entity
    parts = {"origin": find("terminal_seattle"), "destination": find("SFO"), "shipment": find("shipment1")}
    with pytest.raises(ValidationError):
        ShippingRequest(**{**parts, field: find(name)})


def test_clear_keeps_entities_that_were_already_in_the_world_graph():
    reasoner = Reasoner()
    lp = shipment_in_seattle(LogisticsPlanner(reasoner.world))
    lp.clear()
    assert reasoner.find_entity("shipment1") is not None
    assert not list(reasoner.world.snapshot.by(At, lambda a: a.entity1.name == "shipment1"))


def test_setting_the_current_facts_again_replaces_the_earlier_ones():
    reasoner = Reasoner()
    find = reasoner.find_entity
    lp = shipment_in_seattle(LogisticsPlanner(reasoner.world))
    lp.set_current([At(entity1=find("shipment1"), entity2=find("terminal_boston"))])
    [at] = reasoner.world.snapshot.by(At, lambda a: a.entity1.name == "shipment1")
    assert at.entity2.name == "terminal_boston"


def new_delivery(reasoner: Reasoner) -> tuple[Location, Location, Shipment]:
    """a customer's pickup location in Seattle, drop-off location in San Francisco, and a pallet of chairs to send"""
    find = reasoner.find_entity
    pickup = Location(name="pickup_seattle", city=find("seattle"))
    dropoff = Location(name="dropoff_san_francisco", city=find("san_francisco"))
    chairs = Shipment(name="chairs", details="pallet of office chairs", weight=800)
    return pickup, dropoff, chairs


def test_setup_adds_the_pickup_drop_off_and_shipment_to_the_world_graph():
    reasoner = Reasoner()
    LogisticsPlanner(reasoner.world).setup(*new_delivery(reasoner))
    assert reasoner.find_entity("pickup_seattle").city.name == "seattle"
    assert reasoner.find_entity("dropoff_san_francisco").city.name == "san_francisco"
    assert reasoner.find_entity("chairs").weight == 800
    assert not list(reasoner.world.snapshot.by(At, lambda a: a.entity1.name == "chairs"))


def test_setup_skips_what_is_already_in_the_world_graph():
    reasoner = Reasoner()
    find = reasoner.find_entity
    lp = LogisticsPlanner(reasoner.world)
    lp.setup(find("terminal_seattle"), find("terminal_denver"), find("shipment1"))
    lp.clear()
    assert all(find(name) is not None for name in ("terminal_seattle", "terminal_denver", "shipment1"))


def test_set_goal_does_not_change_the_world_graph():
    reasoner = Reasoner()
    pickup, dropoff, chairs = new_delivery(reasoner)
    lp = LogisticsPlanner(reasoner.world)
    lp.setup(pickup, dropoff, chairs)
    before = reasoner.world.snapshot
    lp.set_goal([At(entity1=chairs, entity2=dropoff)])
    assert reasoner.world.snapshot == before and lp.goal is not None


def test_clear_takes_out_what_setup_and_set_current_added():
    reasoner = Reasoner()
    before = reasoner.world.snapshot
    pickup, dropoff, chairs = new_delivery(reasoner)
    lp = LogisticsPlanner(reasoner.world)
    lp.setup(pickup, dropoff, chairs)
    lp.set_current([At(entity1=chairs, entity2=pickup)])
    lp.set_goal([At(entity1=chairs, entity2=dropoff)])
    lp.clear()
    assert reasoner.world.snapshot == before and lp.current is None and lp.goal is None


def test_a_shipment_is_planned_from_a_new_pickup_to_a_new_drop_off():
    """driving the 800 miles beats flying (795, plus 50 for truck2 to first drive from the San Francisco terminal to SFO):
    truck from the Seattle terminal to the pickup 50, load 48, drive to the drop-off 640, unload 64"""
    reasoner = Reasoner()
    pickup, dropoff, chairs = new_delivery(reasoner)
    lp = LogisticsPlanner(reasoner.world)
    lp.setup(pickup, dropoff, chairs)
    lp.set_current([At(entity1=chairs, entity2=pickup)])
    lp.set_goal([At(entity1=chairs, entity2=dropoff)])
    plan, cost = lp.plan()
    lp.clear()
    assert isinstance(plan[-1], UnloadAction) and plan[-1].location.name == "dropoff_san_francisco"
    assert not any(isinstance(a, FlyAction) for a in plan) and cost == pytest.approx(802.0)
