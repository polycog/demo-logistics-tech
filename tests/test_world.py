"""offline tests of the world model: the static world graph, each evolver, and the logistics planner"""

from typing import Any

import pytest
from cognition import WorldGraph, WorldSnapshot

from domain import (
    AirRoute, Airplane, Airport, At, City, DriveLocalAction, DriveLongDistanceAction, FlyAction, GasPrice, LaborCost, LoadAction, Office,
    Package, Truck, TruckRoute, UnloadAction,
)
from reasoner import Reasoner
from planner import (
    EvolverDriveTruckLocal,
    EvolverDriveTruckLongDistance,
    EvolverFlyAirplane,
    EvolverLoadPackage,
    EvolverUnloadPackage,
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
            return f"drive {option.truck.name} from {option.from_office.name} to {option.to_office.name}"
        case EvolverFlyAirplane():
            return f"fly {option.airplane.name} from {option.from_airport.name} to {option.to_airport.name}"
        case EvolverLoadPackage():
            return f"load {option.package.name} into {option.vehicle.name} at {option.location.name}"
        case EvolverUnloadPackage():
            return f"unload {option.package.name} from {option.vehicle.name} at {option.location.name}"
    raise TypeError(option)


def moves(evolver: type, snapshot: WorldSnapshot) -> dict[str, Any]:
    """every option the evolver proposes in the snapshot, keyed by its label"""
    return {label(option): option for option in evolver.when(snapshot)}


def location_of(snapshot: WorldSnapshot, name: str) -> str:
    return next(iter(snapshot.by(At, lambda a: a.entity1.name == name))).entity2.name


# === static world graph


def test_static_world_has_every_entity():
    snapshot = Reasoner.configure_world_graph().snapshot
    assert {o.name for o in snapshot.by(Office)} == {
        "office_seattle", "office_san_francisco", "office_boston", "office_denver", "office_detroit"
    }
    assert {a.name for a in snapshot.by(Airport)} == {"SEA", "SFO", "BOS", "DEN", "DTW"}
    assert {t.name for t in snapshot.by(Truck)} == {"truck1", "truck2"}
    assert {a.name for a in snapshot.by(Airplane)} == {"air1"}
    assert {p.name for p in snapshot.by(Package)} == {"package1"}
    assert snapshot.find_first(GasPrice).per_mile == 0.8


def test_every_city_has_one_labor_cost():
    costs = list(Reasoner.configure_world_graph().snapshot.by(LaborCost))
    assert sorted(c.city for c in costs) == sorted(City)
    assert all(c.by_weight > 0 for c in costs)


def test_every_entity_has_a_human_readable_name():
    snapshot = Reasoner.configure_world_graph().snapshot
    for kind in (Office, Airport, Truck, Airplane, Package):
        for e in snapshot.by(kind):
            assert e.display_name and e.display_name != e.name


def test_vehicles_can_carry_the_package():
    snapshot = Reasoner.configure_world_graph().snapshot
    package = snapshot.find_first(Package)
    for vehicle in (*snapshot.by(Truck), *snapshot.by(Airplane)):
        assert vehicle.max_payload >= package.weight


def test_static_world_connects_every_pair_of_cities_once():
    snapshot = Reasoner.configure_world_graph().snapshot
    air_pairs = [frozenset((r.entity1.name, r.entity2.name)) for r in snapshot.by(AirRoute)]
    truck_pairs = [frozenset((r.entity1.name, r.entity2.name)) for r in snapshot.by(TruckRoute)]
    assert len(air_pairs) == len(set(air_pairs)) == 10
    assert len(truck_pairs) == len(set(truck_pairs)) == 10


def test_static_world_has_no_locations():
    """where packages and vehicles are is dynamic, so it is not part of the static configuration"""
    assert not list(Reasoner.configure_world_graph().snapshot.by(At))


# === evolvers


def test_local_drive_stays_within_the_city():
    snapshot = world_with(("truck1", "SEA")).snapshot
    assert set(moves(EvolverDriveTruckLocal, snapshot)) == {"drive truck1 locally from SEA to office_seattle"}


def test_local_drive_moves_the_truck():
    snapshot = world_with(("truck1", "SEA")).snapshot
    new, cost = moves(EvolverDriveTruckLocal, snapshot)["drive truck1 locally from SEA to office_seattle"].then(snapshot)
    assert location_of(new, "truck1") == "office_seattle"
    assert cost == 50


def test_long_distance_drive_goes_both_ways_along_truck_routes():
    snapshot = world_with(("truck1", "office_boston")).snapshot
    assert set(moves(EvolverDriveTruckLongDistance, snapshot)) == {
        f"drive truck1 from office_boston to office_{city}" for city in ("seattle", "san_francisco", "denver", "detroit")
    }


def test_long_distance_drive_costs_gas_price_times_distance():
    snapshot = world_with(("truck1", "office_seattle")).snapshot
    drive = moves(EvolverDriveTruckLongDistance, snapshot)["drive truck1 from office_seattle to office_san_francisco"]
    new, cost = drive.then(snapshot)
    assert location_of(new, "truck1") == "office_san_francisco"
    assert cost == pytest.approx(0.8 * 800)


def test_long_distance_drive_needs_an_office():
    assert not moves(EvolverDriveTruckLongDistance, world_with(("truck1", "SEA")).snapshot)


def test_flight_costs_the_route_price_in_either_direction():
    snapshot = world_with(("air1", "DTW")).snapshot
    flights = moves(EvolverFlyAirplane, snapshot)
    assert set(flights) == {f"fly air1 from DTW to {code}" for code in ("SEA", "SFO", "BOS", "DEN")}
    new, cost = flights["fly air1 from DTW to SEA"].then(snapshot)
    assert location_of(new, "air1") == "SEA"
    assert cost == 364


def test_load_and_unload_move_the_package_through_the_vehicle():
    snapshot = world_with(("truck1", "office_seattle"), ("package1", "office_seattle")).snapshot
    loaded, _ = moves(EvolverLoadPackage, snapshot)["load package1 into truck1 at office_seattle"].then(snapshot)
    assert location_of(loaded, "package1") == "truck1"
    assert not moves(EvolverLoadPackage, loaded)

    unloaded, cost = moves(EvolverUnloadPackage, loaded)["unload package1 from truck1 at office_seattle"].then(loaded)
    assert location_of(unloaded, "package1") == "office_seattle"
    assert cost == pytest.approx(1000 * 0.06)


def test_handling_costs_the_package_weight_times_the_city_labor_rate():
    """the 1000 lb package costs 60 to handle in Seattle (0.06 per lb) and 80 in San Francisco (0.08 per lb)"""
    seattle = world_with(("truck1", "office_seattle"), ("package1", "office_seattle")).snapshot
    [load] = EvolverLoadPackage.when(seattle)
    assert load.then(seattle)[1] == pytest.approx(60)

    san_francisco = world_with(("truck1", "SFO"), ("package1", "truck1")).snapshot
    [unload] = EvolverUnloadPackage.when(san_francisco)
    assert unload.then(san_francisco)[1] == pytest.approx(80)


def test_package_cannot_be_loaded_into_a_vehicle_elsewhere():
    snapshot = world_with(("truck1", "SEA"), ("package1", "office_seattle")).snapshot
    assert not moves(EvolverLoadPackage, snapshot)


# === action models


def test_each_evolver_carries_its_action_model():
    snapshot = world_with(("truck1", "office_seattle"), ("air1", "SEA"), ("package1", "office_seattle")).snapshot
    local = moves(EvolverDriveTruckLocal, snapshot)["drive truck1 locally from office_seattle to SEA"].action
    assert isinstance(local, DriveLocalAction)
    assert (local.truck.name, local.from_location.name, local.to_location.name, local.cost) == (
        "truck1", "office_seattle", "SEA", 50
    )
    drive = moves(EvolverDriveTruckLongDistance, snapshot)["drive truck1 from office_seattle to office_boston"].action
    assert isinstance(drive, DriveLongDistanceAction) and drive.truck.name == "truck1"
    assert drive.cost == pytest.approx(0.8 * 3000)
    flight = moves(EvolverFlyAirplane, snapshot)["fly air1 from SEA to BOS"].action
    assert isinstance(flight, FlyAction)
    assert (flight.airplane.name, flight.to_location.name, flight.cost) == ("air1", "BOS", 809)
    load = moves(EvolverLoadPackage, snapshot)["load package1 into truck1 at office_seattle"].action
    assert isinstance(load, LoadAction)
    assert (load.package.name, load.vehicle.name, load.location.name, load.cost) == (
        "package1", "truck1", "office_seattle", 60
    )


def test_action_cost_matches_the_cost_of_the_move():
    snapshot = world_with(("truck1", "office_seattle"), ("package1", "truck1")).snapshot
    for evolver in (EvolverDriveTruckLocal, EvolverDriveTruckLongDistance, EvolverUnloadPackage):
        for option in evolver.when(snapshot):
            assert option.then(snapshot)[1] == option.action.cost
    unload = next(iter(EvolverUnloadPackage.when(snapshot))).action
    assert isinstance(unload, UnloadAction)
    assert (unload.package.name, unload.vehicle.name, unload.location.name, unload.cost) == (
        "package1", "truck1", "office_seattle", 60
    )


# === reasoner


def test_reasoner_world_is_the_static_configuration_plus_the_initial_state():
    static = Reasoner.configure_world_graph().snapshot
    world = Reasoner().world.snapshot
    assert static <= world
    assert world.items - static.items == set(world.by(At))


def test_reasoner_starts_with_the_package_and_every_vehicle_placed():
    snapshot = Reasoner().world.snapshot
    assert {thing: location_of(snapshot, thing) for thing in ("package1", "truck1", "truck2", "air1")} == {
        "package1": "office_seattle", "truck1": "SEA", "truck2": "SFO", "air1": "SEA",
    }


def test_each_reasoner_maintains_its_own_world_graph():
    assert Reasoner().world is not Reasoner().world


def test_planner_plans_over_the_world_it_is_given():
    world = Reasoner().world
    assert LogisticsPlanner(world).world is world


# === planner


@pytest.fixture
def planned() -> tuple[list[Any], float]:
    """a plan toward the planner's default goal"""
    plan, cost = LogisticsPlanner(Reasoner().world).plan()
    return list(plan), cost


def test_planner_starts_with_package1_at_the_san_francisco_office():
    [goal] = LogisticsPlanner(Reasoner().world).goal
    assert (goal.entity1.name, goal.entity2.name) == ("package1", "office_san_francisco")


def test_set_goal_replaces_the_default_goal():
    world = Reasoner().world
    lp = LogisticsPlanner(world)
    lp.set_goal([At(entity1=entity(world, "package1"), entity2=entity(world, "office_denver"))])
    plan, _ = lp.plan()
    assert isinstance(plan[-1], UnloadAction) and plan[-1].location.name == "office_denver"


def test_planner_returns_action_models(planned):
    plan, _ = planned
    assert [type(a) for a in plan] == [DriveLocalAction, LoadAction, DriveLongDistanceAction, UnloadAction]


def test_planner_finds_the_cheapest_plan(planned):
    """with handling at 60-80 per move of the 1000 lb package, driving straight to San Francisco (truck to the
    Seattle office 50, load 60, drive 800 miles 640, unload 80) beats flying, which handles the package 3 more
    times (879 in total)"""
    plan, cost = planned
    assert cost == pytest.approx(830)
    [drive] = [a for a in plan if isinstance(a, DriveLongDistanceAction)]
    assert (drive.from_location.name, drive.to_location.name) == ("office_seattle", "office_san_francisco")


def test_planner_cost_is_the_sum_of_its_actions(planned):
    plan, cost = planned
    assert sum(a.cost for a in plan) == cost


def test_planner_ends_at_the_san_francisco_office(planned):
    plan, _ = planned
    assert isinstance(plan[-1], UnloadAction)
    assert plan[-1].location.name == "office_san_francisco"


def test_goal_check_needs_every_goal_fact():
    world = Reasoner().world
    lp = LogisticsPlanner(world)
    lp.set_goal([
        At(entity1=entity(world, "truck1"), entity2=entity(world, "SEA")),
        At(entity1=entity(world, "package1"), entity2=entity(world, "office_san_francisco")),
    ])
    assert not lp.create_goal_check()(world.snapshot)

    lp.set_goal([At(entity1=entity(world, "truck1"), entity2=entity(world, "SEA"))])
    assert lp.create_goal_check()(world.snapshot)


def test_planner_satisfies_every_goal_fact():
    """the package must reach the San Francisco office and the airplane must end at SFO, so flying the package
    along (879) beats driving it and flying the airplane empty (830 + 309)"""
    world = Reasoner().world
    lp = LogisticsPlanner(world)
    lp.set_goal([
        At(entity1=entity(world, "package1"), entity2=entity(world, "office_san_francisco")),
        At(entity1=entity(world, "air1"), entity2=entity(world, "SFO")),
    ])
    plan, cost = lp.plan()
    [flight] = [a for a in plan if isinstance(a, FlyAction)]
    assert (flight.from_location.name, flight.to_location.name) == ("SEA", "SFO")
    assert isinstance(plan[-1], UnloadAction) and plan[-1].location.name == "office_san_francisco"
    assert cost == pytest.approx(879)
