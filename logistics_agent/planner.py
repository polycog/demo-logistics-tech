from anyio.lowlevel import T
from cognition import SearchPlannerDynamicOption as WorldEvolver, WorldSnapshot, SearchPlanner, WorldGraph, dynamic_opts_succession
from typing import Any, cast, Iterable, Self, Callable, Sequence

from cognition.reasoning.planning import FrontierManager, PriorityQueue
from cognition.util.functypes import Predicate, Supplier
from logistics_agent.domain import GasPrice, LaborCost, Package, Vehicle, Truck, Airplane, Office, Airport, At, AirRoute, TruckRoute
from logistics_agent.domain import Action, DriveLocalAction, DriveLongDistanceAction, FlyAction, LoadAction, UnloadAction


class EvolverDriveTruckLocal(WorldEvolver[WorldSnapshot, Action]):
    """drive a truck between an office and an airport within the same city

    when: a truck is at an office or airport, and there is another office or airport in the same city
    then: the truck is at the other location
    cost: 50
    """

    def __init__(self, truck: Truck, from_location: Office | Airport, to_location: Office | Airport) -> None:
        """a local drive of a truck from one location to another in the same city"""
        super().__init__(DriveLocalAction(truck=truck, from_location=from_location, to_location=to_location, cost=50))
        self.truck = truck
        self.from_location = from_location
        self.to_location = to_location
        self.cost = 50

    @classmethod
    def when(cls, state: WorldSnapshot) -> Iterable[Self]:
        """yields a local drive for every truck at an office or airport, to each other location in its city"""
        ats = state.by(At, lambda a: isinstance(a.entity1, Truck) and isinstance(a.entity2, Office | Airport))
        for a in ats:
            truck = cast(Truck, a.entity1)
            from_location = cast(Office | Airport, a.entity2)
            for loc in state.by((Office, Airport)):
                if loc != from_location and loc.city == from_location.city:
                    to_location = cast(Office | Airport, loc)
                    yield cls(truck, from_location, to_location)

    def then(self, state: WorldSnapshot) -> tuple[WorldSnapshot, int | float]:
        """moves the truck to the destination; returns the new state and the cost of the drive"""
        old_at_rel = next(iter(state.by(At, lambda a: a.entity1 == self.truck and a.entity2 == self.from_location)))
        new_at_rel = At(entity1=self.truck, entity2=self.to_location)
        new_state = state.copy(add=[new_at_rel], remove=[old_at_rel])
        return new_state, self.cost


class EvolverDriveTruckLongDistance(WorldEvolver[WorldSnapshot, Action]):
    """drive a truck between offices in different cities

    when: a truck is at an office, and a truck route connects it to another office (in either direction)
    then: the truck is at the other office
    cost: gas price per mile * distance of the truck route
    """

    def __init__(self, truck: Truck, from_office: Office, to_office: Office, cost: float) -> None:
        """a long-distance drive of a truck between offices, with its cost already computed"""
        super().__init__(DriveLongDistanceAction(truck=truck, from_location=from_office, to_location=to_office, cost=cost))
        self.truck = truck
        self.from_office = from_office
        self.to_office = to_office
        self.cost = cost

    @classmethod
    def when(cls, state: WorldSnapshot) -> Iterable[Self]:
        """yields a drive for every truck at an office, along each truck route from that office,
        costing gas price per mile * route distance (requires a GasPrice entity in the state)"""
        gas_price = state.find_first(GasPrice)
        ats = state.by(At, lambda a: isinstance(a.entity1, Truck) and isinstance(a.entity2, Office))
        for a in ats:
            truck = cast(Truck, a.entity1)
            from_office = cast(Office, a.entity2)
            # truck routes are declared once per pair of offices, so they can be driven in either direction
            for route in state.by(TruckRoute, lambda r: from_office in (r.entity1, r.entity2)):
                to_office = route.entity2 if route.entity1 == from_office else route.entity1
                yield cls(truck, from_office, to_office, gas_price.per_mile * route.distance)

    def then(self, state: WorldSnapshot) -> tuple[WorldSnapshot, int | float]:
        """moves the truck to the destination office; returns the new state and the cost of the drive"""
        old_at_rel = next(iter(state.by(At, lambda a: a.entity1 == self.truck and a.entity2 == self.from_office)))
        new_at_rel = At(entity1=self.truck, entity2=self.to_office)
        new_state = state.copy(add=[new_at_rel], remove=[old_at_rel])
        return new_state, self.cost


class EvolverFlyAirplane(WorldEvolver[WorldSnapshot, Action]):
    """fly an airplane between two airports connected by an air route

    when: an airplane is at an airport, and an air route connects it to another airport (in either direction)
    then: the airplane is at the other airport
    cost: price of the air route
    """

    def __init__(self, airplane: Airplane, from_airport: Airport, to_airport: Airport, price: int) -> None:
        """a flight of an airplane between two airports, costing the price of the air route"""
        super().__init__(FlyAction(airplane=airplane, from_location=from_airport, to_location=to_airport, cost=price))
        self.airplane = airplane
        self.from_airport = from_airport
        self.to_airport = to_airport
        self.cost = price

    @classmethod
    def when(cls, state: WorldSnapshot) -> Iterable[Self]:
        """yields a flight for every airplane at an airport, along each air route from that airport"""
        ats = state.by(At, lambda a: isinstance(a.entity1, Airplane) and isinstance(a.entity2, Airport))
        for a in ats:
            airplane = cast(Airplane, a.entity1)
            from_airport = cast(Airport, a.entity2)
            # air routes are declared once per pair of airports, so they can be flown in either direction
            for route in state.by(AirRoute, lambda r: from_airport in (r.entity1, r.entity2)):
                to_airport = route.entity2 if route.entity1 == from_airport else route.entity1
                yield cls(airplane, from_airport, to_airport, route.price)

    def then(self, state: WorldSnapshot) -> tuple[WorldSnapshot, int | float]:
        """moves the airplane to the destination airport; returns the new state and the price of the flight"""
        old_at_rel = next(iter(state.by(At, lambda a: a.entity1 == self.airplane and a.entity2 == self.from_airport)))
        new_at_rel = At(entity1=self.airplane, entity2=self.to_airport)
        new_state = state.copy(add=[new_at_rel], remove=[old_at_rel])
        return new_state, self.cost


def handling_cost(state: WorldSnapshot, package: Package, location: Office | Airport) -> float:
    """what the crew at the location charges to load or unload the package: package weight * the labor rate
    of the location's city (requires a LaborCost entity for that city in the state)"""
    labor = next(iter(state.by(LaborCost, lambda l: l.city == location.city)))
    return package.weight * labor.by_weight


class EvolverLoadPackage(WorldEvolver[WorldSnapshot, Action]):
    """load a package into a vehicle

    when: a vehicle is at a location, and a package is at the same location
    then: the package is in the vehicle, and no longer at the location
    cost: package weight * labor rate of the location's city
    """

    def __init__(self, package: Package, vehicle: Vehicle, location: Office | Airport, cost: float) -> None:
        """loading of a package into a vehicle at a location, with its cost already computed"""
        super().__init__(LoadAction(package=package, vehicle=vehicle, location=location, cost=cost))
        self.package = package
        self.vehicle = vehicle
        self.location = location
        self.cost = cost

    @classmethod
    def when(cls, state: WorldSnapshot) -> Iterable[Self]:
        """yields a load for every package at the same location as a vehicle, costing the package weight * the
        labor rate of the location's city"""
        vehicle_ats = state.by(At, lambda a: isinstance(a.entity1, Vehicle) and isinstance(a.entity2, Office | Airport))
        for va in vehicle_ats:
            vehicle = cast(Vehicle, va.entity1)
            location = cast(Office | Airport, va.entity2)
            for pa in state.by(At, lambda a: isinstance(a.entity1, Package) and a.entity2 == location):
                package = cast(Package, pa.entity1)
                yield cls(package, vehicle, location, handling_cost(state, package, location))

    def then(self, state: WorldSnapshot) -> tuple[WorldSnapshot, int | float]:
        """moves the package from the location into the vehicle; returns the new state and the cost of loading"""
        old_at_rel = next(iter(state.by(At, lambda a: a.entity1 == self.package and a.entity2 == self.location)))
        new_at_rel = At(entity1=self.package, entity2=self.vehicle)
        new_state = state.copy(add=[new_at_rel], remove=[old_at_rel])
        return new_state, self.cost


class EvolverUnloadPackage(WorldEvolver[WorldSnapshot, Action]):
    """unload a package from a vehicle

    when: a vehicle is at a location, and a package is in the vehicle
    then: the package is at the location, and no longer in the vehicle
    cost: package weight * labor rate of the location's city
    """

    def __init__(self, package: Package, vehicle: Vehicle, location: Office | Airport, cost: float) -> None:
        """unloading of a package from a vehicle at the vehicle's location, with its cost already computed"""
        super().__init__(UnloadAction(package=package, vehicle=vehicle, location=location, cost=cost))
        self.package = package
        self.vehicle = vehicle
        self.location = location
        self.cost = cost

    @classmethod
    def when(cls, state: WorldSnapshot) -> Iterable[Self]:
        """yields an unload for every package inside a vehicle that is at an office or airport, costing the package
        weight * the labor rate of the location's city"""
        vehicle_ats = state.by(At, lambda a: isinstance(a.entity1, Vehicle) and isinstance(a.entity2, Office | Airport))
        for va in vehicle_ats:
            vehicle = cast(Vehicle, va.entity1)
            location = cast(Office | Airport, va.entity2)
            for pa in state.by(At, lambda a: isinstance(a.entity1, Package) and a.entity2 == vehicle):
                package = cast(Package, pa.entity1)
                yield cls(package, vehicle, location, handling_cost(state, package, location))

    def then(self, state: WorldSnapshot) -> tuple[WorldSnapshot, int | float]:
        """moves the package from the vehicle to the location; returns the new state and the cost of unloading"""
        old_at_rel = next(iter(state.by(At, lambda a: a.entity1 == self.package and a.entity2 == self.vehicle)))
        new_at_rel = At(entity1=self.package, entity2=self.location)
        new_state = state.copy(add=[new_at_rel], remove=[old_at_rel])
        return new_state, self.cost

class LogisticsPlanner():

    def __init__(self, world: WorldGraph) -> None:
        """plans over the supplied world graph, which the caller builds and keeps up to date; starts with no goal
        and no current facts, which the caller sets before planning"""
        self.world = world
        self.configure_world_model()
        self.goal: tuple[At, ...] | None = None
        self.current: tuple[At, ...] | None = None

    def configure_world_model(self) -> None:
        self.dynamics = [EvolverDriveTruckLocal,
                                 EvolverFlyAirplane,
                                 EvolverLoadPackage,
                                EvolverUnloadPackage,
                                EvolverDriveTruckLongDistance]

    def set_goal(self, facts: Iterable[At]) -> None:
        """sets the facts that must all hold at the end of the plan, such as where a package should be"""
        self.goal = tuple(facts)

    def set_current(self, facts: Iterable[At]) -> None:
        """sets the facts that hold at the start of the plan on top of the world graph, such as where a package
        is now"""
        self.current = tuple(facts)

    def start(self) -> WorldSnapshot:
        """the snapshot planning starts from: the world graph, plus the current facts if they are set"""
        snapshot = self.world.snapshot
        if self.current is None:
            return snapshot
        return WorldSnapshot.click(*snapshot.items, *self.current)

    def plan(self) -> tuple[Sequence[Action], float]:
        planner = SearchPlanner(
                self.start(),
                self.create_goal_check(),
                dynamic_opts_succession(EvolverDriveTruckLocal,
                                        EvolverFlyAirplane,
                                        EvolverLoadPackage,
                                        EvolverUnloadPackage,
                                        EvolverDriveTruckLongDistance)
            )
        planner.run()
        return planner.plan, planner.plan_cost

    def create_goal_check(self) -> Callable[[WorldSnapshot], bool]:
        """builds the planner's goal test from the goal facts"""
        assert self.goal is not None, "set_goal must be called before planning"
        desired = WorldSnapshot.click(*self.goal)
        def reached_desired(snapshot: WorldSnapshot) -> bool:
            """returns True if the snapshot contains all of the desired facts"""
            return (desired <= snapshot)
        return reached_desired


if __name__ == "__main__":
    from logistics_agent.reasoner import Reasoner

    world = Reasoner().world
    package1 = world.get_entity("package1", linked=False)
    lp = LogisticsPlanner(world)
    lp.set_current([At(entity1=package1, entity2=world.get_entity("office_seattle", linked=False))])
    lp.set_goal([At(entity1=package1, entity2=world.get_entity("office_san_francisco", linked=False))])
    plan, cost = lp.plan()
    for action in plan:
        print(action)
    print(cost)