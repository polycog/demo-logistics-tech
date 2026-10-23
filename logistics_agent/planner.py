from anyio.lowlevel import T
from cognition import SearchPlannerDynamicOption as WorldEvolver, WorldSnapshot, SearchPlanner, WorldGraph, dynamic_opts_succession
from typing import Any, cast, Iterable, Self, Callable, Sequence

from cognition.reasoning.planning import FrontierManager, PriorityQueue
from cognition.util.functypes import Predicate, Supplier
from logistics_agent.domain import GasPrice, LaborCost, Shipment, Vehicle, Truck, Airplane, Location, Terminal, Airport, At, AirRoute, TruckRoute
from logistics_agent.domain import Action, DriveLocalAction, DriveLongDistanceAction, FlyAction, LoadAction, UnloadAction
from logistics_agent.domain import ShippingRequest


class EvolverDriveTruckLocal(WorldEvolver[WorldSnapshot, Action]):
    """drive a truck between a terminal and an airport within the same city

    when: a truck is at a terminal or airport, and there is another terminal or airport in the same city
    then: the truck is at the other location
    cost: 50
    """

    def __init__(self, truck: Truck, from_location: Location, to_location: Location) -> None:
        """a local drive of a truck from one location to another in the same city"""
        super().__init__(DriveLocalAction(truck=truck, from_location=from_location, to_location=to_location, cost=50))
        self.truck = truck
        self.from_location = from_location
        self.to_location = to_location
        self.cost = 50

    @classmethod
    def when(cls, state: WorldSnapshot) -> Iterable[Self]:
        """yields a local drive for every truck at a terminal or airport, to each other location in its city"""
        ats = state.by(At, lambda a: isinstance(a.entity1, Truck) and isinstance(a.entity2, Location))
        for a in ats:
            truck = cast(Truck, a.entity1)
            from_location = cast(Location, a.entity2)
            for loc in state.by(Location):
                if loc != from_location and loc.city.name == from_location.city.name:
                    to_location = cast(Location, loc)
                    yield cls(truck, from_location, to_location)

    def then(self, state: WorldSnapshot) -> tuple[WorldSnapshot, int | float]:
        """moves the truck to the destination; returns the new state and the cost of the drive"""
        old_at_rel = next(iter(state.by(At, lambda a: a.entity1 == self.truck and a.entity2 == self.from_location)))
        new_at_rel = At(entity1=self.truck, entity2=self.to_location)
        new_state = state.copy(add=[new_at_rel], remove=[old_at_rel])
        return new_state, self.cost


class EvolverDriveTruckLongDistance(WorldEvolver[WorldSnapshot, Action]):
    """drive a truck between locations in different cities, where each city has a terminal

    when: a truck is at a location in a city with a terminal, and a truck route connects that city to another city
          with a terminal (in either direction)
    then: the truck is at a location in the other city
    cost: gas price per mile * distance of the truck route
    """

    def __init__(self, truck: Truck, from_location: Location, to_location: Location, cost: float) -> None:
        """a long-distance drive of a truck between locations, with its cost already computed"""
        super().__init__(DriveLongDistanceAction(truck=truck, from_location=from_location, to_location=to_location, cost=cost))
        self.truck = truck
        self.from_location = from_location
        self.to_location = to_location
        self.cost = cost

    @classmethod
    def when(cls, state: WorldSnapshot) -> Iterable[Self]:
        """yields a drive for every truck at a location in a city with a terminal, along each truck route from that
        city to every location in the other city, when that city has a terminal too; costs gas price per mile *
        route distance (requires a GasPrice entity in the state)"""
        gas_price = state.find_first(GasPrice)
        has_terminal = {t.city.name for t in state.by(Terminal)}
        ats = state.by(At, lambda a: isinstance(a.entity1, Truck) and isinstance(a.entity2, Location))
        for a in ats:
            truck = cast(Truck, a.entity1)
            from_location = cast(Location, a.entity2)
            city = from_location.city.name
            if city not in has_terminal:
                continue
            # truck routes are declared once per pair of cities, so they can be driven in either direction
            for route in state.by(TruckRoute, lambda r: city in (r.entity1.name, r.entity2.name)):
                to_city = route.entity2 if route.entity1.name == city else route.entity1
                if to_city.name not in has_terminal:
                    continue
                for to_location in state.by(Location, lambda l: l.city.name == to_city.name):
                    yield cls(truck, from_location, to_location, gas_price.per_mile * route.distance)

    def then(self, state: WorldSnapshot) -> tuple[WorldSnapshot, int | float]:
        """moves the truck to the destination location; returns the new state and the cost of the drive"""
        old_at_rel = next(iter(state.by(At, lambda a: a.entity1 == self.truck and a.entity2 == self.from_location)))
        new_at_rel = At(entity1=self.truck, entity2=self.to_location)
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


def handling_cost(state: WorldSnapshot, shipment: Shipment, location: Location) -> float:
    """what the crew at the location charges to load or unload the shipment: shipment weight * the labor rate
    of the location's city (requires a LaborCost entity for that city in the state)"""
    labor = next(iter(state.by(LaborCost, lambda l: l.city.name == location.city.name)))
    return shipment.weight * labor.by_weight


class EvolverLoadShipment(WorldEvolver[WorldSnapshot, Action]):
    """load a shipment into a vehicle

    when: a vehicle is at a location, and a shipment is at the same location
    then: the shipment is in the vehicle, and no longer at the location
    cost: shipment weight * labor rate of the location's city
    """

    def __init__(self, shipment: Shipment, vehicle: Vehicle, location: Location, cost: float) -> None:
        """loading of a shipment into a vehicle at a location, with its cost already computed"""
        super().__init__(LoadAction(shipment=shipment, vehicle=vehicle, location=location, cost=cost))
        self.shipment = shipment
        self.vehicle = vehicle
        self.location = location
        self.cost = cost

    @classmethod
    def when(cls, state: WorldSnapshot) -> Iterable[Self]:
        """yields a load for every shipment at the same location as a vehicle, costing the shipment weight * the
        labor rate of the location's city"""
        vehicle_ats = state.by(At, lambda a: isinstance(a.entity1, Vehicle) and isinstance(a.entity2, Location))
        for va in vehicle_ats:
            vehicle = cast(Vehicle, va.entity1)
            location = cast(Location, va.entity2)
            for pa in state.by(At, lambda a: isinstance(a.entity1, Shipment) and a.entity2 == location):
                shipment = cast(Shipment, pa.entity1)
                yield cls(shipment, vehicle, location, handling_cost(state, shipment, location))

    def then(self, state: WorldSnapshot) -> tuple[WorldSnapshot, int | float]:
        """moves the shipment from the location into the vehicle; returns the new state and the cost of loading"""
        old_at_rel = next(iter(state.by(At, lambda a: a.entity1 == self.shipment and a.entity2 == self.location)))
        new_at_rel = At(entity1=self.shipment, entity2=self.vehicle)
        new_state = state.copy(add=[new_at_rel], remove=[old_at_rel])
        return new_state, self.cost


class EvolverUnloadShipment(WorldEvolver[WorldSnapshot, Action]):
    """unload a shipment from a vehicle

    when: a vehicle is at a location, and a shipment is in the vehicle
    then: the shipment is at the location, and no longer in the vehicle
    cost: shipment weight * labor rate of the location's city
    """

    def __init__(self, shipment: Shipment, vehicle: Vehicle, location: Location, cost: float) -> None:
        """unloading of a shipment from a vehicle at the vehicle's location, with its cost already computed"""
        super().__init__(UnloadAction(shipment=shipment, vehicle=vehicle, location=location, cost=cost))
        self.shipment = shipment
        self.vehicle = vehicle
        self.location = location
        self.cost = cost

    @classmethod
    def when(cls, state: WorldSnapshot) -> Iterable[Self]:
        """yields an unload for every shipment inside a vehicle that is at a terminal or airport, costing the shipment
        weight * the labor rate of the location's city"""
        vehicle_ats = state.by(At, lambda a: isinstance(a.entity1, Vehicle) and isinstance(a.entity2, Location))
        for va in vehicle_ats:
            vehicle = cast(Vehicle, va.entity1)
            location = cast(Location, va.entity2)
            for pa in state.by(At, lambda a: isinstance(a.entity1, Shipment) and a.entity2 == vehicle):
                shipment = cast(Shipment, pa.entity1)
                yield cls(shipment, vehicle, location, handling_cost(state, shipment, location))

    def then(self, state: WorldSnapshot) -> tuple[WorldSnapshot, int | float]:
        """moves the shipment from the vehicle to the location; returns the new state and the cost of unloading"""
        old_at_rel = next(iter(state.by(At, lambda a: a.entity1 == self.shipment and a.entity2 == self.vehicle)))
        new_at_rel = At(entity1=self.shipment, entity2=self.location)
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
        # what setup and set_current added to the world graph, so clear can take exactly that out again
        self._setup_entities: list[str] = []
        self._current_facts: list[At] = []

    def configure_world_model(self) -> None:
        self.dynamics = [EvolverDriveTruckLocal,
                                 EvolverFlyAirplane,
                                 EvolverLoadShipment,
                                EvolverUnloadShipment,
                                EvolverDriveTruckLongDistance]

    def setup(self, pickup: Location, dropoff: Location, shipment: Shipment) -> None:
        """adds the pickup and drop-off locations and the shipment to the world graph the planner shares with the
        reasoner, so the plan can reach them; skips any that are already in it, such as a terminal"""
        for entity in (pickup, dropoff, shipment):
            if self.world.snapshot.entity_by_name(entity.name) is None:
                self.world.add_entity(entity)
                self._setup_entities.append(entity.name)

    def set_current(self, facts: Iterable[At]) -> None:
        """sets the facts that hold at the start of the plan, such as where a shipment is now, and adds them to the
        world graph; the entities they name must be in it already; replaces the facts an earlier call added"""
        self._remove_current_facts()
        self.current = tuple(facts)
        for fact in self.current:
            self.world.add_relation(fact)
            self._current_facts.append(fact)

    def set_goal(self, facts: Iterable[At]) -> None:
        """sets the facts that must all hold at the end of the plan, such as where a shipment should be"""
        self.goal = tuple(facts)

    def clear(self) -> None:
        """takes out of the world graph the facts set_current added and the entities setup added, leaving what was
        there before them, and forgets the current facts and the goal"""
        self._remove_current_facts()
        for name in self._setup_entities:
            self.world.remove_node(name)
        self._setup_entities = []
        self.current = None
        self.goal = None

    def _remove_current_facts(self) -> None:
        """takes out of the world graph the facts set_current added"""
        for fact in self._current_facts:
            self.world.remove_edge(fact.entity1.name, fact.entity2.name, fact.type)
        self._current_facts = []

    def has_current(self) -> bool:
        """whether the current facts are set, such as where a shipment is now"""
        return self.current is not None

    def has_goal(self) -> bool:
        """whether the goal is set, such as where a shipment should be"""
        return self.goal is not None

    def can_plan(self) -> bool:
        """whether both the current facts and the goal are set, so planning can begin"""
        return self.has_current() and self.has_goal()

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
                                        EvolverLoadShipment,
                                        EvolverUnloadShipment,
                                        EvolverDriveTruckLongDistance)
            )
        planner.run()
        return [action for action in planner.plan], planner.plan_cost

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

    reasoner = Reasoner()
    find = reasoner.find_entity
    # a customer's pickup and drop-off locations, which are not terminals or airports, and a shipment to send
    pickup = Location(name="pickup_seattle", city=find("seattle"))
    dropoff = Location(name="dropoff_san_francisco", city=find("san_francisco"))
    chairs = Shipment(name="shipment1", details="pallet of office chairs", weight=800)

    request = ShippingRequest(origin=pickup, destination=dropoff, shipment=chairs)
    lp = LogisticsPlanner(reasoner.world)
    lp.setup(request.origin, request.destination, request.shipment)
    lp.set_current([At(entity1=request.shipment, entity2=request.origin)])
    lp.set_goal([At(entity1=request.shipment, entity2=request.destination)])
    try:
        plan, price = lp.plan()
    finally:
        lp.clear()
    for action in plan:
        print(action)
    print(f"{request.shipment.details} from {request.origin.city.display_name} to {request.destination.city.display_name}: "
          f"${price}")
