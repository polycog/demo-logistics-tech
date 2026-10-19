"""the reasoner of the pricing support cogent, in charge of maintaining the world graph"""

from typing import Any

from cognition import BinaryRelation, Entity, WorldGraph

from domain import At, GasPrice, LaborCost, Package, Truck, Airplane, Office, Airport, AirRoute, TruckRoute, City


def _normalize(name: str) -> str:
    """a name in lowercase with underscores as spaces, so "Gas Price" and "gas_price" compare equal"""
    return name.replace("_", " ").strip().lower()


def _matches(entity: Entity, name: str | None) -> bool:
    """whether the name refers to the entity, by its name or its display name, ignoring case and underscores; None
    matches anything"""
    return name is None or _normalize(name) in (_normalize(entity.name), _normalize(getattr(entity, "display_name", "")))


class Reasoner:
    """maintains the world graph the cogent reasons about

    the world graph is built in two parts:
    static configuration: what does not change while the cogent runs: offices, airports, vehicles, the package,
        truck and air routes, the gas price and labor costs (configure_world_graph)
    initial state: where the package and each vehicle start, which changes as deliveries happen
        (set_initial_state)
    """

    def __init__(self) -> None:
        self.world = self.configure_world_graph()
        self.set_initial_state()

    @staticmethod
    def configure_world_graph() -> WorldGraph:
        """builds the static parts of the world: an office and airport in each city, truck and air routes between
        every pair of cities, two trucks, one airplane, a package, the gas price, and the labor cost in each city"""
        office_seattle = Office(name="office_seattle", display_name="Seattle Office", city=City.SEATTLE,
                                address="2400 Airport Way S, Seattle, WA")
        sea = Airport(name="SEA", display_name="Seattle-Tacoma International Airport", city=City.SEATTLE)
        office_san_francisco = Office(name="office_san_francisco", display_name="San Francisco Office", city=City.SAN_FRANCISCO,
                                      address="1550 Evans Ave, San Francisco, CA")
        sfo = Airport(name="SFO", display_name="San Francisco International Airport", city=City.SAN_FRANCISCO)
        office_boston = Office(name="office_boston", display_name="Boston Office", city=City.BOSTON,
                               address="100 Terminal St, Boston, MA")
        bos = Airport(name="BOS", display_name="Boston Logan International Airport", city=City.BOSTON)
        office_denver = Office(name="office_denver", display_name="Denver Office", city=City.DENVER,
                               address="4800 Dahlia St, Denver, CO")
        den = Airport(name="DEN", display_name="Denver International Airport", city=City.DENVER)
        office_detroit = Office(name="office_detroit", display_name="Detroit Office", city=City.DETROIT,
                                address="9000 Michigan Ave, Detroit, MI")
        dtw = Airport(name="DTW", display_name="Detroit Metropolitan Wayne County Airport", city=City.DETROIT)
        truck1 = Truck(name="truck1", display_name="Big Bertha", model="Freightliner M2 106 26ft box truck",
                       max_payload=10000, license_plate="C12345A")
        truck2 = Truck(name="truck2", display_name="Little Rocket", model="Isuzu NPR-HD 16ft box truck",
                       max_payload=6000, license_plate="C67890B")
        air1 = Airplane(name="air1", display_name="The Pelican", model="Cessna 408 SkyCourier freighter",
                        max_payload=6000, tail_number="N408CX")

        airroute1 = AirRoute(entity1=sea, entity2=sfo, price=309)
        airroute2 = AirRoute(entity1=sea, entity2=bos, price=809)
        airroute3 = AirRoute(entity1=sea, entity2=den, price=249)
        airroute4 = AirRoute(entity1=sfo, entity2=bos, price=639)
        airroute5 = AirRoute(entity1=sfo, entity2=den, price=229)
        airroute6 = AirRoute(entity1=bos, entity2=den, price=289)
        airroute7 = AirRoute(entity1=sea, entity2=dtw, price=364)
        airroute8 = AirRoute(entity1=sfo, entity2=dtw, price=629)
        airroute9 = AirRoute(entity1=bos, entity2=dtw, price=489)
        airroute10 = AirRoute(entity1=den, entity2=dtw, price=239)

        truckroute1 = TruckRoute(entity1=office_seattle, entity2=office_san_francisco, distance=800)
        truckroute2 = TruckRoute(entity1=office_seattle, entity2=office_denver, distance=1300)
        truckroute3 = TruckRoute(entity1=office_seattle, entity2=office_boston, distance=3000)
        truckroute4 = TruckRoute(entity1=office_san_francisco, entity2=office_denver, distance=1250)
        truckroute5 = TruckRoute(entity1=office_san_francisco, entity2=office_boston, distance=3125)
        truckroute6 = TruckRoute(entity1=office_denver, entity2=office_boston, distance=1970)
        truckroute7 = TruckRoute(entity1=office_seattle, entity2=office_detroit, distance=2300)
        truckroute8 = TruckRoute(entity1=office_san_francisco, entity2=office_detroit, distance=2400)
        truckroute9 = TruckRoute(entity1=office_boston, entity2=office_detroit, distance=725)
        truckroute10 = TruckRoute(entity1=office_denver, entity2=office_detroit, distance=1250)

        package1 = Package(name="package1", display_name="crate of machine parts", weight=1000)

        world = WorldGraph()
        world.add_entity(office_seattle)
        world.add_entity(sea)
        world.add_entity(office_san_francisco)
        world.add_entity(sfo)
        world.add_entity(office_boston)
        world.add_entity(bos)
        world.add_entity(office_denver)
        world.add_entity(den)
        world.add_entity(office_detroit)
        world.add_entity(dtw)
        world.add_entity(package1)
        world.add_entity(truck1)
        world.add_entity(truck2)
        world.add_entity(air1)
        world.add_relation(airroute1)
        world.add_relation(airroute2)
        world.add_relation(airroute3)
        world.add_relation(airroute4)
        world.add_relation(airroute5)
        world.add_relation(airroute6)
        world.add_relation(airroute7)
        world.add_relation(airroute8)
        world.add_relation(airroute9)
        world.add_relation(airroute10)
        world.add_relation(truckroute1)
        world.add_relation(truckroute2)
        world.add_relation(truckroute3)
        world.add_relation(truckroute4)
        world.add_relation(truckroute5)
        world.add_relation(truckroute6)
        world.add_relation(truckroute7)
        world.add_relation(truckroute8)
        world.add_relation(truckroute9)
        world.add_relation(truckroute10)

        world.add_entity(GasPrice(name="gas_price", per_mile=0.8))

        # handling crews cost more where wages are higher
        world.add_entity(LaborCost(name="labor_cost_seattle", city=City.SEATTLE, by_weight=0.06))
        world.add_entity(LaborCost(name="labor_cost_san_francisco", city=City.SAN_FRANCISCO, by_weight=0.08))
        world.add_entity(LaborCost(name="labor_cost_boston", city=City.BOSTON, by_weight=0.07))
        world.add_entity(LaborCost(name="labor_cost_denver", city=City.DENVER, by_weight=0.05))
        world.add_entity(LaborCost(name="labor_cost_detroit", city=City.DETROIT, by_weight=0.04))

        return world

    def set_initial_state(self) -> None:
        """places the package and each vehicle where they start: the package at the Seattle office, truck1 and
        air1 at SEA, and truck2 at SFO"""

        def entity(name: str) -> Any:
            return self.world.get_entity(name, linked=False)

        self.world.add_relation(At(entity1=entity("package1"), entity2=entity("office_seattle")))
        self.world.add_relation(At(entity1=entity("truck1"), entity2=entity("SEA")))
        self.world.add_relation(At(entity1=entity("truck2"), entity2=entity("SFO")))
        self.world.add_relation(At(entity1=entity("air1"), entity2=entity("SEA")))

    def find_entities[E: Entity](self, entity_type: type[E]) -> list[E]:
        """answers an entity query: every entity of the given type (subtypes included) in the world graph"""
        return list(self.world.snapshot.by(entity_type))

    def find_relations[R: BinaryRelation](
        self, relation_type: type[R], entity1: str | None = None, entity2: str | None = None
    ) -> list[R]:
        """answers a relation query: every relation of the given type in the world graph whose entity1 and entity2
        match the given names; a side left as None is unconstrained. a name matches an entity's name or its display
        name, ignoring case and underscores"""
        return list(self.world.snapshot.by(
            relation_type,
            lambda relation: _matches(relation.entity1, entity1) and _matches(relation.entity2, entity2),
        ))

    def find_entity(self, name: str) -> Entity | None:
        """the entity the name refers to, by its name or its display name ignoring case and underscores; None if there
        is none"""
        for entity in self.world.snapshot.by(Entity, lambda e: _matches(e, name)):
            return self.world.get_entity(entity.name, linked=False)
        return None

    def fresh_name(self, prefix: str) -> str:
        """an entity name not yet used in the world graph, made of the prefix and a number, such as package2"""
        taken = {e.name for e in self.world.snapshot.by(Entity)}
        n = 1
        while f"{prefix}{n}" in taken:
            n += 1
        return f"{prefix}{n}"

    def add_entity(self, entity: Entity, location: str | None = None) -> None:
        """adds a new entity to the world graph, at the named location if one is given; raises ValueError when the
        name is already taken, KeyError when the location refers to nothing, and pydantic.ValidationError when the
        entity cannot be at that location"""
        if self.world.snapshot.entity_by_name(entity.name) is not None:
            raise ValueError(f"an entity named {entity.name} already exists")
        at = None
        if location is not None:
            place = self.find_entity(location)
            if place is None:
                raise KeyError(location)
            at = At(entity1=entity, entity2=place)
        self.world.add_entity(entity)
        if at is not None:
            self.world.add_relation(at)

    def update_entity(self, entity: Entity) -> None:
        """replaces the field values of the existing entity with the same name, keeping its relations; raises
        KeyError when no entity has that name and TypeError when the existing one is of another type"""
        current = self.world.get_entity(entity.name, linked=False)
        if type(current) is not type(entity):
            raise TypeError(f"{entity.name} is a {type(current).__name__}, not a {type(entity).__name__}")
        self.world.add_entity(entity)
