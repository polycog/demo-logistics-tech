from cognition import Entity, BinaryRelation
from pydantic import BaseModel, Field

class City(Entity):
    """city in the logistics network; each city has one terminal and one airport, and truck routes connect cities"""
    display_name: str = Field(description="name of the city")

class Location(Entity):
    """base type for any place in a city where shipments and vehicles can be: a terminal or an airport"""
    city: City = Field(description="city the location is in")

class Terminal(Location):
    """terminal location in a city, where shipments start and end their journeys

    trucks can drive between a terminal and the airport in the same city,
    and between terminals in different cities
    """
    display_name: str = Field(description="name of the terminal as customers know it")

class Airport(Location):
    """airport location in a city, the only place airplanes can be

    trucks connect an airport to the terminal in the same city;
    airplanes fly between airports connected by an air route
    """
    display_name: str = Field(description="full name of the airport")

class Shipment(Entity):
    """shipment to be moved through the network

    a shipment is either at a location (terminal or airport) or inside a vehicle,
    and can only travel by being loaded into a vehicle
    """
    details: str = Field(description="what is shipped, such as 'pallet of office chairs'")
    weight: float = Field(description="weight of the shipment, in pounds")

class Vehicle(Entity):
    """base type for anything that can carry shipments

    a vehicle is always at a location (terminal or airport); shipments inside it
    move along with it
    """
    display_name: str = Field(description="name the carrier uses for the vehicle")
    model: str = Field(description="make and model of the vehicle")
    max_payload: float = Field(description="heaviest load the vehicle can carry, in pounds")

class Truck(Vehicle):
    """vehicle that travels by road

    moves between a terminal and the airport in the same city,
    or between terminals in different cities
    """
    license_plate: str

class Airplane(Vehicle):
    """vehicle that travels by air

    stays at airports and flies between airports connected by an air route
    """
    tail_number: str = Field(description="registration number painted on the airplane")

class TruckRoute(BinaryRelation):
    """road connection between two different cities, driven between their terminals

    declared once per pair of cities and can be driven in either direction; distance is the length of the drive
    """
    entity1: City = Field(frozen=True)
    entity2: City = Field(frozen=True)
    distance: int = Field(default=0, description="length of the drive, in miles")

class AirRoute(BinaryRelation):
    """flight connection between two airports

    declared once per pair of airports and can be flown in either direction;
    price is the cost of flying the route
    """
    entity1: Airport = Field(frozen=True)
    entity2: Airport = Field(frozen=True)
    price: int = Field(description="cost of flying the route, in US dollars")

class At(BinaryRelation):
    """where a shipment or vehicle currently is

    entity1 is the shipment or vehicle; entity2 is a location (terminal or airport),
    or a vehicle when a shipment has been loaded into it
    """
    entity1: Shipment | Vehicle = Field(frozen=True)
    entity2: Location | Vehicle = Field(frozen=True)

class GasPrice(Entity):
    """what fuel costs the carrier for every mile a truck drives"""
    per_mile: float = Field(description="fuel cost per mile driven, in US dollars")

class ShortDistanceFactor(Entity):
    per_mile: float

class LongDistanceFactor(Entity):
    per_mile: float

class LaborCost(Entity):
    """what the carrier pays its crew in a city to handle shipments, such as loading and unloading"""
    by_weight: float = Field(description="labor cost per pound of shipment handled, in US dollars")
    city: City


# the docstring and the field descriptions are the questions the agent asks the human to fill in the request: the
# docstring for the whole request, each description for its part; a part is None until it is known
class ShippingRequest(BaseModel):
    """What are you shipping, how much does it weigh, and which cities is it going from and to?"""
    origin: Location | None = Field(default=None, description="Which city is the shipment picked up from?")
    destination: Location | None = Field(default=None, description="Which city is the shipment going to?")
    shipment: Shipment | None = Field(default=None, description="What are you shipping, and how much does it weigh?")

    def missing(self) -> list[str]:
        """the names of the parts not known yet, in the order origin, destination, shipment"""
        return [part for part in ("origin", "destination", "shipment") if getattr(self, part) is None]


class Action(BaseModel):
    """an action the agent takes to deliver a shipment; cost is the price of carrying it out"""
    cost: float = Field(description="price of carrying out the action, in US dollars")

class DriveLocalAction(Action):
    """drive a truck between a terminal and the airport in the same city"""
    truck: Truck
    from_location: Location
    to_location: Location

class DriveLongDistanceAction(Action):
    """drive a truck between locations in different cities, each city with a terminal"""
    truck: Truck
    from_location: Location
    to_location: Location

class FlyAction(Action):
    """fly an airplane between two airports connected by an air route"""
    airplane: Airplane
    from_location: Airport
    to_location: Airport

class LoadAction(Action):
    """load a shipment into a vehicle"""
    shipment: Shipment
    vehicle: Truck | Airplane
    location: Location

class UnloadAction(Action):
    """unload a shipment from a vehicle"""
    shipment: Shipment
    vehicle: Truck | Airplane
    location: Location
