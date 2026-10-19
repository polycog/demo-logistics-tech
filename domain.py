from cognition import Entity, BinaryRelation
from enum import StrEnum
from pydantic import BaseModel, Field

class City(StrEnum):
    """cities in the logistics network; each city has one office and one airport"""

    SEATTLE = "seattle"
    SAN_FRANCISCO = "san_francisco"
    BOSTON = "boston"
    DENVER = "denver"
    DETROIT = "detroit"


class Office(Entity):
    """office location in a city, where packages start and end their journeys

    trucks can drive between an office and the airport in the same city,
    and between offices in different cities
    """
    display_name: str = Field(description="name of the office as customers know it")
    city: City
    address: str = Field(description="street address of the office")

class Airport(Entity):
    """airport location in a city, the only place airplanes can be

    trucks connect an airport to the office in the same city;
    airplanes fly between airports connected by an air route
    """
    display_name: str = Field(description="full name of the airport")
    city: City

class Package(Entity):
    """shipment to be moved through the network

    a package is either at a location (office or airport) or inside a vehicle,
    and can only travel by being loaded into a vehicle
    """
    display_name: str = Field(description="short description of what is being shipped")
    weight: float = Field(description="weight of the package, in pounds")

class Vehicle(Entity):
    """base type for anything that can carry packages

    a vehicle is always at a location (office or airport); packages inside it
    move along with it
    """
    display_name: str = Field(description="name the carrier uses for the vehicle")
    model: str = Field(description="make and model of the vehicle")
    max_payload: float = Field(description="heaviest load the vehicle can carry, in pounds")

class Truck(Vehicle):
    """vehicle that travels by road

    moves between an office and the airport in the same city,
    or between offices in different cities
    """
    license_plate: str

class Airplane(Vehicle):
    """vehicle that travels by air

    stays at airports and flies between airports connected by an air route
    """
    tail_number: str = Field(description="registration number painted on the airplane")

class TruckRoute(BinaryRelation):
    """road connection between offices in two different cities

    declared once per pair of offices; distance is the length of the drive
    """
    entity1: Office = Field(frozen=True)
    entity2: Office = Field(frozen=True)
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
    """where a package or vehicle currently is

    entity1 is the package or vehicle; entity2 is a location (office or airport),
    or a vehicle when a package has been loaded into it
    """
    entity1: Package | Vehicle = Field(frozen=True)
    entity2: Office | Airport | Vehicle = Field(frozen=True)

class GasPrice(Entity):
    """what fuel costs the carrier for every mile a truck drives"""
    per_mile: float = Field(description="fuel cost per mile driven, in US dollars")

class ShortDistanceFactor(Entity):
    per_mile: float

class LongDistanceFactor(Entity):
    per_mile: float

class LaborCost(Entity):
    """what the carrier pays its crew in a city to handle packages, such as loading and unloading"""
    by_weight: float = Field(description="labor cost per pound of package handled, in US dollars")
    city: City


class Action(BaseModel):
    """an action the agent takes to deliver a package; cost is the price of carrying it out"""
    cost: float = Field(description="price of carrying out the action, in US dollars")

class DriveLocalAction(Action):
    """drive a truck between an office and the airport in the same city"""
    truck: Truck
    from_location: Office | Airport
    to_location: Office | Airport

class DriveLongDistanceAction(Action):
    """drive a truck between offices in different cities"""
    truck: Truck
    from_location: Office
    to_location: Office

class FlyAction(Action):
    """fly an airplane between two airports connected by an air route"""
    airplane: Airplane
    from_location: Airport
    to_location: Airport

class LoadAction(Action):
    """load a package into a vehicle"""
    package: Package
    vehicle: Truck | Airplane
    location: Office | Airport

class UnloadAction(Action):
    """unload a package from a vehicle"""
    package: Package
    vehicle: Truck | Airplane
    location: Office | Airport
