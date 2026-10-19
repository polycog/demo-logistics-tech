"""language understanding and generation for the pricing support agent, built on cognition's language tools"""

import os
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from cognition import DocEnum, Entity, WorldGraph
from pydantic import BaseModel, Field
from pydantic_ai.models import Model, infer_model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from logistics_agent.domain import Airplane, Airport, At, City, GasPrice, LaborCost, Office, Package, Truck, Vehicle

# the config lives at the root of the repository, next to pyproject.toml
LLM_CONFIG = Path(__file__).resolve().parent.parent / "llm_config.toml"


def load_llm(config_path: Path = LLM_CONFIG) -> Model:
    """builds the language model named by the config file: OpenAI's hosted model, or a local model served through an
    OpenAI-compatible API such as LM Studio; the LLM_PROVIDER environment variable overrides the file's provider"""
    with config_path.open("rb") as f:
        config = tomllib.load(f)
    provider = os.environ.get("LLM_PROVIDER", config["provider"])
    match provider:
        case "openai":
            return infer_model(f"openai:{config['openai']['model']}")
        case "local":
            local = config["local"]
            return OpenAIChatModel(
                local["model"], provider=OpenAIProvider(base_url=local["base_url"], api_key=local["api_key"])
            )
    raise ValueError(f"unknown LLM provider {provider!r}; expected 'openai' or 'local'")


DEFAULT_LLM: Model = load_llm()


@dataclass(frozen=True)
class Utterance:
    """natural language message exchanged with the human; phrase is None when nothing was said"""

    phrase: str | None


class HumanIntent(DocEnum):
    """what the human wants from the logistics support agent"""

    GET_PRICE_ESTIMATE = (
        "get_price_estimate",
        "human asks how much it would cost to deliver a package, for example from one city to another",
    )
    STATE_DESIRED_OUTCOME = (
        "state_desired_outcome",
        "human says what outcome they want, for example where a package should end up or by when",
    )
    ASK_ABOUT_ENTITIES = (
        "ask_about_entities",
        "human asks what things the carrier has or what they are, for example which trucks, airplanes, packages, "
        "offices, or airports there are, or what the gas price or labor cost is",
    )
    ASK_CURRENT_STATE = (
        "ask_current_state",
        "human asks where a package or vehicle is right now, or what is currently at a place or inside a vehicle",
    )
    ADD_ENTITY = (
        "add_entity",
        "human wants to add something new to the network, for example a new package, truck, or airplane",
    )
    UPDATE_ENTITY = (
        "update_entity",
        "human wants to change something that already exists, for example the gas price, a labor cost, "
        "or a package's weight",
    )
    ASK_CAPABILITIES = (
        "ask_capabilities",
        "human asks what the agent can do, or how it can help",
    )
    GREETING = (
        "greeting",
        "human says hello, thanks the agent, or makes small talk",
    )


class DeliveryRequest(BaseModel):
    """a delivery the human wants a price estimate for"""

    origin: City | None = Field(description="city the package is shipped from, if the human said it")
    destination: City | None = Field(description="city the package is delivered to, if the human said it")
    weight: float | None = Field(description="weight of the package, if the human said it")


class EntityKind(DocEnum):
    """kind of thing in the logistics network the human can ask about"""

    OFFICE = ("office", "offices where packages start and end their journeys")
    AIRPORT = ("airport", "airports where airplanes are")
    PACKAGE = ("package", "packages being shipped")
    VEHICLE = ("vehicle", "any vehicle, truck or airplane, when the human does not say which")
    TRUCK = ("truck", "trucks that travel by road")
    AIRPLANE = ("airplane", "airplanes that fly between airports")
    GAS_PRICE = ("gas_price", "what fuel costs for every mile a truck drives")
    LABOR_COST = ("labor_cost", "what crews in each city cost to handle packages")

    @property
    def entity_type(self) -> type[Entity]:
        """domain type of the entities of this kind"""
        return _ENTITY_TYPES[self]


_ENTITY_TYPES: dict[EntityKind, type[Entity]] = {
    EntityKind.OFFICE: Office,
    EntityKind.AIRPORT: Airport,
    EntityKind.PACKAGE: Package,
    EntityKind.VEHICLE: Vehicle,
    EntityKind.TRUCK: Truck,
    EntityKind.AIRPLANE: Airplane,
    EntityKind.GAS_PRICE: GasPrice,
    EntityKind.LABOR_COST: LaborCost,
}


class EntityQuery(BaseModel):
    """kinds of things the human is asking about"""

    kinds: list[EntityKind] = Field(description="every kind of thing the human asks about; empty if none fits")


class StateQuery(BaseModel):
    """what the human wants to know about where things currently are; leave a side empty when it is what they
    are asking for, e.g. "where is X" names only entity1, "what is at Y" names only entity2"""

    entity1: str | None = Field(
        description="name of the package or vehicle asked about, exactly as listed in the known entities, if any"
    )
    entity2: str | None = Field(
        description="name of the office, airport, or vehicle asked about, exactly as listed in the known entities, "
        "if any"
    )


class NewEntity(BaseModel):
    """a new thing the human wants to add to the logistics network"""

    kind: EntityKind | None = Field(description="kind of thing being added, if the human said")
    location: str | None = Field(
        description="name of the office, airport, or vehicle the new thing is at, exactly as listed in the known "
        "entities, if the human said"
    )


class EntityChange(BaseModel):
    """an existing thing the human wants to change"""

    entity: str | None = Field(
        description="name of the thing being changed, exactly as listed in the known entities, if the human said"
    )


class AtFact(BaseModel):
    """one fact the human wants to hold once the delivery is done: a package or vehicle is at a place"""

    entity1: str = Field(description="name of the package or vehicle, exactly as listed in the known entities")
    entity2: str = Field(
        description="name of the office, airport, or vehicle it should be at, exactly as listed in the known entities"
    )


class DesiredOutcome(BaseModel):
    """outcome the human wants the delivery to achieve, as facts about where things should end up"""

    facts: list[AtFact] = Field(
        description="where each package or vehicle should end up; empty if the human did not say"
    )

    def to_relations(self, find_entity: Callable[[str], Entity | None]) -> list[At]:
        """resolves the named facts into At relations, looking each name up with find_entity (such as the reasoner's,
        which also accepts display names); raises KeyError for a name it cannot find and pydantic.ValidationError
        when an entity cannot be at the named place"""

        def resolve(name: str) -> Entity:
            entity = find_entity(name)
            if entity is None:
                raise KeyError(name)
            return entity

        return [At(entity1=resolve(fact.entity1), entity2=resolve(fact.entity2)) for fact in self.facts]


def known_entities(world: WorldGraph) -> list[str]:
    """describes the packages, vehicles, offices, airports, and costs in the world, so the language model can refer
    to them by name when it fills in a desired outcome, a query, or a change"""
    snapshot = world.snapshot
    lines = [f"package {p.name}: {p.display_name}" for p in snapshot.by(Package)]
    lines += [f"truck {t.name}: {t.display_name}" for t in snapshot.by(Truck)]
    lines += [f"airplane {a.name}: {a.display_name}" for a in snapshot.by(Airplane)]
    lines += [f"office {o.name}: {o.display_name} in {o.city}" for o in snapshot.by(Office)]
    lines += [f"airport {a.name}: {a.display_name} in {a.city}" for a in snapshot.by(Airport)]
    lines += [f"gas price {g.name}" for g in snapshot.by(GasPrice)]
    lines += [f"labor cost {c.name}: labor cost in {c.city}" for c in snapshot.by(LaborCost)]
    return lines
