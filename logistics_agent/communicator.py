"""language understanding and generation for the logistics support agent, built on cognition's language tools"""

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from cognition import DocEnum, Entity, WorldGraph
from pydantic import BaseModel, Field
from pydantic_ai.models import Model, infer_model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from logistics_agent.domain import Airplane, Airport, City, GasPrice, LaborCost, Terminal, Shipment, Truck, VehicleType

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

    ASK_CAPABILITIES = (
        "ask_capabilities",
        "human asks what the agent can do, or how it can help",
    )
    ESTIMATE_PRICE = (
        "estimate_price",
        "human asks what a delivery costs, or for a price estimate or quote",
    )
    SHIPPING_REQUEST = (
        "shipping_request",
        "human describes a delivery they want: where the shipment is picked up, where it is dropped off, "
        "or what the shipment is and how much it weighs",
    )
    GREETING = (
        "greeting",
        "human says hello, thanks the agent, or makes small talk",
    )
    ADD_TO_FLEET = (
        "add_to_fleet",
        "human wants to add a vehicle, such as a truck or an airplane, to the carrier's fleet, or describes the "
        "vehicle being added: what kind it is, its name, make and model, payload, registration, or where it is",
    )
    CONFIRM = (
        "confirm",
        "human says yes, agrees, or tells the agent to go ahead",
    )
    DENY = (
        "deny",
        "human says no, declines, or tells the agent not to go ahead",
    )


@dataclass(frozen=True)
class AgentIntent:
    """what the logistics support agent wants from the conversation, as opposed to what the human wants; frozen, so two
    intents with the same parts are equal and the stack holds each once"""


@dataclass(frozen=True)
class NotUnderstood(AgentIntent):
    """the agent understood the human's last utterance as none of the intents it knows how to handle, and has to say
    so"""


@dataclass(frozen=True)
class ConfirmAddition(AgentIntent):
    """the agent has the vehicle to add to the fleet and where it is, and asks the human to confirm before it adds
    it; the human's yes or no answers it"""


@dataclass(frozen=True)
class AskForKind(AgentIntent):
    """the agent asks which kind of vehicle the human wants to add to the fleet; the question is VehicleType's own
    docstring, and the ask is satisfied once the agent has a vehicle of a kind to fill in"""

    @property
    def question(self) -> str:
        """what the agent asks the human"""
        assert VehicleType.__doc__ is not None
        return VehicleType.__doc__.strip()


@dataclass(frozen=True)
class AskFor(AgentIntent):
    """the agent needs a model it is filling in, or one attribute of it, and asks the human for it

    the question is the model's own: its docstring for the whole model, the attribute's Field description for one
    attribute
    """

    # the model being filled in, such as ShippingRequest
    model: type[BaseModel]
    # the attribute asked for, such as "destination"; None for the whole model
    attribute: str | None = None

    def __post_init__(self) -> None:
        """the attribute must be one of the model's"""
        if self.attribute is not None and self.attribute not in self.model.model_fields:
            raise ValueError(f"{self.model.__name__} has no attribute {self.attribute!r}")

    @property
    def question(self) -> str:
        """what the agent asks the human"""
        if self.attribute is None:
            assert self.model.__doc__ is not None, f"{self.model.__name__} needs a docstring to ask for it"
            return self.model.__doc__.strip()
        description = self.model.model_fields[self.attribute].description
        assert description is not None, f"{self.model.__name__}.{self.attribute} needs a description to ask for it"
        return description

    def satisfied(self, instance: BaseModel | None) -> bool:
        """whether the instance being filled in has what is asked for: the attribute, or every attribute of the whole
        model; never while there is no instance"""
        if instance is None:
            return False
        attributes = self.model.model_fields if self.attribute is None else [self.attribute]
        return all(getattr(instance, attribute) is not None for attribute in attributes)


def label(entity: Entity) -> str:
    """how to refer to an entity in a reply: its display name, or its name with spaces"""
    return getattr(entity, "display_name", entity.name.replace("_", " "))



def known_entities(world: WorldGraph) -> list[str]:
    """describes the cities, shipments, vehicles, terminals, airports, and costs in the world, so the language model
    can refer to them by name when it fills in a model"""
    snapshot = world.snapshot
    lines = [f"city {c.name}: {c.display_name}" for c in snapshot.by(City)]
    lines += [f"shipment {p.name}: {p.details}" for p in snapshot.by(Shipment)]
    lines += [f"truck {t.name}: {t.display_name}" for t in snapshot.by(Truck)]
    lines += [f"airplane {a.name}: {a.display_name}" for a in snapshot.by(Airplane)]
    lines += [f"terminal {o.name}: {o.display_name} in {o.city.name}" for o in snapshot.by(Terminal)]
    lines += [f"airport {a.name}: {a.display_name} in {a.city.name}" for a in snapshot.by(Airport)]
    lines += [f"gas price {g.name}" for g in snapshot.by(GasPrice)]
    lines += [f"labor cost {c.name}: labor cost in {c.city.name}" for c in snapshot.by(LaborCost)]
    return lines


def in_context(asked: str | None, phrase: str) -> str:
    """the phrase as a classifier should see it: after the agent asked a question, a short answer such as "denver" is
    classified together with that question"""
    if asked is None:
        return phrase
    return f'the agent asked: "{asked}" and the human answered: "{phrase}"'


def asked_context(asked: str | None) -> list[str]:
    """extra information for the language model when the utterance answers a question the agent had just asked, so
    that a short answer such as "denver" is understood"""
    return [f'the agent had just asked the human: "{asked}"'] if asked else []
