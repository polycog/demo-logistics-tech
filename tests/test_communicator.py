"""offline tests of the communicator's language models: what they turn into, and what they tell the LLM"""

from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel

from logistics_agent.communicator import LLM_CONFIG, AtFact, DesiredOutcome, EntityKind, HumanIntent, known_entities, load_llm
from logistics_agent.domain import Airplane, Airport, At, GasPrice, LaborCost, Office, Package, Truck, Vehicle
from logistics_agent.reasoner import Reasoner


def test_every_intent_has_a_description():
    assert all(intent.__doc__ for intent in HumanIntent)


@pytest.mark.parametrize("kind, entity_type", [
    (EntityKind.OFFICE, Office),
    (EntityKind.AIRPORT, Airport),
    (EntityKind.PACKAGE, Package),
    (EntityKind.VEHICLE, Vehicle),
    (EntityKind.TRUCK, Truck),
    (EntityKind.AIRPLANE, Airplane),
    (EntityKind.GAS_PRICE, GasPrice),
    (EntityKind.LABOR_COST, LaborCost),
])
def test_every_entity_kind_maps_to_its_domain_type(kind, entity_type):
    assert kind.entity_type is entity_type and kind.__doc__


def test_desired_outcome_resolves_to_at_relations():
    outcome = DesiredOutcome(facts=[
        AtFact(entity1="package1", entity2="office_denver"),
        AtFact(entity1="air1", entity2="SFO"),
    ])
    relations = outcome.to_relations(Reasoner().find_entity)
    assert all(isinstance(r, At) for r in relations)
    assert [(r.entity1.name, r.entity2.name) for r in relations] == [("package1", "office_denver"), ("air1", "SFO")]


def test_desired_outcome_resolves_display_names():
    [relation] = DesiredOutcome(facts=[AtFact(entity1="crate of machine parts", entity2="Denver Office")]).to_relations(
        Reasoner().find_entity
    )
    assert (relation.entity1.name, relation.entity2.name) == ("package1", "office_denver")


def test_empty_desired_outcome_resolves_to_no_relations():
    assert DesiredOutcome(facts=[]).to_relations(Reasoner().find_entity) == []


def test_desired_outcome_with_an_unknown_name_raises_key_error():
    with pytest.raises(KeyError):
        DesiredOutcome(facts=[AtFact(entity1="package1", entity2="narnia")]).to_relations(Reasoner().find_entity)


def test_desired_outcome_that_cannot_hold_raises_validation_error():
    with pytest.raises(ValidationError):
        DesiredOutcome(facts=[AtFact(entity1="office_denver", entity2="truck1")]).to_relations(Reasoner().find_entity)


def test_known_entities_names_everything_the_human_can_refer_to():
    lines = known_entities(Reasoner().world)
    for expected in (
        "package package1: crate of machine parts",
        "truck truck1: Big Bertha",
        "airplane air1: The Pelican",
        "office office_boston: Boston Office in boston",
        "airport SEA: Seattle-Tacoma International Airport in seattle",
        "gas price gas_price",
        "labor cost labor_cost_denver: labor cost in denver",
    ):
        assert expected in lines
    assert len(lines) == 1 + 2 + 1 + 5 + 5 + 1 + 5


def test_known_entities_follows_changes_to_the_world():
    reasoner = Reasoner()
    reasoner.add_entity(Package(name="package2", display_name="box of books", weight=200), "office_boston")
    assert "package package2: box of books" in known_entities(reasoner.world)


# === language model configuration


def write_config(tmp_path: Path, provider: str) -> Path:
    path = tmp_path / "llm_config.toml"
    path.write_text(f"""provider = "{provider}"

[openai]
model = "gpt-4o-mini"

[local]
model = "some/local-model"
base_url = "http://localhost:9999/v1"
api_key = "anything"
""")
    return path


@pytest.fixture(autouse=True)
def no_provider_override(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")


def test_openai_provider_uses_the_openai_model(tmp_path):
    llm = load_llm(write_config(tmp_path, "openai"))
    assert isinstance(llm, OpenAIResponsesModel) and llm.model_name == "gpt-4o-mini"


def test_local_provider_uses_the_local_model_at_its_base_url(tmp_path):
    llm = load_llm(write_config(tmp_path, "local"))
    assert isinstance(llm, OpenAIChatModel) and llm.model_name == "some/local-model"
    assert llm.base_url == "http://localhost:9999/v1/"


def test_environment_overrides_the_configured_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert isinstance(load_llm(write_config(tmp_path, "local")), OpenAIResponsesModel)


def test_unknown_provider_is_refused(tmp_path):
    with pytest.raises(ValueError, match="'gemini'"):
        load_llm(write_config(tmp_path, "gemini"))


@pytest.mark.parametrize("provider", ["openai", "local"])
def test_shipped_config_works_for_both_providers(provider, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", provider)
    assert load_llm(LLM_CONFIG).model_name
