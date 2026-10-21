"""offline tests of the communicator's language models: what they turn into, and what they tell the LLM"""

from pathlib import Path

import pytest
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel

from logistics_agent.communicator import (
    LLM_CONFIG,
    HumanIntent,
    asked_context,
    known_entities,
    load_llm,
)
from logistics_agent.domain import Shipment
from logistics_agent.reasoner import Reasoner


def test_every_intent_has_a_description():
    assert all(intent.__doc__ for intent in HumanIntent)


def test_known_entities_names_everything_the_human_can_refer_to():
    lines = known_entities(Reasoner().world)
    for expected in (
        "shipment shipment1: crate of machine parts",
        "city boston: Boston",
        "truck truck1: Big Bertha",
        "airplane air1: The Pelican",
        "terminal terminal_boston: Boston Terminal in boston",
        "airport SEA: Seattle-Tacoma International Airport in seattle",
        "gas price gas_price",
        "labor cost labor_cost_denver: labor cost in denver",
    ):
        assert expected in lines
    assert len(lines) == 5 + 1 + 2 + 1 + 5 + 5 + 1 + 5


def test_known_entities_follows_changes_to_the_world():
    reasoner = Reasoner()
    reasoner.add_entity(Shipment(name="shipment2", details="box of books", weight=200), "terminal_boston")
    assert "shipment shipment2: box of books" in known_entities(reasoner.world)


def test_asked_context_adds_the_question_only_when_there_is_one():
    assert asked_context(None) == []
    assert asked_context("Where is the shipment now?") == ['the agent had just asked the human: "Where is the shipment now?"']


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
