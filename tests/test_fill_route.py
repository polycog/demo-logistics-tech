"""offline tests of the RouteFiller sub-process, which fills in which route the human reports blocked

a fake EnumClassifier and a fake ModelPopulator stand in for the LLM, answering with the kind and the cities they are
given, or failing, so the sub-process runs deterministically with no API key
"""

import pytest
from pydantic_ai.exceptions import UnexpectedModelBehavior

from logistics_agent import fill_route
from logistics_agent.communicator import AskFor, AskForKind, Utterance, asked_context, in_context
from logistics_agent.domain import AirRoute, RouteReport, RouteType, TruckRoute
from logistics_agent.fill_route import ChooseRouteType, RouteFiller, cities_model
from logistics_agent.reasoner import Reasoner

PHRASE = "the route between Seattle and San Francisco is snowed in"
ASK_KIND = AskForKind(RouteType)
ASK_CITY1, ASK_CITY2 = AskFor(RouteReport, "city1"), AskFor(RouteReport, "city2")


class FakeEnumClassifier:
    """stands in for EnumClassifier: answers with the kind in `kind`; records how it was built, and every text it
    classified"""

    kind: RouteType | None = RouteType.ROAD
    built: list[tuple] = []
    calls: list[str] = []

    def __init__(self, enum_type, task_desc: str) -> None:
        FakeEnumClassifier.built.append((enum_type, task_desc))

    def __call__(self, text: str, llm, **kwargs):
        FakeEnumClassifier.calls.append(text)
        return FakeEnumClassifier.kind, 1.0


class FakeModelPopulator:
    """stands in for ModelPopulator: answers with the cities in `given` that the model it was built for has, or raises
    the error in `given`; records the models it was built for, and every call"""

    given: dict | Exception = {}
    built: list[type] = []
    calls: list[tuple] = []

    def __init__(self, model_type, task_desc: str) -> None:
        self.model_type = model_type
        FakeModelPopulator.built.append(model_type)

    def __call__(self, utterance: str, llm, *extra: str, **kwargs):
        FakeModelPopulator.calls.append((utterance, extra))
        if isinstance(FakeModelPopulator.given, Exception):
            raise FakeModelPopulator.given
        fields = self.model_type.model_fields
        return self.model_type.model_validate({k: v for k, v in FakeModelPopulator.given.items() if k in fields})


@pytest.fixture(autouse=True)
def fake_language(monkeypatch):
    """replaces the LLM-backed EnumClassifier and ModelPopulator; returns the fake populator"""
    FakeEnumClassifier.kind, FakeEnumClassifier.built, FakeEnumClassifier.calls = RouteType.ROAD, [], []
    FakeModelPopulator.given = {"city1": "Seattle", "city2": "San Francisco"}
    FakeModelPopulator.built, FakeModelPopulator.calls = [], []
    monkeypatch.setattr(fill_route, "EnumClassifier", FakeEnumClassifier)
    monkeypatch.setattr(fill_route, "ModelPopulator", FakeModelPopulator)
    return FakeModelPopulator


def fill(report: RouteReport | None = None, kind: RouteType | None = None, answering=None):
    """runs a fresh RouteFiller over the phrase, as the conductor does"""
    return RouteFiller()(
        utterance=Utterance(PHRASE), answering=answering, reasoner=Reasoner(), report=report or RouteReport(),
        kind=kind, suppress_errors=False,
    )


def cities(report: RouteReport) -> tuple:
    return tuple(None if city is None else city.name for city in (report.city1, report.city2))


def seattle() -> object:
    return Reasoner().find_entity("seattle")


# --- the route types


@pytest.mark.parametrize("kind, model", [(RouteType.ROAD, TruckRoute), (RouteType.AIR, AirRoute)])
def test_each_route_type_knows_its_model(kind, model):
    assert kind.model is model


def test_the_kind_question_is_the_route_types_docstring():
    assert ASK_KIND.question == RouteType.__doc__ and ASK_KIND.question.endswith("?")
    assert all(kind.__doc__ for kind in RouteType)


# --- the kind


def test_the_kind_is_classified_among_the_route_types():
    ChooseRouteType("op")
    assert FakeEnumClassifier.built == [(RouteType, ChooseRouteType.TASK_DESC)]


def test_the_route_is_filled_in_from_one_utterance_and_nothing_more_is_asked():
    result = fill()
    assert cities(result.report) == ("seattle", "san_francisco") and result.kind is RouteType.ROAD
    assert result.asks == [] and FakeEnumClassifier.calls == [PHRASE]


def test_the_kind_is_asked_for_last_when_the_cause_does_not_tell():
    FakeEnumClassifier.kind = None
    result = fill()
    assert result.kind is None and result.asks == [ASK_KIND]


def test_a_known_kind_is_not_classified_again():
    assert fill(kind=RouteType.AIR).kind is RouteType.AIR and FakeEnumClassifier.calls == []


def test_the_kind_is_classified_as_the_answer_to_the_question_asked():
    fill(answering=ASK_KIND)
    assert FakeEnumClassifier.calls == [in_context(ASK_KIND.question, PHRASE)]


# --- the cities


def test_the_cities_model_has_only_the_ends_to_fill_and_names_every_city():
    model = cities_model(["city2"], Reasoner().find_entities(fill_route.City))
    assert set(model.model_fields) == {"city2"}
    description = model.model_fields["city2"].description
    assert RouteReport.model_fields["city2"].description in description and "Denver" in description


def test_only_the_cities_missing_from_the_report_are_filled_in(fake_language):
    result = fill(RouteReport(city1=seattle()))
    [model] = fake_language.built
    assert set(model.model_fields) == {"city2"} and cities(result.report) == ("seattle", "san_francisco")


def test_the_cities_are_read_as_the_answer_to_the_question_asked(fake_language):
    fill(answering=ASK_CITY2)
    assert fake_language.calls[0][1] == tuple(asked_context(ASK_CITY2.question))


@pytest.mark.parametrize("given, asks", [
    ({"city1": "Seattle", "city2": "Portland"}, [ASK_CITY2]),
    ({"city1": "Seattle"}, [ASK_CITY2]),
    ({}, [ASK_CITY1, ASK_CITY2]),
    (UnexpectedModelBehavior("no answer"), [ASK_CITY1, ASK_CITY2]),
])
def test_the_cities_still_missing_are_asked_for_in_order(fake_language, given, asks):
    fake_language.given = given
    assert fill().asks == asks


def test_missing_cities_are_asked_for_before_the_kind():
    FakeEnumClassifier.kind = None
    FakeModelPopulator.given = {}
    assert fill().asks == [ASK_CITY1, ASK_CITY2, ASK_KIND]


def test_the_report_given_is_left_as_it_was():
    report = RouteReport(city1=seattle())
    fill(report)
    assert report.city2 is None


def test_a_complete_report_does_not_call_the_language_model_for_the_cities(fake_language):
    find = Reasoner().find_entity
    result = fill(RouteReport(city1=find("seattle"), city2=find("denver")), kind=RouteType.AIR)
    assert result.asks == [] and fake_language.calls == []
