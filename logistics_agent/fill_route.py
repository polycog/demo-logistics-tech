"""RouteFiller: the decision sub-process that fills in which route the human reports blocked, such as "the route between
Seattle and San Francisco is snowed in", and asks for what the utterance does not give

the conductor's ReactToObstructRouteIntent operator creates it once, then for every utterance the human makes about the
blocked route, restarts it and calls it with five arguments, which its operators read from io.a:
    utterance: the utterance the human made
    answering: the ask the utterance answers, or None; the utterance is read together with its question
    reasoner: the reasoner, whose world graph holds the cities
    report: the RouteReport filled in so far: the cities at the two ends of the route
    kind: the RouteType of the route, a road or the flights, or None while it is not known
the kind is classified from the cause, such as snow for a road or grounded flights for the air; the cities missing from
the report are extracted together, in one call to the language model, so each city the human names fills in one end
only; the sub-process finishes with the report, the kind, and the asks for what is still missing: the cities in order,
then the kind, which is empty once the route is known
"""

from dataclasses import dataclass, field
from typing import Any

from cognition import DecisionProcess, EnumClassifier, IOContainer, ModelPopulator, Operator, uniform_evaluator
from pydantic import BaseModel, Field, ValidationError, create_model
from pydantic_ai.exceptions import UnexpectedModelBehavior

from logistics_agent.communicator import DEFAULT_LLM, AskFor, AskForKind, Utterance, asked_context, in_context
from logistics_agent.domain import City, RouteReport, RouteType
from logistics_agent.fill_shipping_request import city_named


@dataclass
class RouteState:
    """state of the sub-process, kept for one utterance"""

    # the report being filled in, taken from the conductor; None until it has been taken
    report: RouteReport | None = None
    # the kind of route; None while it is not known
    kind: RouteType | None = None
    # the cities missing from the report, which are extracted from the utterance
    to_fill: list[str] = field(default_factory=list)
    # whether the kind has been classified, and the cities extracted, from the utterance
    kind_tried: bool = False
    extracted: bool = False
    # what is still missing once the utterance has been used, as asks in the order to ask them; None until then
    asks: list[AskFor | AskForKind] | None = None


class TakeReport(Operator[RouteState]):
    """
    WHEN: the sub-process has not taken the report yet
    THEN: take a copy of the report filled in so far, and the kind of route, from the conductor
    """

    def can_perform(self, state: RouteState, io: IOContainer) -> bool:
        """applies once, at the start"""
        return state.report is None

    def perform(self, state: RouteState, io: IOContainer) -> None:
        """copies the report, so the conductor's stays as it was until it takes the result"""
        report: RouteReport = io.a.report
        state.report = report.model_copy()
        state.kind = io.a.kind
        state.to_fill = [name for name in RouteReport.model_fields if getattr(report, name) is None]


def question(io: IOContainer) -> str | None:
    """the question of the ask the utterance answers, or None"""
    answering: AskFor | AskForKind | None = io.a.answering
    return None if answering is None else answering.question


class ChooseRouteType(Operator[RouteState]):
    """
    WHEN: the report has been taken, the kind of route is not known, and has not been classified from the utterance
    THEN: classify the kind of route in the utterance among the RouteTypes, read as the answer to the question asked,
          if any; it stays unknown when the utterance does not tell a road from the flights
    """

    TASK_DESC = (
        "a human tells a logistics carrier that a route between two cities is blocked; which kind of route is it, "
        "judging by what blocks it"
    )

    def __init__(self, name: str, **kwargs: Any) -> None:
        """sets up the kind classifier once, so its prompt is reused for every utterance"""
        super().__init__(name, **kwargs)
        self._kind_classifier = EnumClassifier(RouteType, task_desc=self.TASK_DESC)

    def can_perform(self, state: RouteState, io: IOContainer) -> bool:
        """applies once per utterance, while the kind is not known"""
        return state.report is not None and state.kind is None and not state.kind_tried

    def perform(self, state: RouteState, io: IOContainer) -> None:
        """classifies the kind"""
        utterance: Utterance = io.a.utterance
        assert utterance.phrase is not None, "expect the conductor to supply what the human said"
        state.kind_tried = True
        state.kind, _confidence = self._kind_classifier(
            in_context(question(io), utterance.phrase), DEFAULT_LLM, num_trials=1
        )


def cities_model(fields: list[str], cities: list[City]) -> type[BaseModel]:
    """one model of the ends of the route to fill in, each the name of a city, None when the utterance does not give
    it; a plain name, checked against the cities afterwards, so a city the carrier does not serve leaves only that end
    missing"""
    served = ", ".join(city.display_name for city in cities)
    return create_model(
        "RouteEnds",
        __doc__="the cities at the ends of the blocked route; each city the human names fills in one end only",
        **{
            name: (str | None, Field(default=None, description=(
                f"{RouteReport.model_fields[name].description} The name of a city, one of {served}; None when the "
                "human does not say"
            )))
            for name in fields
        },
    )


class ExtractCities(Operator[RouteState]):
    """
    WHEN: the report has cities missing, and they have not been extracted from the utterance yet
    THEN: fill in the missing cities together from the utterance, read as the answer to the question asked, if any;
          a city stays missing when the utterance does not give it, or gives one the carrier does not serve
    """

    TASK_DESC = (
        "a human tells a logistics carrier that a route between two cities is blocked; fill in the cities at the ends "
        "of the route the human gives, and leave an end None when the human does not give it"
    )

    def can_perform(self, state: RouteState, io: IOContainer) -> bool:
        """applies once per utterance, while cities are missing"""
        return state.report is not None and bool(state.to_fill) and not state.extracted

    def perform(self, state: RouteState, io: IOContainer) -> None:
        """populates the cities model in one call, then keeps each city the world graph knows"""
        utterance: Utterance = io.a.utterance
        assert utterance.phrase is not None and state.report is not None
        state.extracted = True
        cities = io.a.reasoner.find_entities(City)
        populator = ModelPopulator(cities_model(state.to_fill, cities), task_desc=self.TASK_DESC)
        try:
            ends = populator(utterance.phrase, DEFAULT_LLM, *asked_context(question(io)))
        except (UnexpectedModelBehavior, ValidationError):
            return
        for name in state.to_fill:
            setattr(state.report, name, city_named(getattr(ends, name), cities))


class ComposeAsks(Operator[RouteState]):
    """
    WHEN: the utterance has been used: the kind classified unless it was known, and the missing cities extracted unless
          none were
    THEN: ask for each city still missing, in order, then for the kind if it is still not known; ask for nothing once
          the route is known
    """

    def can_perform(self, state: RouteState, io: IOContainer) -> bool:
        """applies once nothing is left to extract, and only once"""
        return (
            state.report is not None
            and state.asks is None
            and (state.kind is not None or state.kind_tried)
            and (state.extracted or not state.to_fill)
        )

    def perform(self, state: RouteState, io: IOContainer) -> None:
        """records the asks"""
        assert state.report is not None
        state.asks = [AskFor(RouteReport, name) for name in RouteReport.model_fields if getattr(state.report, name) is None]
        if state.kind is None:
            state.asks.append(AskForKind(RouteType))


def asks_composed(state: RouteState, _io: IOContainer) -> bool:
    """the sub-process is done once it has composed the asks, which are empty once the route is known"""
    return state.asks is not None


class RouteFiller(DecisionProcess[RouteState]):
    """decision sub-process that fills in which route the human reports blocked and asks for what is missing; see the
    module docstring for how it is called"""

    def __init__(self) -> None:
        """registers the termination check, and an evaluator that ranks every action equally; starts over a fresh
        RouteState"""
        super().__init__(RouteState)
        self.add_operator(TakeReport("take_report"))
        self.add_operator(ChooseRouteType("choose_route_type"))
        self.add_operator(ExtractCities("extract_cities"))
        self.add_operator(ComposeAsks("compose_asks"))
        self.add_termination_check(asks_composed)
        self.add_action_evaluator(uniform_evaluator(0))
