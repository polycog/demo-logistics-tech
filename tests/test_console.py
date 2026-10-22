"""offline tests of how the conversation is shown in the terminal, and of tracing the agent's reasoning"""

import io
import re

from enum import Enum

from cognition import DecisionProcess, EnumClassifier, IOContainer, ModelPopulator, Operator, SearchPlanner
from pydantic import BaseModel

from logistics_agent import conductor

from logistics_agent.console import AGENT_PREFIX, HUMAN_PREFIX, THINKING, Console
from logistics_agent.trace import Step, trace

PLAN = "Delivering the chairs costs $802.00.\n\n1. Drive **Big Bertha** to the pickup.\n2. Load the chairs."


class FakeTerminal(io.StringIO):
    """captures what is shown, as a terminal would, or as a pipe when it is not a terminal"""

    def __init__(self, terminal: bool) -> None:
        super().__init__()
        self.terminal = terminal

    def isatty(self) -> bool:
        return self.terminal


def styled(width: int = 60) -> tuple[Console, FakeTerminal]:
    out = FakeTerminal(terminal=True)
    return Console(out, styled=True, width=width), out


# === plain output, when not a terminal


def test_a_console_that_is_not_a_terminal_is_not_styled(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert not Console(FakeTerminal(terminal=False)).styled
    assert Console(FakeTerminal(terminal=True)).styled


def test_no_color_turns_styling_off(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    assert not Console(FakeTerminal(terminal=True)).styled


def test_plain_output_shows_each_reply_on_one_line_after_its_label():
    out = FakeTerminal(terminal=False)
    console = Console(out, width=20)
    console.show_reply(PLAN.replace("\n", " "))
    console.show_note("(type 'quit' to stop)")
    console.show_step(1, "fill_shipping_request", "extract_origin")
    assert out.getvalue() == (
        f"{AGENT_PREFIX}{PLAN.replace(chr(10), ' ')}\n(type 'quit' to stop)\n    · fill_shipping_request: extract_origin\n"
    )


def test_plain_input_is_read_after_the_humans_label(monkeypatch):
    prompts = []
    monkeypatch.setattr("builtins.input", lambda prompt="": prompts.append(prompt) or "hello")
    assert Console(FakeTerminal(terminal=False)).read_line() == "hello" and prompts == [HUMAN_PREFIX]


def test_there_is_no_spinner_when_not_styled():
    out = FakeTerminal(terminal=False)
    console = Console(out)
    console.start_thinking()
    console.stop_thinking()
    assert out.getvalue() == ""


# === styled output, with Rich


def test_a_reply_is_shown_after_the_agents_label_and_followed_by_a_thin_line():
    console, out = styled()
    console.show_reply("Hello!")
    lines = re.sub(r"\x1b\[[0-9;]*m", "", out.getvalue()).splitlines()
    assert lines[0].startswith(f"{AGENT_PREFIX}Hello!")
    assert lines[-1] == "─" * 60
    assert "╭" not in out.getvalue() and "│" not in out.getvalue(), "no box around the reply"
    assert "\x1b[" in out.getvalue()


def test_the_lines_of_a_long_reply_are_indented_under_the_label():
    console, out = styled(width=40)
    console.show_reply("word " * 30)
    lines = [line for line in re.sub(r"\x1b\[[0-9;]*m", "", out.getvalue()).splitlines() if "word" in line]
    assert len(lines) > 1 and lines[0].startswith(AGENT_PREFIX)
    assert all(line.startswith(" " * len(AGENT_PREFIX) + "word") for line in lines[1:])


def test_a_reply_is_rendered_as_markdown_so_a_plan_shows_as_a_list():
    console, out = styled()
    console.show_reply(PLAN)
    shown = re.sub(r"\x1b\[[0-9;]*m", "", out.getvalue())
    assert "**" not in shown, "bold text is drawn bold, not with its markup"
    assert "1 Drive Big Bertha to the pickup." in shown and "2 Load the chairs." in shown


def test_styled_input_is_read_after_the_humans_label(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt="": "hello")
    console, out = styled()
    assert console.read_line() == "hello" and HUMAN_PREFIX in out.getvalue()


def test_the_spinner_shows_while_thinking_and_is_taken_away_by_a_reply():
    console, out = styled()
    console.start_thinking()
    assert console._status is not None
    console.show_reply("Hello!")
    assert console._status is None and THINKING in out.getvalue()


def test_stopping_the_spinner_twice_is_harmless():
    console, _ = styled()
    console.start_thinking()
    console.stop_thinking()
    console.stop_thinking()
    assert console._status is None


def test_conductor_planner_and_language_model_steps_each_have_their_own_color():
    shown = {}
    for kind in ("decision", "planner", "llm"):
        console, out = styled()
        console.show_step(0, "source", "action", kind)
        shown[kind] = out.getvalue().split("·")[0]
    assert len(set(shown.values())) == 3
    assert shown["planner"].startswith("\x1b[33m"), "planner steps are yellow"


def test_a_step_keeps_the_spinner_and_is_shown_indented():
    console, out = styled()
    console.start_thinking()
    console.show_step(1, "fill_shipping_request", "extract_origin")
    assert console._status is not None
    console.stop_thinking()
    assert "    · fill_shipping_request: extract_origin" in out.getvalue()


def test_progress_is_shown_beside_the_spinner_and_a_step_puts_it_back_to_thinking():
    console, out = styled()
    console.start_thinking()
    console.show_progress("planning… 3 search steps")
    assert "planning… 3 search steps" in str(console._status.renderable.text)
    console.show_step(0, "conductor", "describe_plan")
    assert THINKING in str(console._status.renderable.text)
    console.stop_thinking()


def test_progress_shows_nothing_without_a_spinner():
    out = FakeTerminal(terminal=False)
    console = Console(out)
    console.show_progress("planning… 3 search steps")
    assert out.getvalue() == ""


# === tracing


class Note(Operator[list]):
    """notes its name, once"""

    def can_perform(self, state: list, io: IOContainer) -> bool:
        return self.name not in state

    def perform(self, state: list, io: IOContainer) -> None:
        state.append(self.name)


class RunInner(Operator[list]):
    """runs an inner decision process, once"""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.inner = DecisionProcess[list](list)
        self.inner.add_operator(Note("inner_step"))
        self.inner.add_termination_check(lambda state, _io: "inner_step" in state)

    def can_perform(self, state: list, io: IOContainer) -> bool:
        return self.name not in state

    def perform(self, state: list, io: IOContainer) -> None:
        self.inner()
        state.append(self.name)


def test_tracing_reports_each_operator_applied_with_sub_processes_nested():
    steps = []
    stop = trace(steps.append)
    try:
        outer = DecisionProcess[list](list)
        outer.add_operator(RunInner("run_inner"))
        outer.add_termination_check(lambda state, _io: "run_inner" in state)
        outer()
    finally:
        stop()
    assert steps == [Step(0, "decision", "DecisionProcess", "run_inner"), Step(1, "decision", "DecisionProcess", "inner_step")]


def test_decision_processes_created_after_tracing_stops_are_not_traced():
    steps = []
    trace(steps.append)()
    outer = DecisionProcess[list](list)
    outer.add_operator(Note("step"))
    outer.add_termination_check(lambda state, _io: "step" in state)
    outer()
    assert steps == []


class Shade(Enum):
    LIGHT = "light"
    DARK = "dark"


class Paint(BaseModel):
    color: str


def test_tracing_reports_each_use_of_a_language_model(monkeypatch):
    monkeypatch.setattr(EnumClassifier, "__call__", lambda self, *args, **kwargs: (Shade.LIGHT, 1.0))
    monkeypatch.setattr(ModelPopulator, "__call__", lambda self, *args, **kwargs: Paint(color="red"))
    monkeypatch.setattr(conductor, "describe_facts", lambda *args, **kwargs: "a red paint")
    steps = []
    stop = trace(steps.append)
    try:
        assert EnumClassifier(Shade, task_desc=None)("a pale wall", None) == (Shade.LIGHT, 1.0)
        assert ModelPopulator(Paint, task_desc=None)("red paint", None) == Paint(color="red")
        assert conductor.describe_facts([Paint(color="red")], None, None) == "a red paint"
    finally:
        stop()
    assert steps == [
        Step(0, "llm", "llm", "classify Shade"),
        Step(0, "llm", "llm", "populate Paint"),
        Step(0, "llm", "llm", "describe facts"),
    ]


class FinishedSearch:
    """stands in for a search planner that has explored seven states"""

    states_explored = 7


def test_tracing_reports_each_search_the_planner_makes_once_it_is_over(monkeypatch):
    calls = []
    monkeypatch.setattr(SearchPlanner, "run", lambda self, *args, **kwargs: calls.append(self) or self)
    steps = []
    stop = trace(steps.append)
    try:
        search = FinishedSearch()
        assert SearchPlanner.run(search) is search and calls == [search]
    finally:
        stop()
    assert steps == [Step(0, "planner", "planner", "search for the cheapest plan: 0 search steps, 7 states explored")]


def test_tracing_counts_the_steps_of_a_real_search():
    from logistics_agent.domain import At, City, Location, Shipment
    from logistics_agent.planner import LogisticsPlanner
    from logistics_agent.reasoner import Reasoner

    steps = []
    stop = trace(steps.append)
    try:
        reasoner = Reasoner()
        origin = Location(name="shipping-origin", city=reasoner.find_entity("seattle"))
        destination = Location(name="shipping-destination", city=reasoner.find_entity("san_francisco"))
        shipment = Shipment(name="shipment", details="pallet of office chairs", weight=800)
        planner = LogisticsPlanner(reasoner.world)
        planner.setup(origin, destination, shipment)
        planner.set_current([At(entity1=shipment, entity2=origin)])
        planner.set_goal([At(entity1=shipment, entity2=destination)])
        planner.plan()
        planner.clear()
    finally:
        stop()
    search_steps = [step for step in steps if step.source == "BaseDecisionProcess"]
    [report] = [step for step in steps if step.kind == "planner"]
    assert report.action.startswith(f"search for the cheapest plan: {len(search_steps)} search steps, ")
    assert report.action.endswith(" states explored") and len(search_steps) > 0


def test_stopping_tracing_restores_what_it_wrapped():
    originals = (EnumClassifier.__call__, ModelPopulator.__call__, SearchPlanner.run, conductor.describe_facts)
    trace(lambda step: None)()
    assert (EnumClassifier.__call__, ModelPopulator.__call__, SearchPlanner.run, conductor.describe_facts) == originals
