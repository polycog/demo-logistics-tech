"""shows the cognition at work, step by step:
    decision: which operator each decision process applies, as it applies it, including the operators of decision
        sub-processes, which run inside an operator of their parent, and the steps of the planner's search
    planner: each search the planner makes for a plan, once it is over, with how many search steps it took and how
        many states it explored
    llm: when a language model is used, to classify an intent, to fill in a model, or to describe facts

a decision process binds its apply phase when it is created, so tracing must be turned on before the decision
processes to be traced are created
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from cognition import EnumClassifier, ModelPopulator, SearchPlanner
from cognition.decision.core import BaseDecisionProcess

from logistics_agent import conductor


@dataclass(frozen=True)
class Step:
    """one step of the agent's reasoning"""

    # how deeply the step is nested: 0 for the outermost decision process, one more inside each operator it applies
    depth: int
    # what kind of step it is: "decision", "planner", or "llm"
    kind: str
    # what took the step: the decision process's type name, "planner", or "llm"
    source: str
    # what the step did: the operator applied, or what the planner or language model was used for
    action: str


def trace(report: Callable[[Step], None]) -> Callable[[], None]:
    """from now on, reports every step: each operator applied by a decision process created from now on, and each use
    of a language model, before it happens, and each search the planner makes, once it is over; returns a function
    that stops tracing"""
    depth = 0
    # the steps the planner's search decision process has taken in the search under way
    search_steps = 0
    original_apply = BaseDecisionProcess._apply
    original_classify = EnumClassifier.__call__
    original_populate = ModelPopulator.__call__
    original_run = SearchPlanner.run
    original_describe_facts = conductor.describe_facts

    def traced_apply(self: BaseDecisionProcess) -> bool:
        """reports the chosen operator, then applies it one level deeper, so what it does reports nested"""
        nonlocal depth, search_steps
        chosen = self._chosen
        if type(self) is BaseDecisionProcess:
            search_steps += 1
        if chosen is not None:
            report(Step(depth, "decision", type(self).__name__, getattr(chosen, "name", str(chosen))))
        depth += 1
        try:
            return original_apply(self)
        finally:
            depth -= 1

    def traced_classify(self: EnumClassifier, *args: Any, **kwargs: Any) -> Any:
        report(Step(depth, "llm", "llm", f"classify {self._schema.__name__}"))
        return original_classify(self, *args, **kwargs)

    def traced_populate(self: ModelPopulator, *args: Any, **kwargs: Any) -> Any:
        report(Step(depth, "llm", "llm", f"populate {self._schema.__name__}"))
        return original_populate(self, *args, **kwargs)

    def traced_run(self: SearchPlanner, *args: Any, **kwargs: Any) -> Any:
        """runs the search, then reports it with how many search steps it took and how many states it explored"""
        nonlocal search_steps
        search_steps = 0
        result = original_run(self, *args, **kwargs)
        report(Step(depth, "planner", "planner", (
            f"search for the cheapest plan: {search_steps} search steps, {self.states_explored} states explored"
        )))
        return result

    def traced_describe_facts(*args: Any, **kwargs: Any) -> Any:
        report(Step(depth, "llm", "llm", "describe facts"))
        return original_describe_facts(*args, **kwargs)

    BaseDecisionProcess._apply = traced_apply  # type: ignore[method-assign]
    EnumClassifier.__call__ = traced_classify  # type: ignore[method-assign]
    ModelPopulator.__call__ = traced_populate  # type: ignore[method-assign]
    SearchPlanner.run = traced_run  # type: ignore[method-assign]
    conductor.describe_facts = traced_describe_facts

    def stop() -> None:
        BaseDecisionProcess._apply = original_apply  # type: ignore[method-assign]
        EnumClassifier.__call__ = original_classify  # type: ignore[method-assign]
        ModelPopulator.__call__ = original_populate  # type: ignore[method-assign]
        SearchPlanner.run = original_run  # type: ignore[method-assign]
        conductor.describe_facts = original_describe_facts

    return stop
