"""ExtractAndSetGoal: the decision sub-process that turns the human's desired outcome into the planner's goal

the parent operator (ReactToDesiredOutcomeIntent) creates it once, then for every outcome restarts it and calls it
with three arguments, which its operators read from io.a:
    outcome: the human's utterance stating where things should end up
    reasoner: the reasoner, whose world graph names the things the human can refer to
    planner: the planner whose goal is set
it runs until it has worded a reply, and returns its final GoalState
"""

from dataclasses import dataclass
from typing import Any

from cognition import DecisionProcess, IOContainer, ModelPopulator, Operator
from pydantic import ValidationError

from logistics_agent.communicator import DEFAULT_LLM, DesiredOutcome, Utterance, known_entities
from logistics_agent.domain import At

ASK_AGAIN = "Sorry, I did not catch where things should end up. Where should the package go?"


@dataclass
class GoalState:
    """state of the sub-process, kept for one outcome

    desired_outcome: the outcome as the language model filled it in, naming things by name
    goal: the outcome resolved into At facts; empty when nothing in it could be resolved
    reply: what to tell the human; the sub-process is done once it is set
    """

    desired_outcome: DesiredOutcome | None = None
    goal: list[At] | None = None
    reply: str | None = None


class ExtractOutcome(Operator[GoalState]):
    """
    WHEN: the outcome has not been extracted yet
    THEN: fill in a DesiredOutcome from the human's utterance, naming the things the reasoner knows about
    """

    TASK_DESC = (
        "a human is telling a logistics support agent where they want packages or vehicles to end up; "
        "the agent has one office and one airport in each city it serves, and a package sent to a city "
        "should end up at that city's office unless the human names the airport or a vehicle"
    )

    def __init__(self, name: str, **kwargs: Any) -> None:
        """sets up the outcome populator once, so its prompt is reused for every utterance"""
        super().__init__(name, **kwargs)
        self._outcome_populator = ModelPopulator(DesiredOutcome, task_desc=self.TASK_DESC)

    def can_perform(self, state: GoalState, io: IOContainer) -> bool:
        """applies once, at the start of the sub-process"""
        return state.desired_outcome is None

    def perform(self, state: GoalState, io: IOContainer) -> None:
        """populates the desired outcome from the outcome utterance the parent supplied"""
        outcome: Utterance = io.a.outcome
        assert outcome.phrase is not None, "expect the parent operator to supply what the human said"
        state.desired_outcome = self._outcome_populator(
            outcome.phrase, DEFAULT_LLM, *known_entities(io.a.reasoner.world)
        )


class ResolveGoal(Operator[GoalState]):
    """
    WHEN: the outcome has been extracted but not resolved into goal facts
    THEN: look up each named thing with the reasoner and build the At facts; no facts if any name is unknown or a
          fact cannot hold
    """

    def can_perform(self, state: GoalState, io: IOContainer) -> bool:
        """applies once the desired outcome is known"""
        return state.desired_outcome is not None and state.goal is None

    def perform(self, state: GoalState, io: IOContainer) -> None:
        """resolves the desired outcome into At facts, or into no facts when it cannot"""
        assert state.desired_outcome is not None
        try:
            state.goal = state.desired_outcome.to_relations(io.a.reasoner.find_entity)
        except (KeyError, ValidationError):
            state.goal = []


class SetGoal(Operator[GoalState]):
    """
    WHEN: the outcome resolved into goal facts
    THEN: set them as the planner's goal, and confirm them back to the human
    """

    def can_perform(self, state: GoalState, io: IOContainer) -> bool:
        """applies when there are goal facts and no reply yet"""
        return bool(state.goal) and state.reply is None

    def perform(self, state: GoalState, io: IOContainer) -> None:
        """sets the planner's goal and words the confirmation"""
        assert state.goal
        io.a.planner.set_goal(state.goal)
        outcomes = "; ".join(f"the {fact.entity1.display_name} at the {fact.entity2.display_name}" for fact in state.goal)
        state.reply = f"Got it: {outcomes}. Ask me for a price estimate whenever you are ready."


class AskAgain(Operator[GoalState]):
    """
    WHEN: the outcome resolved into no goal facts
    THEN: leave the planner's goal as it is, and ask the human where things should end up
    """

    def can_perform(self, state: GoalState, io: IOContainer) -> bool:
        """applies when resolving produced no facts and there is no reply yet"""
        return state.goal == [] and state.reply is None

    def perform(self, state: GoalState, io: IOContainer) -> None:
        """words the request to say the outcome again"""
        state.reply = ASK_AGAIN


def has_reply(state: GoalState, _io: IOContainer) -> bool:
    """the sub-process is done once it has worded a reply"""
    return state.reply is not None


class ExtractAndSetGoal(DecisionProcess[GoalState]):
    """decision sub-process that extracts the goal from the human's desired outcome, sets it on the planner, and
    words the reply; see the module docstring for the arguments it is called with"""

    def __init__(self) -> None:
        """registers the operators and the termination check over a fresh GoalState"""
        super().__init__(GoalState)
        self.add_operator(ExtractOutcome("extract_outcome"))
        self.add_operator(ResolveGoal("resolve_goal"))
        self.add_operator(SetGoal("set_goal"))
        self.add_operator(AskAgain("ask_again"))
        self.add_termination_check(has_reply)
