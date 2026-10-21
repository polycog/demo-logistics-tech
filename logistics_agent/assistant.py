"""logistics support cogent that talks to a human through the terminal: its terminal interface to the human,
the builders that put the decision process and the cogent together, and the loop that runs it"""

import argparse

from cognition import Actuator, Cogent, DecisionProcess, DecisionProcessErrorMessage, Sensor

from logistics_agent.communicator import Utterance
from logistics_agent.console import Console
from logistics_agent.conductor import (
    Acknowledge,
    AskForShippingRequest,
    ConversationState,
    DescribePlan,
    Introduce,
    ReactToAskCapabilitiesIntent,
    ReactToEstimatePriceIntent,
    ReactToGreetingIntent,
    ReactToShippingRequestIntent,
    RecognizeIntent,
    Reply,
    RespondUnknown,
    SayGoodbye,
)
from logistics_agent.trace import Step, trace

# how the reasoning steps name each decision process, by its type's name
PROCESS_LABELS = {"DecisionProcess": "conductor", "ExtractShippingRequest": "extract_shipping_request"}
# the decision process the planner searches with, whose steps are counted on the spinner instead of shown one by one
SEARCH_PROCESS = "BaseDecisionProcess"


class StepReporter:
    """shows the steps of the agent's reasoning on the console, one line per step, in the color of its kind, except
    the steps of the planner's search, which number in the hundreds and are counted beside the spinner"""

    def __init__(self, console: Console) -> None:
        self.console = console
        # the search steps the planner has taken since the last step shown
        self.searched = 0

    def __call__(self, step: Step) -> None:
        """reports one step"""
        if step.kind == "decision" and step.source == SEARCH_PROCESS:
            self.searched += 1
            self.console.show_progress(f"planning… {self.searched} search steps")
            return
        self.searched = 0
        self.console.show_step(step.depth, PROCESS_LABELS.get(step.source, step.source), step.action, step.kind)


class HumanInteraction(Sensor[Utterance], Actuator[Utterance, None]):
    """sensor and actuator for talking to the human through terminal input and output

    sensor: reads one line typed by the human
    actuator: shows the cogent's reply
    """

    def __init__(self, console: Console | None = None) -> None:
        """talks to the human on the console, which shows that the cogent is thinking while it works on a line"""
        self.console = console or Console()
        self._conversation_started = False

    @property
    def name(self) -> str:
        """name of the interface; the decision process reads it at io.i.interaction and writes to io.o.interaction"""
        return "interaction"

    def sense(self) -> Utterance:
        """waits for the human to type a line; end of input (Ctrl+D) counts as quitting

        the very first perception happens before the cogent has spoken, so nothing is read
        then; this gives the cogent a chance to introduce itself before the human is prompted
        """
        if not self._conversation_started:
            self._conversation_started = True
            return Utterance(phrase=None)
        self.console.stop_thinking()
        try:
            phrase = self.console.read_line().strip()
        except EOFError:
            print(file=self.console.out)
            phrase = "quit"
        if phrase:
            self.console.start_thinking()
        return Utterance(phrase=phrase or None)

    def actuate(self, param: Utterance) -> None:
        """shows the cogent's reply on the console"""
        if param.phrase:
            self.console.show_reply(param.phrase)


def build_decision_process() -> tuple[DecisionProcess[ConversationState], ConversationState]:
    """builds the decision process with its operators, over a fresh conversation state"""
    state = ConversationState()
    dp = DecisionProcess[ConversationState](lambda: state)
    dp.add_operator(Introduce("introduce", terminal=True))
    dp.add_operator(SayGoodbye("say_goodbye", terminal=True))
    dp.add_operator(Acknowledge("acknowledge", terminal=True))
    dp.add_operator(RecognizeIntent("recognize_intent"))
    dp.add_operator(RespondUnknown("respond_unknown"))
    # small talk
    dp.add_operator(ReactToGreetingIntent("react_to_greeting_intent"))
    dp.add_operator(ReactToAskCapabilitiesIntent("react_to_ask_capabilities_intent"))
    # taking shipping requests
    dp.add_operator(ReactToShippingRequestIntent("react_to_shipping_request_intent"))
    # estimating prices
    dp.add_operator(AskForShippingRequest("ask_for_shipping_request"))
    dp.add_operator(ReactToEstimatePriceIntent("react_to_estimate_price_intent"))
    dp.add_operator(DescribePlan("describe_plan"))
    # ending the turn once the stack is empty
    dp.add_operator(Reply("reply", terminal=True))
    return dp, state


def build_assistant(
    interaction: HumanInteraction | None = None,
) -> tuple[Cogent[DecisionProcess[ConversationState]], ConversationState]:
    """builds the cogent around the decision process, with the terminal interaction as its sensor and actuator"""
    dp, state = build_decision_process()
    assistant: Cogent[DecisionProcess[ConversationState]] = Cogent(decision_process=dp, name="logistics_support_agent")
    interaction = interaction or HumanInteraction()
    assistant.add_sensor(interaction).add_actuator(interaction)
    return assistant, state


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """the command-line options: argv, or the arguments the program was started with"""
    parser = argparse.ArgumentParser(description="talk to the logistics support agent")
    parser.add_argument(
        "--trace", action="store_true", help="show which operator each decision process applies, step by step"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """runs the perceive-decide-act loop until the human quits; with --trace, shows the steps of the agent's
    reasoning as it goes"""
    args = parse_args(argv)
    console = Console()
    stop_tracing = None
    if args.trace:
        # tracing must start before the decision processes are created, so it covers them all
        stop_tracing = trace(StepReporter(console))
    assistant, state = build_assistant(HumanInteraction(console))

    def keep_going(_cogent: Cogent[DecisionProcess[ConversationState]]) -> bool:
        """continue looping until the human has said goodbye"""
        return not state.done

    def ignore_empty_input(err: DecisionProcessErrorMessage, _cogent: Cogent[DecisionProcess[ConversationState]]) -> bool:
        """an empty line proposes no operator; wait for the next line instead of failing"""
        if err is DecisionProcessErrorMessage.NO_PROPOSAL:
            return True
        raise RuntimeError(err)

    console.show_note("(type 'quit' to stop)")
    try:
        assistant(keep_going, dp_err_p=ignore_empty_input)
    except KeyboardInterrupt:
        console.stop_thinking()
        console.show_note("\nExecution interrupted.")
    finally:
        if stop_tracing is not None:
            stop_tracing()


if __name__ == "__main__":
    main()
