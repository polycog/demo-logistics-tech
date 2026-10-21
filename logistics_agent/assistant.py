"""logistics support cogent that talks to a human through the terminal: its terminal interface to the human,
the builders that put the decision process and the cogent together, and the loop that runs it"""

from cognition import Actuator, Cogent, DecisionProcess, DecisionProcessErrorMessage, Sensor

from logistics_agent.communicator import Utterance
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


class HumanInteraction(Sensor[Utterance], Actuator[Utterance, None]):
    """sensor and actuator for talking to the human through terminal input and output

    sensor: reads one line typed by the human
    actuator: prints the cogent's reply
    """

    def __init__(self) -> None:
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
        try:
            phrase = input("you> ").strip()
        except EOFError:
            print()
            phrase = "quit"
        return Utterance(phrase=phrase or None)

    def actuate(self, param: Utterance) -> None:
        """prints the cogent's reply to the terminal"""
        if param.phrase:
            print(f"cogent> {param.phrase}")


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


def build_assistant() -> tuple[Cogent[DecisionProcess[ConversationState]], ConversationState]:
    """builds the cogent around the decision process, with the terminal interaction as its sensor and actuator"""
    dp, state = build_decision_process()
    assistant: Cogent[DecisionProcess[ConversationState]] = Cogent(decision_process=dp, name="logistics_support_agent")
    interaction = HumanInteraction()
    assistant.add_sensor(interaction).add_actuator(interaction)
    return assistant, state


def main() -> None:
    """runs the perceive-decide-act loop until the human quits"""
    assistant, state = build_assistant()

    def keep_going(_cogent: Cogent[DecisionProcess[ConversationState]]) -> bool:
        """continue looping until the human has said goodbye"""
        return not state.done

    def ignore_empty_input(err: DecisionProcessErrorMessage, _cogent: Cogent[DecisionProcess[ConversationState]]) -> bool:
        """an empty line proposes no operator; wait for the next line instead of failing"""
        if err is DecisionProcessErrorMessage.NO_PROPOSAL:
            return True
        raise RuntimeError(err)

    print("(type 'quit' to stop)")
    try:
        assistant(keep_going, dp_err_p=ignore_empty_input)
    except KeyboardInterrupt:
        print("\nExecution interrupted.")


if __name__ == "__main__":
    main()
