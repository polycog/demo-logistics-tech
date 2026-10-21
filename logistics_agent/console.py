"""how the conversation looks in the terminal, drawn with Rich: the agent's replies rendered as Markdown so a plan shows
as a list, each followed by a thin line, a colored prompt for the human, a spinner while the agent works, and, when
asked for, the steps of its reasoning

styling is only used when output goes to a terminal, so piped or captured output stays plain text, one reply per line
"""

import os
import shutil
import sys
from typing import TextIO

from rich.console import Console as RichConsole
from rich.markdown import Markdown
from rich.rule import Rule
from rich.status import Status
from rich.table import Table
from rich.text import Text

HUMAN_LABEL = "you"
AGENT_LABEL = "agent"
# what comes between a label and what is said, in plain output
SEPARATOR = " › "
HUMAN_PREFIX = f"{HUMAN_LABEL}{SEPARATOR}"
AGENT_PREFIX = f"{AGENT_LABEL}{SEPARATOR}"
THINKING = "thinking…"
# the widest the conversation is drawn, even in a wide terminal, so replies stay easy to read
MAX_WIDTH = 100
# the color of each kind of reasoning step: the conductor's, the planner's, and the language model's
STEP_STYLES = {"decision": "steel_blue", "planner": "yellow", "llm": "orchid"}


class Console:
    """the terminal the conversation is shown on"""

    def __init__(self, out: TextIO | None = None, styled: bool | None = None, width: int | None = None) -> None:
        """shows the conversation on out (standard output by default); styles it with Rich when out is a terminal and
        the NO_COLOR environment variable is not set, unless styled says otherwise; draws it at most width wide, or
        as wide as the terminal"""
        self.out = out or sys.stdout
        if styled is None:
            styled = self.out.isatty() and "NO_COLOR" not in os.environ
        self.styled = styled
        self.width = width or min(shutil.get_terminal_size().columns, MAX_WIDTH)
        self._rich = RichConsole(file=self.out, force_terminal=styled, width=self.width, highlight=False)
        # the spinner on screen while the agent works, until what comes next stops it
        self._status: Status | None = None

    def read_line(self) -> str:
        """prompts the human and reads the line they type"""
        if self.styled:
            return self._rich.input(f"[bold green]{HUMAN_PREFIX}[/]")
        return input(HUMAN_PREFIX)

    def show_reply(self, text: str) -> None:
        """shows what the agent says after its label: when styled, rendered as Markdown with its lines indented under
        the label, and followed by a thin line that separates it from what comes next; otherwise on one line"""
        self.stop_thinking()
        if self.styled:
            reply = Table.grid()
            reply.add_column(no_wrap=True)
            reply.add_column()
            reply.add_row(Text(AGENT_PREFIX, style="bold cyan"), Markdown(text))
            self._rich.print(reply)
            self._rich.print(Rule(style="grey35"))
        else:
            print(f"{AGENT_PREFIX}{text}", file=self.out)

    def show_note(self, text: str) -> None:
        """shows a remark that is not part of the conversation, such as how to stop"""
        self.stop_thinking()
        if self.styled:
            self._rich.print(Text(text, style="dim"))
        else:
            print(text, file=self.out)

    def start_thinking(self) -> None:
        """shows a spinner while the agent works on what the human said, until the agent says something; only when
        styled; steps of its reasoning shown meanwhile appear above the spinner"""
        if self.styled and self._status is None:
            self._status = self._rich.status(Text(THINKING, style="dim"), spinner="dots")
            self._status.start()

    def stop_thinking(self) -> None:
        """takes the spinner away, if it is on screen"""
        if self._status is not None:
            self._status.stop()
            self._status = None

    def show_progress(self, text: str) -> None:
        """shows the text beside the spinner in place of thinking, for work with too many steps to show one by one,
        such as the planner's search; only when styled, and while the spinner is on screen"""
        if self._status is not None:
            self._status.update(Text(text, style="dim"))

    def show_step(self, depth: int, source: str, action: str, kind: str = "decision") -> None:
        """shows one step of the agent's reasoning: what took it and what it did, indented by how deeply it is nested,
        in the color of its kind; it keeps any spinner, which stays below it, back to thinking"""
        self.show_progress(THINKING)
        line = f"{'  ' * (depth + 1)}· {source}: {action}"
        if self.styled:
            self._rich.print(Text(line, style=STEP_STYLES.get(kind, "dim")))
        else:
            print(line, file=self.out)
