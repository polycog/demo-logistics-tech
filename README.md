# Logistics support agent

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/downloads/)
[![cognition inside](https://img.shields.io/badge/cognition-inside-7b2fbe?labelColor=f3effa&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjIgMCAzNiAzNiIgZmlsbD0ibm9uZSI%2BPHBvbHlnb24gcG9pbnRzPSIyMCw0IDM0LjMsMTQuNCAyOC44LDMxLjYgMTEuMiwzMS42IDUuNywxNC40IiBzdHJva2U9IiM3YjJmYmUiIHN0cm9rZS13aWR0aD0iMi4yIi8%2BPGcgc3Ryb2tlPSIjYzQ3ZDIwIiBzdHJva2Utd2lkdGg9IjIiPjxsaW5lIHgxPSIyMCIgeTE9IjIwIiB4Mj0iMjAiIHkyPSI0Ii8%2BPGxpbmUgeDE9IjIwIiB5MT0iMjAiIHgyPSIzNC4zIiB5Mj0iMTQuNCIvPjxsaW5lIHgxPSIyMCIgeTE9IjIwIiB4Mj0iMjguOCIgeTI9IjMxLjYiLz48bGluZSB4MT0iMjAiIHkxPSIyMCIgeDI9IjExLjIiIHkyPSIzMS42Ii8%2BPGxpbmUgeDE9IjIwIiB5MT0iMjAiIHgyPSI1LjciIHkyPSIxNC40Ii8%2BPC9nPjxnIGZpbGw9IiM3YjJmYmUiPjxjaXJjbGUgY3g9IjIwIiBjeT0iNCIgcj0iMy4yIi8%2BPGNpcmNsZSBjeD0iMzQuMyIgY3k9IjE0LjQiIHI9IjMuMiIvPjxjaXJjbGUgY3g9IjI4LjgiIGN5PSIzMS42IiByPSIzLjIiLz48Y2lyY2xlIGN4PSIxMS4yIiBjeT0iMzEuNiIgcj0iMy4yIi8%2BPGNpcmNsZSBjeD0iNS43IiBjeT0iMTQuNCIgcj0iMy4yIi8%2BPC9nPjxjaXJjbGUgY3g9IjIwIiBjeT0iMjAiIHI9IjQuNiIgZmlsbD0iI2M0N2QyMCIvPjwvc3ZnPgo%3D)](https://github.com/polycog/cognition)
[![cognition on PyPI](https://img.shields.io/pypi/v/polycog-cognition?label=cognition%20on%20PyPI&color=c47d20&labelColor=f3effa&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjIgMCAzNiAzNiIgZmlsbD0ibm9uZSI%2BPHBvbHlnb24gcG9pbnRzPSIyMCw0IDM0LjMsMTQuNCAyOC44LDMxLjYgMTEuMiwzMS42IDUuNywxNC40IiBzdHJva2U9IiM3YjJmYmUiIHN0cm9rZS13aWR0aD0iMi4yIi8%2BPGcgc3Ryb2tlPSIjYzQ3ZDIwIiBzdHJva2Utd2lkdGg9IjIiPjxsaW5lIHgxPSIyMCIgeTE9IjIwIiB4Mj0iMjAiIHkyPSI0Ii8%2BPGxpbmUgeDE9IjIwIiB5MT0iMjAiIHgyPSIzNC4zIiB5Mj0iMTQuNCIvPjxsaW5lIHgxPSIyMCIgeTE9IjIwIiB4Mj0iMjguOCIgeTI9IjMxLjYiLz48bGluZSB4MT0iMjAiIHkxPSIyMCIgeDI9IjExLjIiIHkyPSIzMS42Ii8%2BPGxpbmUgeDE9IjIwIiB5MT0iMjAiIHgyPSI1LjciIHkyPSIxNC40Ii8%2BPC9nPjxnIGZpbGw9IiM3YjJmYmUiPjxjaXJjbGUgY3g9IjIwIiBjeT0iNCIgcj0iMy4yIi8%2BPGNpcmNsZSBjeD0iMzQuMyIgY3k9IjE0LjQiIHI9IjMuMiIvPjxjaXJjbGUgY3g9IjI4LjgiIGN5PSIzMS42IiByPSIzLjIiLz48Y2lyY2xlIGN4PSIxMS4yIiBjeT0iMzEuNiIgcj0iMy4yIi8%2BPGNpcmNsZSBjeD0iNS43IiBjeT0iMTQuNCIgcj0iMy4yIi8%2BPC9nPjxjaXJjbGUgY3g9IjIwIiBjeT0iMjAiIHI9IjQuNiIgZmlsbD0iI2M0N2QyMCIvPjwvc3ZnPgo%3D)](https://pypi.org/project/polycog-cognition/)

This repository is a neurosymbolic agent built with the
[polycog cognition](https://pypi.org/project/polycog-cognition/) library. It pairs a neural part, a language model,
with a symbolic part, a world graph reasoner, and a search planner, and gives each the job it does best. The
language model understands what a human says and generates the agent's replies; the symbolic part reasons with ground truth and makes decisions.
Every answer the agent gives is reasoned out from what it knows, not guessed by the language
model.

**[▶ Watch the demo](https://drive.google.com/file/d/1rDBJO3MaF7s5Xn8SjfjTN-VBQL3Mcc5_/view?usp=sharing)** to see the agent at work.

## The use case: quoting deliveries

Imagine a small carrier company. It has terminals and airport access in five cities (Seattle, San Francisco, Boston,
Denver, and Detroit), and it owns only a few vehicles: two box trucks and one cargo airplane. Customers keep asking
the same question: *how much will it cost to get my shipment there?*

Answering takes real work. The cheapest way to deliver depends on where each vehicle is right now, which roads and air
routes connect the cities, today's gas price, and what crews in each city charge to load and unload. A truck may
need to drive to the shipment before it can carry it. Sometimes flying beats driving, and sometimes it doesn't. A quote
that is guessed instead of worked out loses money or loses the customer.

The agent is a logistics support agent for that carrier. You describe your delivery in plain language, and it works
out the cheapest sequence of drives, flights, loads, and unloads that gets the shipment where you want it, and replies
with a step-by-step estimate. The language model understands your request, but it never sets a price: the planner
finds the plan and its cost, and the language model only words the plan, so the description of the steps varies from
run to run.

> **Work in progress:** the human describes the whole delivery in one go: what is shipped, how much it weighs, and
> the cities it goes from and to. The agent does not yet ask for a part that is missing; a new delivery described
> later replaces the earlier one.

## Quick start

You need [uv](https://docs.astral.sh/uv/getting-started/installation/) and either an OpenAI API key or a local model
(see [Choosing a language model](#choosing-a-language-model)).

```bash
git clone https://github.com/polycog/demo-logistics-tech.git
cd demo-logistics-tech
uv sync                                  # creates .venv with Python 3.13+ and all dependencies
export OPENAI_API_KEY=sk-...             # or use a local model, see below
uv run logistics-agent
```

Type your questions at the `you ›` prompt. Type `quit`, `exit`, or `bye` (or press Ctrl+D) to stop.

In a terminal, the console is drawn with [Rich](https://github.com/Textualize/rich): each of the agent's replies
is rendered as Markdown so a plan shows as a list, with a thin line after it, and a spinner turns while the agent works (set
`NO_COLOR=1` for plain text). To watch the cognition at work, add `--trace`: it shows, step by step, which operator
the conductor and its sub-processes apply, when the planner searches (in yellow), and where a language model is used,
each in its own color (the planner's own search steps are counted beside the spinner rather than listed):

```
uv run logistics-agent --trace
you › I need to ship a pallet of office chairs, 800 pounds, from Seattle to San Francisco
  · conductor: recognize_intent
    · llm: classify HumanIntent
  · conductor: react_to_shipping_request_intent
    · extract_shipping_request: extract_origin
      · llm: populate Place
    · extract_shipping_request: extract_destination
      · llm: populate Place
    · extract_shipping_request: extract_shipment
      · llm: populate Shipment
  · conductor: reply
agent › Got it: the pallet of office chairs (800 lb) from Seattle to San Francisco. Ask me what it costs whenever you are ready.
```

### Without uv

The only dependencies are `polycog-cognition`, pydantic-ai's OpenAI client, and Rich, all on PyPI. Installing the
project with pip pulls them in and adds the `logistics-agent` command:

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.13 or newer
pip install -e .
logistics-agent
```

## Choosing a language model

`llm_config.toml` picks the model the agent talks through:

```toml
provider = "openai"          # "openai" or "local"

[openai]
model = "gpt-4o"             # the API key is read from OPENAI_API_KEY

[local]
model = "openai/gpt-oss-20b"
base_url = "http://localhost:1234/v1"
api_key = "lm-studio"        # LM Studio accepts any key
```

- **OpenAI** — set `OPENAI_API_KEY` in your environment. Never put the key in `llm_config.toml`.
- **Local** — any server with an OpenAI-compatible API works. With [LM Studio](https://lmstudio.ai), download
  `openai/gpt-oss-20b`, load it, and start the local server (it listens on `http://localhost:1234/v1`).

To switch for a single run without editing the file, set `LLM_PROVIDER`:

```bash
LLM_PROVIDER=local uv run logistics-agent
```

Smaller local models are less reliable at producing structured output than `gpt-4o`, so expect the occasional
misunderstanding.

## What you can ask

| Ask | Example |
| --- | --- |
| Say hello | "hello there" |
| What the agent can do | "what can you do?" |
| Acknowledge the agent | "thanks" (the agent replies and repeats any question it was waiting on); "ok", "got it" (the agent says nothing) |
| Stop | "quit", "exit", "bye" |

## The world

Five cities — Seattle, San Francisco, Boston, Denver, and Detroit — each with one terminal and one airport.

- **Trucks** drive between locations in the same city, or between locations in two cities that a truck route
  connects, as long as both cities have a terminal. Long-distance drives cost the gas price per mile of the route.
- **Airplanes** fly between airports along air routes, each with its own price.
- **Shipments** move only by being loaded into a vehicle. Loading and unloading cost the shipment's weight times the
  labor rate in that city.

At the start, Big Bertha and The Pelican (an airplane) are at Seattle-Tacoma airport, and Little Rocket is at San
Francisco airport. There are no shipments at the start, and a shipment is wherever you say it is. The planner searches for the cheapest
sequence of drives, flights, loads, and unloads that reaches the goal.

## How it works

The agent lives in the `logistics_agent` package, one job per module:

```
logistics_agent/   the agent
tests/             offline tests
llm_config.toml    which language model to use
pyproject.toml     dependencies and the logistics-agent command
LICENSE, NOTICE    the Apache License 2.0, and the copyright notice
CITATION.cff       how to cite this repository
```

| Module | Role |
| --- | --- |
| `assistant.py` | Runs the agent: reads lines from the terminal, prints replies, and loops until the human quits. |
| `conductor.py` | The decision process: the conversation state and one operator per thing the agent can do. |
| `console.py` | How the conversation looks in the terminal, drawn with Rich: the replies and the thin lines between them, the prompt, the spinner, and the reasoning steps. |
| `trace.py` | Reports, for `--trace`, which operator each decision process applies, when the planner searches, and where a language model is used. |
| `extract_shipping_request.py` | The sub-process that extracts a shipping request from what the human says: the cities it goes from and to, which can only be cities the carrier serves, and the shipment. |
| `communicator.py` | Language understanding: the intents, the models the LLM fills in, and the LLM configuration. |
| `reasoner.py` | Builds and maintains the world graph, and answers queries and updates against it. |
| `planner.py` | The search planner: what each action does to the world, and the search for the cheapest plan. |
| `domain.py` | The domain model: cities, locations (terminals and airports, each in a city), vehicles, shipments, routes, costs, actions, and the shipping query (a shipment, the terminal it is sent from, and the terminal it goes to). |

The conductor manages the conversation with a **stack of intents**. Every turn, it recognizes the human's intent
with an `EnumClassifier` and pushes it onto the stack; the intent on top decides what happens next.
The intents live in `communicator.py` as `HumanIntent`, what the human wants. Each intent is popped once it is done.

The language model never decides facts or prices — it only translates between the human and the graph.

## Running the tests

The tests run offline: fakes stand in for the language model, so no API key or local server is needed.

```bash
uv run pytest
```

## Citation

If you use this agent in your work, please cite it. The citation metadata is in [`CITATION.cff`](CITATION.cff), which
GitHub also offers through the "Cite this repository" button:

```bibtex
@software{polycog_logistics_support_agent,
  author  = {{Polycog}},
  title   = {Logistics support agent: a demo of polycog cognition},
  url     = {https://github.com/polycog/demo-logistics-tech},
  version = {0.1.0},
  license = {Apache-2.0},
  year    = {2026}
}
```

## License

Copyright 2026 Polycog. Licensed under the [Apache License 2.0](LICENSE); see also [`NOTICE`](NOTICE).
