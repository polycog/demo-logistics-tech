# Logistics support agent

Imagine a small carrier company. It has terminals and airport access in five cities (Seattle, San Francisco, Boston,
Denver, and Detroit), and it owns only a few vehicles: two box trucks and one cargo airplane. Customers keep asking
the same question: *how much will it cost to get my shipment there?*

Answering takes real work. The cheapest way to deliver depends on where each vehicle is right now, which roads and air
routes connect the cities, today's gas price, and what crews in each city charge to load and unload. A truck may
need to drive to the shipment before it can carry it. Sometimes flying beats driving, and sometimes it doesn't. A quote
that is guessed instead of worked out loses money or loses the customer.

This repository is a logistics support agent for that carrier. You talk to it in plain language. The goal is for it
to work out the cheapest sequence of drives, flights, loads, and unloads that gets the shipment where the customer
wants it, and reply with a step-by-step estimate.

> **Work in progress:** the human describes the whole delivery in one go: what is shipped, how much it weighs, and
> the cities it goes from and to. The agent does not yet ask for a part that is missing; a new delivery described
> later replaces the earlier one.

It is a demo of [polycog cognition](https://pypi.org/project/polycog-cognition/). A language model understands what
the human says, but it never sets a price. A world graph, a reasoner, and a search planner decide what is true and
what a delivery costs. The language model only words the plan, so the description of the steps varies from run
to run.

```
cogent> Hi - I am your logistics support agent. I can help you estimate prices for various deliveries.
you> hello there
cogent> Hello! I am happy to help with your deliveries.
you> I need to ship a pallet of office chairs, 800 pounds, from Seattle to San Francisco
cogent> Got it: the pallet of office chairs (800 lb) from Seattle to San Francisco. Ask me what it costs whenever you are ready.
you> how much will it cost?
cogent> Delivering the pallet of office chairs from Seattle to San Francisco costs $802.00. Big Bertha drives from the Seattle Terminal to the pickup in Seattle for $50, where the chairs are loaded for $48; it then drives to the drop-off in San Francisco for $640, where they are unloaded for $64.
you> thanks!
cogent> You're welcome.
you> bye
cogent> Goodbye!
```

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

Type your questions at the `you>` prompt. Type `quit`, `exit`, or `bye` (or press Ctrl+D) to stop.

### Without uv

The only dependencies are `polycog-cognition` and pydantic-ai's OpenAI client, both on PyPI. Installing the
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
```

| Module | Role |
| --- | --- |
| `assistant.py` | Runs the agent: reads lines from the terminal, prints replies, and loops until the human quits. |
| `conductor.py` | The decision process: the conversation state and one operator per thing the agent can do. |
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

## License

Licensed under the [Apache License 2.0](LICENSE).
