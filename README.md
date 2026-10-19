# Logistics pricing support agent

A small conversational agent that prices package deliveries across a five-city logistics network. It is a demo of
[polycog cognition](https://pypi.org/project/polycog-cognition/): a language model understands what the human says,
while a world graph, a reasoner, and a search planner decide what is true and what a delivery costs.

```
cogent> Hi - I am your pricing support agent. I can help you estimate prices for various deliveries.
you> which trucks do we have?
cogent> Big Bertha is a Freightliner M2 106 26ft box truck with a max payload of 10,000 pounds and license plate
        C12345A. Little Rocket is an Isuzu NPR-HD 16ft box truck with a max payload of 6,000 pounds and license
        plate C67890B.
you> where is big bertha right now?
cogent> Big Bertha is currently at Seattle-Tacoma International Airport.
you> set the gas price to 0.95 per mile
cogent> Updated the gas price: per_mile is now 0.95.
you> how much would it cost to ship the crate?
cogent> 1. Drive truck 'Big Bertha' from Seattle-Tacoma International Airport to Seattle Office. Cost: $50.0
        ...
        Final Estimate: $879.0
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
| A price estimate | "how much to ship the crate to san francisco?" |
| Where a delivery should end up | "I want the crate of machine parts to end up in denver" |
| What the network has | "which trucks do we have?", "what is the gas price?" |
| Where things are right now | "where is big bertha?", "what is at the seattle airport?" |
| Add a package, truck, or airplane | "add a 200 lb package of books at the boston office" |
| Change something | "set the gas price to 0.95 per mile", "the crate now weighs 500 lb" |

A price estimate plans toward the current goal: the crate of machine parts at the San Francisco office, until you
say otherwise. Changes you make (a new gas price, a new package) feed straight into the next estimate.

## The world

Five cities — Seattle, San Francisco, Boston, Denver, and Detroit — each with one office and one airport.

- **Trucks** drive between an office and the airport in the same city, or between offices in different cities.
  Long-distance drives cost the gas price per mile.
- **Airplanes** fly between airports along air routes, each with its own price.
- **Packages** move only by being loaded into a vehicle. Loading and unloading cost the package's weight times the
  labor rate in that city.

At the start, the crate of machine parts (1000 lb) is at the Seattle office, Big Bertha and The Pelican (an airplane)
are at Seattle-Tacoma airport, and Little Rocket is at San Francisco airport. The planner searches for the cheapest
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
| `communicator.py` | Language understanding: the intents, the models the LLM fills in, and the LLM configuration. |
| `reasoner.py` | Builds and maintains the world graph, and answers queries and updates against it. |
| `planner.py` | The search planner: what each action does to the world, and the search for the cheapest plan. |
| `domain.py` | The domain model: cities, offices, airports, vehicles, packages, routes, costs, and actions. |

Every turn, the conductor recognizes the human's intent with an `EnumClassifier`, then one operator reacts to it.
Operators use `ModelPopulator` to turn the utterance into a typed model (for example, a `StateQuery` naming the
truck being asked about), ask the reasoner or planner for the answer, and use `describe_facts` to say it in plain
language. The language model never decides facts or prices — it only translates between the human and the graph.

## Running the tests

The tests run offline: fakes stand in for the language model, so no API key or local server is needed.

```bash
uv run pytest
```

## License

Licensed under the [Apache License 2.0](LICENSE).
