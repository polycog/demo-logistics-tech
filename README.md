# Logistics support agent

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/downloads/)
[![cognition inside](https://img.shields.io/badge/cognition-inside-7b2fbe?labelColor=f3effa&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjIgMCAzNiAzNiIgZmlsbD0ibm9uZSI%2BPHBvbHlnb24gcG9pbnRzPSIyMCw0IDM0LjMsMTQuNCAyOC44LDMxLjYgMTEuMiwzMS42IDUuNywxNC40IiBzdHJva2U9IiM3YjJmYmUiIHN0cm9rZS13aWR0aD0iMi4yIi8%2BPGcgc3Ryb2tlPSIjYzQ3ZDIwIiBzdHJva2Utd2lkdGg9IjIiPjxsaW5lIHgxPSIyMCIgeTE9IjIwIiB4Mj0iMjAiIHkyPSI0Ii8%2BPGxpbmUgeDE9IjIwIiB5MT0iMjAiIHgyPSIzNC4zIiB5Mj0iMTQuNCIvPjxsaW5lIHgxPSIyMCIgeTE9IjIwIiB4Mj0iMjguOCIgeTI9IjMxLjYiLz48bGluZSB4MT0iMjAiIHkxPSIyMCIgeDI9IjExLjIiIHkyPSIzMS42Ii8%2BPGxpbmUgeDE9IjIwIiB5MT0iMjAiIHgyPSI1LjciIHkyPSIxNC40Ii8%2BPC9nPjxnIGZpbGw9IiM3YjJmYmUiPjxjaXJjbGUgY3g9IjIwIiBjeT0iNCIgcj0iMy4yIi8%2BPGNpcmNsZSBjeD0iMzQuMyIgY3k9IjE0LjQiIHI9IjMuMiIvPjxjaXJjbGUgY3g9IjI4LjgiIGN5PSIzMS42IiByPSIzLjIiLz48Y2lyY2xlIGN4PSIxMS4yIiBjeT0iMzEuNiIgcj0iMy4yIi8%2BPGNpcmNsZSBjeD0iNS43IiBjeT0iMTQuNCIgcj0iMy4yIi8%2BPC9nPjxjaXJjbGUgY3g9IjIwIiBjeT0iMjAiIHI9IjQuNiIgZmlsbD0iI2M0N2QyMCIvPjwvc3ZnPgo%3D)](https://github.com/polycog/cognition)
[![cognition on PyPI](https://img.shields.io/pypi/v/polycog-cognition?label=cognition%20on%20PyPI&color=c47d20&labelColor=f3effa&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjIgMCAzNiAzNiIgZmlsbD0ibm9uZSI%2BPHBvbHlnb24gcG9pbnRzPSIyMCw0IDM0LjMsMTQuNCAyOC44LDMxLjYgMTEuMiwzMS42IDUuNywxNC40IiBzdHJva2U9IiM3YjJmYmUiIHN0cm9rZS13aWR0aD0iMi4yIi8%2BPGcgc3Ryb2tlPSIjYzQ3ZDIwIiBzdHJva2Utd2lkdGg9IjIiPjxsaW5lIHgxPSIyMCIgeTE9IjIwIiB4Mj0iMjAiIHkyPSI0Ii8%2BPGxpbmUgeDE9IjIwIiB5MT0iMjAiIHgyPSIzNC4zIiB5Mj0iMTQuNCIvPjxsaW5lIHgxPSIyMCIgeTE9IjIwIiB4Mj0iMjguOCIgeTI9IjMxLjYiLz48bGluZSB4MT0iMjAiIHkxPSIyMCIgeDI9IjExLjIiIHkyPSIzMS42Ii8%2BPGxpbmUgeDE9IjIwIiB5MT0iMjAiIHgyPSI1LjciIHkyPSIxNC40Ii8%2BPC9nPjxnIGZpbGw9IiM3YjJmYmUiPjxjaXJjbGUgY3g9IjIwIiBjeT0iNCIgcj0iMy4yIi8%2BPGNpcmNsZSBjeD0iMzQuMyIgY3k9IjE0LjQiIHI9IjMuMiIvPjxjaXJjbGUgY3g9IjI4LjgiIGN5PSIzMS42IiByPSIzLjIiLz48Y2lyY2xlIGN4PSIxMS4yIiBjeT0iMzEuNiIgcj0iMy4yIi8%2BPGNpcmNsZSBjeD0iNS43IiBjeT0iMTQuNCIgcj0iMy4yIi8%2BPC9nPjxjaXJjbGUgY3g9IjIwIiBjeT0iMjAiIHI9IjQuNiIgZmlsbD0iI2M0N2QyMCIvPjwvc3ZnPgo%3D)](https://pypi.org/project/polycog-cognition/)

## 1. Summary

A **neurosymbolic** logistics support agent, built with the [`cognition`](https://github.com/polycog/cognition) library, that helps a small-fleet cargo carrier quote delivery prices. The carrier dispatcher describes a shipment in plain language, and the agent works out the quote given the carrier's fleet, routes, labor rates, and fuel prices, and replies with a step-by-step estimate. The language model never sets a price: a symbolic planner finds the plan and its cost.

### Contents

1. [Summary](#1-summary)
2. [Problem Statement](#2-problem-statement)
3. [Technical Approach and Demo](#3-technical-approach-and-demo)
4. [Install, Run, and Test](#4-install-run-and-test)
5. [License and Disclaimers](#5-license-and-disclaimers)

---

## 2. Problem Statement

Small-fleet carriers run on thin margins and field a steady stream of fast-turnaround quote requests, often by email or text. They have no pricing desk, so owner-dispatchers price by hand. The cheapest way to deliver depends on where each vehicle is right now, which roads and air routes connect the cities, today's gas price, and what crews in each city charge to load and unload. A truck may need to drive to the shipment before it can carry it, and sometimes flying beats driving. A quote that is guessed rather than worked out either loses money or loses the job. An agent could let customers simply ask, e.g. *"How much to ship 800 pounds of office chairs from Seattle to San Francisco?"*, but only if its prices come from the carrier's actual fleet and costs, with no hallucinated numbers.

See [the full problem statement and opportunity](docs/usecase.md).

---

## 3. Technical Approach and Demo

The agent prices a delivery the way a careful dispatcher would, while letting the dispatcher speak naturally. An LLM only interprets what the dispatcher types, turning it into structured proposals: their intent, and the shipment, its weight, and the cities it goes from and to. When a part is missing, the agent asks for it, and keeps asking until it has the whole delivery. Symbolic WHEN/THEN operators, written with `cognition`, decide everything else: they manage the conversation through a stack of intents, keep a world graph of the carrier's cities, vehicles, and routes, and hand the delivery to a search planner that finds the cheapest plan and its cost. With no LLM in these decisions, the same request always gets the same price, and every step can be traced. The same approach lets the agent add vehicles to the fleet, close blocked routes, and answer questions about the fleet. See [the technical approach](docs/approach.md) for details.

**Demo:** [watch the agent quote a delivery](https://drive.google.com/file/d/1KbiW5nJUbtehfRNWvjA9JTc0np8MuhKm/view?usp=sharing)

---

## 4. Install, Run, and Test

### 4.1 Clone the repository

```bash
git clone https://github.com/polycog/demo-logistics-tech.git
cd demo-logistics-tech
```

### 4.2 Install the dependencies

`cognition` requires **Python 3.13 or later**. The project is managed with [uv](https://docs.astral.sh/uv/), which creates the environment and installs everything in [`pyproject.toml`](pyproject.toml):

```bash
uv sync
```

This installs:

- **`polycog-cognition`**, the `cognition` library, published on PyPI under that name;
- **`pydantic-ai-slim[openai]`**, which `cognition` builds on, with the OpenAI extra for the language model;
- **`rich`**, for the terminal console.

Without uv, install the project with pip, which also adds the `logistics-agent` command:

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.13 or newer
pip install -e .
```

### 4.3 Choose a language model

[`llm_config.toml`](llm_config.toml) picks the model the agent talks through. Set `provider` to one of:

| Provider | Model | Needs |
|---|---|---|
| `openai` (default) | `gpt-4o` | `OPENAI_API_KEY` |
| `local` | `openai/gpt-oss-20b` | An OpenAI-compatible server at `http://localhost:1234/v1` (such as LM Studio) |

For `openai`, export your key in the shell you run the agent from, and never put it in `llm_config.toml`:

```bash
export OPENAI_API_KEY="sk-..."
```

For `local`, with [LM Studio](https://lmstudio.ai), download `openai/gpt-oss-20b`, load it, and start the local server. To switch for a single run without editing the file, set `LLM_PROVIDER`:

```bash
LLM_PROVIDER=local uv run logistics-agent
```

Smaller local models are less reliable at producing structured output than `gpt-4o`, so expect the occasional misunderstanding.

### 4.4 Run the agent

```bash
uv run logistics-agent            # add --trace to see which operator runs each step
```

Describe your delivery at the `you ›` prompt, e.g. "I need to ship a pallet of office chairs, 800 pounds, from Seattle to San Francisco", then ask what it costs. Type `quit` or press Ctrl+D to stop. See [the technical approach](docs/approach.md#what-you-can-ask) for everything else you can ask, and what `--trace` shows.

### 4.5 Run the tests

The test suite uses [pytest](https://docs.pytest.org/) and runs fully offline: fakes stand in for the language model, so it needs no API key or local server and makes no paid calls.

```bash
uv run pytest
```

---

## 5. License and Disclaimers

This project is released under the [Apache License 2.0](LICENSE). Copyright © 2026 Polycog, Inc. If you use it, please cite it using the metadata in [`CITATION.cff`](CITATION.cff).

> **Disclaimer:** this project is a technical demonstration of neurosymbolic reasoning. It is not a production pricing or dispatch system, and its quotes are not binding offers. The carrier, its fleet, routes, and costs are fictional.
