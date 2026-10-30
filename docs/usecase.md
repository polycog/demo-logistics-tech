# Problem Statement and Opportunity

## Problem statement

### How quotes are made today

Small-fleet carriers (1–10 trucks) operate on razor-thin margins and are increasingly overwhelmed by fast-turnaround spot and partial-load (LTL) requests arriving through unstructured channels like email and SMS. Lacking enterprise-grade pricing desks, owner-dispatchers price each request by hand, with back-of-the-envelope calculations and ad hoc Excel sheets. For example, to answer *"How much to ship 800 pounds of office chairs from Seattle to San Francisco?"*, a dispatcher typically must:

- Work out where each vehicle is right now, and whether a truck has to drive empty to the shipment before it can carry it.
- Look up which roads and air routes connect the cities, and whether any are closed.
- Plug today's fuel price, the route mileage, and each city's loading and unloading labor rates into a spreadsheet kept up to date by hand.
- Compare driving with flying, or with a combination of the two, and pick the cheapest option they can find.

These heuristics are slow and error-prone, and they fail to synthesize real-time telematics, driver Hours of Service (HOS) limits, and destination-lane liquidity. As a result, up to 30% of commercial truck miles run completely empty ([Foundation Capital](https://foundationcapital.com/ideas/overhauling-logistics-with-ai-a-79-billion-opportunity)).

### The cost of getting it wrong

A quote that is guessed rather than worked out hurts either way: pricing too low means the carrier accepts an underpriced run and loses margin, while pricing too high or answering too slowly loses the job to a competitor.

## Opportunity and challenge

According to [Foundation Capital](https://foundationcapital.com/ideas/overhauling-logistics-with-ai-a-79-billion-opportunity), solving these operational bottlenecks anchors a $30 billion AI market opportunity in route optimization and planning, where autonomous agents can eliminate deadhead waste, protect driver labor, and capture large back-office productivity savings by turning unstructured demand into dynamically priced, profitable routes in real time. However, capturing this value requires prices that are worked out from the carrier's actual fleet, routes, and costs, not estimated. A language model that guesses at a plan or a price, or hallucinates a vehicle or a route, produces exactly the kind of wrong quote the carrier cannot afford.

## Why this needs neurosymbolic AI

### What rules alone can't do

Routing and cost optimization are well-studied symbolic problems, but a solver needs a fully specified request. Quote requests arrive as emails and texts, often incomplete ("800 lb of chairs, Seattle to SF, how much?" with no pickup time, or "a couple of pallets"), and a web form that demands every field up front is exactly what a busy customer won't fill in. Someone still has to turn the message into a problem the solver can take, and ask for what's missing.

### What an LLM alone can't do

Ask an LLM for a quote and it returns a confident, plausible number. But:

- **it doesn't search,** so it misses that sending a truck from a nearby city is cheaper than using the one at the origin, or that flying one leg beats driving;
- **it does arithmetic unreliably** across fuel, mileage, and per-city labor rates;
- **it can invent fleet and network facts,** quoting a truck that is busy elsewhere, or a road that is closed;
- **it isn't consistent,** so two customers asking the same question can get different prices.

For a carrier on thin margins, an unexplainable quote that's too low is worse than no quote.

### Splitting the work

| Neural: understanding the request | Symbolic: pricing the delivery |
|---|---|
| Extract the shipment, weight, origin, and destination | Track the conversation and ask for missing parts |
| Recognize what the dispatcher wants to do | Keep the world model of cities, vehicles, and routes |
| | Search for the cheapest plan and compute its cost |
| | Explain the quote step by step |

## Impact

With agents, small-fleet carriers can automate shipping price quotes that are computed exactly from their own fleet and costs, with explainable, step-by-step plans. This replaces manual spreadsheet work, prevents margin loss, and keeps the carrier competitive and profitable in real time.

## Technical approach

See [the technical approach](approach.md) for how an agent with cognition inside meets this challenge.
