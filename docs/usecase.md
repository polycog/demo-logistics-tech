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

According to [Foundation Capital](https://foundationcapital.com/ideas/overhauling-logistics-with-ai-a-79-billion-opportunity), solving these operational bottlenecks anchors a $30 billion AI market opportunity in route optimization and planning, where agents can eliminate deadhead waste, protect driver labor, and capture large back-office productivity savings by turning unstructured demand into dynamically priced, profitable routes in real time. However, capturing this value requires guaranteed accuracy in price estimation. A language model that guesses at a plan or a price, or hallucinates a vehicle or a route, produces exactly the kind of wrong quote the carrier cannot afford.

## Impact

With agents, small-fleet carriers can automate shipping price quotes with guaranteed accuracy and explainable, step-by-step plans. This replaces manual spreadsheet work, prevents margin loss, and keeps the carrier competitive and profitable in real time.

## Technical approach

See [the technical approach](approach.md) for how an agent with cognition inside meets this challenge.
