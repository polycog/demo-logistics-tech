# State inspection

How the agent answers the human's questions about the state of the world, starting with the carrier's fleet. Nothing
changes in the world, so there is nothing to confirm.

```
you › which trucks do you have?
agent › The fleet has 2 trucks: Big Bertha (Freightliner M2 106 26ft box truck), a truck that carries up to 10,000 lb,
        at Seattle Terminal; Little Rocket (Isuzu NPR-HD 16ft box truck), a truck that carries up to 6,000 lb, at San
        Francisco Terminal.
```

## The language model understands, the reasoner answers

The facts of an answer always come from the world graph, never from the language model:

1. **Classify the question:** an `EnumClassifier` over `QuestionType`, a `DocEnum` of the questions the agent can
   answer. A question of no known type gets "Sorry, I can only answer questions about the fleet …".
2. **Fill in what it is about:** one `ModelPopulator` call fills the subject model of that question type.
3. **Look up the facts:** the reasoner queries the world graph.
4. **Word the answer:** a template in the conductor fills a fixed sentence from the facts. It makes no LLM call, so
   the answer is exact and the same every time.

Each question type is a fixed query, so only known questions are answered, and a new question type is a new
`QuestionType` member with its subject model, look-up operator, and template.

## Question types

| `QuestionType` | Subject | Facts | Answer |
| --- | --- | --- | --- |
| `FLEET` | `FleetQuestion.kind`: a `VehicleType`, or `None` for the whole fleet | each vehicle of that kind, or every vehicle, in the order of their names, with where it is (`Sighting`) | "The fleet has 2 trucks: …", or "The fleet has no airplanes." |

`FleetQuestion.kind` is optional, so a question about the fleet never needs asking back: no kind means the whole
fleet. If the language model fails to fill it in, the whole fleet is answered too.

Where a vehicle is comes from its `At` relation (`place_of`). A vehicle the world graph does not place is answered
"at a place I do not know". Vehicles added to the fleet during the run are in the answer, because the reasoner looks
them up in the same world graph.

## `StateInspector` (`inspect_state.py`)

Inputs on `io.a`: `utterance`, `answering` (the ask the utterance answers, whose question is given as context), and
`reasoner`.

| Operator | WHEN | THEN |
| --- | --- | --- |
| `ClassifyQuestion` | the question has not been classified | classify it among the `QuestionType`s, or None |
| `FillFleetQuestion` | a fleet question whose subject is not filled in | fill in `FleetQuestion`; the whole fleet if that fails |
| `LookUpFleet` | the subject is filled in, the facts not looked up | look up the vehicles and where each is |

**Termination:** the question is classified as none the agent answers, or its facts are looked up.

## How the conductor uses it

| Top of the stack | Operator | Then |
| --- | --- | --- |
| `INSPECT_STATE` | `ReactToInspectStateIntent` | run `StateInspector`, note the answer worded by `fleet_answer`, or that the agent cannot answer that kind of question, and pop the intent |

Since the answer is only a partial response, whatever was under `INSPECT_STATE` carries on afterwards. A question
about the fleet asked while the agent waits for a destination gets its answer, then the destination question again.

### Wording

`fleet_answer` names each vehicle with its make and model, its kind, its payload, and where it is. The world graph gives
back frozen copies of the vehicles, whose class name is not the model's, so the kind is the `VehicleType` whose model
the vehicle is an instance of (`kind_of`). `with_article` puts "a" or "an" in front of the kind.

## Known limitations

- **Only the fleet.** Where one vehicle is, what is at a place, a vehicle's details, routes, and prices are not
  answered yet. Each would be a new `QuestionType`.
- **Long answers.** The whole fleet is one sentence, which grows with every vehicle added.
