# Fleet updates

How the human adds a vehicle, such as a truck or an airplane, to the carrier's fleet. The agent fills in the vehicle,
then where it is, and asks the human to confirm before it changes the world graph. It uses the same intent stack and
asks as the shipping request ([model-filler.md](model-filler.md)), with two fillers specific to vehicles. Only adding
is supported. Vehicles added last only until the agent quits, because the world graph is rebuilt every run.

```
you › I want to add a new truck called Road Runner
agent › What make and model is the vehicle?
you › a Ford F-750 that carries 12000 pounds, plate 7XYZ123
agent › Where is it now: which terminal or airport?
you › at the Denver terminal
agent › Shall I add Road Runner, a Ford F-750 truck that carries up to 12,000 lb, at Denver Terminal, to the fleet?
you › ok
agent › Added Road Runner to the fleet, at Denver Terminal.
```

## The fleet and the vehicle types

Every vehicle is part of the carrier's fleet. `Vehicle` is tagged `FLEET` (`domain.py`), so `Truck` and `Airplane`
are too:

```python
class Vehicle(Entity):
    tags: ClassVar[frozenset[Tag]] = frozenset({Tag.FLEET})
    name_prefix: ClassVar[str] = "vehicle"     # Truck: "truck", Airplane: "air"
```

Tags are class variables, not fields: they never reach the language model's schema or the world graph, and a
subclass inherits or overrides them. Nothing reads the tag yet.

The kinds of vehicle the human can add are listed explicitly in the `VehicleType` `DocEnum`. Each member has a
description for the classifier, and knows its model:

```python
class VehicleType(DocEnum):
    """Which kind of vehicle would you like to add to the fleet: a truck or an airplane?"""

    TRUCK = ("truck", "a vehicle that travels by road, between terminals and airports")
    AIRPLANE = ("airplane", "a vehicle that travels by air, between airports")

    @property
    def model(self) -> type[Vehicle]: ...       # Truck, Airplane
```

A new kind of vehicle needs a model and a `VehicleType` member.

**Questions come from the model,** as for the shipping request:
- The vehicles' `Field` descriptions are written as questions ("What make and model is the vehicle?").
- So is `At.entity2` ("Where is it now: which terminal or airport?").
- The kind is asked with `VehicleType`'s own docstring, through the `AskForKind(VehicleType)` agent intent.

## Drafts

`ShippingRequest` could be half filled because every field is optional. A `Truck` or an `At` relation cannot: every
field is required. So the agent fills in a **draft** (`drafts.py`):

- `draft_model(Truck)` is a model with the same fields, each optional and `None` until known, and the same
  descriptions. It is made once per model.
- An `AskFor(Truck, …)` is satisfied by the draft, so the asks work as for any model.
- `missing(draft)` lists the fields not known yet, and `build(draft)` makes the real model once none is missing,
  running its validation.

## Two fills: the vehicle, then where it is

The vehicle and where it is are filled in separately. Each has its own sub-process, and each extracts the fields it
needs together, in one call to the language model.

### `VehicleFiller` (`fill_vehicle.py`)

Inputs on `io.a`: `utterance`, `answering` (the ask the utterance answers), `reasoner`, and `draft` (the draft so far,
or `None` before the kind is known).

| Operator | WHEN | THEN |
| --- | --- | --- |
| `TakeDraft` | the draft has not been taken | copy the conductor's draft, if any, and note its missing fields |
| `ChooseKind` | no draft, and the kind not tried on this utterance | classify the kind with an `EnumClassifier` over `VehicleType`, with the question asked as context, so it can only be one of them; start a draft of its model, named with `reasoner.fresh_name(model.name_prefix)`, such as `truck3` |
| `ExtractFields` | a draft with missing fields, not extracted yet | fill in the missing fields together; keep text that is not blank and numbers above zero |
| `ComposeAsks` | the utterance has been used | ask for the kind (`AskForKind(VehicleType)`) if there is still no draft; otherwise ask for each missing field, in order |

The agent names the vehicle itself, so `name` is never asked.

### `PlacementFiller` (`fill_placement.py`)

Inputs on `io.a`: `utterance`, `answering`, `reasoner`, and `draft`, a draft of `At` with the new vehicle as `entity1`.

The location is chosen among the locations the world graph already has: only airports for an airplane, and every
terminal and airport for any other vehicle (`places_for`). The language model gives a name, which is checked against
those places afterwards (`location_named`), so it can only pick a real one.

## How the conductor uses them

The conductor keeps `vehicle_draft` and `placement_draft` in its state. `model_of` maps each model the agent asks for
to them: a `Vehicle` model to the vehicle draft, and `At` to the placement draft. `ConversationState.has(ask)` says
whether the agent has what an ask asks for: the kind once there is a vehicle draft, and otherwise what `AskFor`'s
`satisfied` says of the draft. `PopSatisfiedAsk` pops `AskForKind(VehicleType)` once it has, and `AskQuestion` asks it
until then.

| Top of the stack | Operator | Then |
| --- | --- | --- |
| `ADD_TO_FLEET` | `ReactToAddToFleetIntent` | if the utterance answers where the vehicle is, run `PlacementFiller`; otherwise run `VehicleFiller`, continuing the vehicle being asked for or starting a new one. Pop the intent, then push the ask for the whole vehicle (or placement) with the asks for what is missing above it |
| `AskFor(Truck)` or `AskFor(Airplane)`, complete | `PlaceVehicle` | build the vehicle, start a draft of its `At` relation, and push `AskFor(At)` and `AskFor(At, "entity2")` |
| `AskFor(At)`, complete | `ProposeAddition` | build the placement, keep it as the pending change, a `VehicleAddition`, and swap the ask for `ConfirmChange` |
| `ConfirmChange` | `AskToConfirmChange` | ask the pending change's question ("Shall I add …?") and end the turn; it stays on the stack until a yes or no |
| `CONFIRM` | `ReactToConfirmIntent` | if `ConfirmChange` is under it, make the pending change: `VehicleAddition` adds the vehicle with `reasoner.add_entity` and the `At` relation the human described with `reasoner.add_relation`. Say so, and forget the addition; a yes to nothing is let go |
| `DENY` | `ReactToDenyIntent` | if `ConfirmChange` is under it, say the vehicle was not added, and forget the addition; a no to nothing is let go |

Confirming is shared with closing blocked routes ([route-closures.md](route-closures.md)): the pending change knows its
question, how to make itself, what to say when declined, and what to forget afterwards.

Attribute asks are popped by `PopSatisfiedAsk` and asked by `AskQuestion`, as for the shipping request.

**Which fill an utterance goes to.** `ADD_TO_FLEET` covers both asking to add a vehicle and describing it, including
where it is. When the ask right under the intent is for `At`, the utterance answers where the vehicle is. Otherwise it
describes the vehicle. Only one fill runs per utterance, so a location given before the vehicle is complete is not
used: the agent asks for it once the vehicle is done.

**Starting over.** A request to add a vehicle while the agent is not asking for one drops any addition that was not
finished, with every ask for it still on the stack (`forget_fleet_addition`).

**Yes and no.** `CONFIRM` and `DENY` are human intents like any other. While `ConfirmChange` is on top, "ok", "great"
and the other acknowledgements are classified as an answer, not let go silently by `Acknowledge`. Something else, such
as a greeting, is answered, and then the confirmation question is asked again.

The new vehicle is added to the same world graph the planner plans over, so the next price estimate can use it.

## Known limitations

- **Uniqueness.** Nothing stops a second vehicle with the same display name or registration. A check at `build`
  could clear that field and ask again.
- **One fill per utterance.** "Road Runner, a Ford F-750, at the Denver terminal" fills in the vehicle, and the agent
  then still asks where it is.
- **No changes after confirming a field.** A field filled in wrongly can only be fixed by saying no at the
  confirmation and starting over.

## Design suggestion: build `VehicleType` from the vehicles

Not implemented. Today `VehicleType` lists the vehicle types by hand, so a new kind of vehicle needs both a model and
a member, with its description written twice. Instead, the vehicles could declare themselves, and `VehicleType` could
be built from them.

A Python enum is closed once it is created, so `Truck` and `Airplane` cannot add members to an existing `VehicleType`.
They can register themselves with `Vehicle` when they are defined, and `domain.py` then builds `VehicleType` from that
registry, once, after the last vehicle class:

```python
class Vehicle(Entity):
    """base type for anything that can carry shipments ..."""
    # the vehicle types the human can add to the fleet, by name, filled in as subclasses are defined
    types: ClassVar[dict[str, type["Vehicle"]]] = {}

    def __init_subclass__(cls, vehicle_type: str | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if vehicle_type is not None:
            Vehicle.types[vehicle_type] = cls


class Truck(Vehicle, vehicle_type="truck"):
    """a vehicle that travels by road, between terminals and airports ..."""

class Airplane(Vehicle, vehicle_type="airplane"):
    """a vehicle that travels by air, between airports ..."""


class _VehicleTypes(DocEnum):
    @property
    def model(self) -> type[Vehicle]:
        return Vehicle.types[self.value]

# built once every vehicle type has been defined; each member is documented by its model's docstring
VehicleType = _VehicleTypes("VehicleType", {name.upper(): (name, model.__doc__) for name, model in Vehicle.types.items()})
VehicleType.__doc__ = "Which kind of vehicle would you like to add to the fleet: truck or airplane?"
```

A prototype of this produced the same members and the same `EnumClassifier` prompt as the hand-written `VehicleType`.

**Why a class keyword rather than every subclass.** `vehicle_type="truck"` makes joining the list explicit. Abstract
intermediate classes don't pass the keyword, and neither do classes the world graph or pydantic derive from the models
internally, so none of them become vehicle types. In the prototype, the world graph's frozen copies of the models did
not register.

**What it gains.** Each vehicle type is declared in one place. A new one is just
`class Van(Vehicle, vehicle_type="van")`, and its docstring is the classifier's description of it.

**What it costs:**

- **The order inside `domain.py` matters.** `VehicleType` must be built after the last vehicle class. A `Vehicle`
  subclass defined in another module, imported later, would not be included.
- **Type checkers cannot see the members.** `VehicleType.TRUCK` works at runtime, but mypy and IDEs won't know it
  exists, because the enum is built dynamically.
- **The question cannot be fully generated.** Generating it with articles gives "a truck or a airplane", so it would
  be generated without them ("truck or airplane"), or written by hand.
- **The docstrings become classifier prompts.** Each model's docstring goes to the classifier as the description of
  its type, so editing a docstring changes how kinds are classified.
