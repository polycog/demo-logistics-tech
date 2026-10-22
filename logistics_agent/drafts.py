"""drafts: models being filled in over several turns whose fields are all required, such as a Truck or an At relation

a draft of a model has the same fields, each optional and None until it is known, with the same descriptions, which
are the questions the agent asks for them; so an AskFor of the model is satisfied by its draft, and the model is built
from the draft once nothing is missing
"""

from functools import cache

from pydantic import BaseModel, Field, create_model


@cache
def draft_model[M: BaseModel](model: type[M]) -> type[BaseModel]:
    """the model of drafts of the model: the same fields, each optional and None until known, with the same
    descriptions; made once per model"""
    draft = create_model(
        f"{model.__name__}Draft",
        __doc__=model.__doc__,
        **{
            name: (info.annotation | None, Field(default=None, description=info.description))
            for name, info in model.model_fields.items()
        },
    )
    draft.drafts = model
    return draft


def drafted(draft: BaseModel) -> type[BaseModel]:
    """the model the draft is a draft of"""
    return type(draft).drafts


def missing(draft: BaseModel) -> list[str]:
    """the fields of the draft not known yet, in the model's order"""
    return [name for name in type(draft).model_fields if getattr(draft, name) is None]


def build(draft: BaseModel) -> BaseModel:
    """the model the draft is of, built from its fields once none is missing"""
    assert not missing(draft), f"{drafted(draft).__name__} is missing {missing(draft)}"
    return drafted(draft).model_validate({name: getattr(draft, name) for name in type(draft).model_fields})
