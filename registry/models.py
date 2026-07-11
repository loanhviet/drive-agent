from typing import Callable

from pydantic import BaseModel, ConfigDict


class ToolDefinition(BaseModel):
    """Definition of a tool in the registry."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    input_schema: dict
    required_scopes: list[str]
    handler: Callable
