from typing import Callable
from pydantic import BaseModel


class ToolDefinition(BaseModel):
    """Definition of a tool in the registry."""
    name: str
    description: str
    input_schema: dict
    required_scopes: list[str]
    handler: Callable

    class Config:
        arbitrary_types_allowed = True
