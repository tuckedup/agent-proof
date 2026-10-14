"""Base definitions and schema conversion for agentproof tools."""

import inspect
from typing import Any, Callable, Dict, Optional, Type

from pydantic import BaseModel


class ToolResult(BaseModel):
    """Standardized response from tool execution."""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
        }


class ToolDefinition:
    """Represents a callable tool with metadata and Pydantic argument model."""

    def __init__(
        self,
        name: str,
        description: str,
        args_model: Type[BaseModel],
        func: Callable[..., Any],
        is_write: bool = False,
        requestor: str = "assistant"
    ):
        self.name = name
        self.description = description
        self.args_model = args_model
        self.func = func
        self.is_write = is_write
        self.requestor = requestor

    def to_openai_schema(self) -> Dict[str, Any]:
        """Convert tool to OpenAI/LiteLLM function calling schema."""
        json_schema = self.args_model.model_json_schema()
        # Clean up json schema fields for OpenAI
        parameters = {
            "type": "object",
            "properties": json_schema.get("properties", {}),
            "required": json_schema.get("required", [])
        }
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            }
        }

    def execute(self, db: Any, **kwargs: Any) -> ToolResult:
        """Execute the tool function with validated arguments."""
        try:
            validated = self.args_model(**kwargs)
            sig = inspect.signature(self.func)
            if "db" in sig.parameters:
                result = self.func(db=db, **validated.model_dump())
            else:
                result = self.func(**validated.model_dump())

            if isinstance(result, ToolResult):
                return result
            return ToolResult(success=True, data=result)
        except Exception as e:
            return ToolResult(success=False, error=str(e))
