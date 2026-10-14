"""Base definitions and schema conversion for agentproof tools."""

import inspect
from typing import Any, Callable, Dict, Optional, Type

from pydantic import BaseModel


class ToolResult(BaseModel):
    """Standardized response from tool execution."""
    success: bool = True
    data: Optional[Any] = None
    error: Optional[str] = None
    is_error: Optional[bool] = None
    error_message: Optional[str] = None

    def model_post_init(self, __context: Any) -> None:
        if self.is_error is not None:
            self.success = not self.is_error
        else:
            self.is_error = not self.success

        if self.error_message is not None and self.error is None:
            self.error = self.error_message
        elif self.error is not None and self.error_message is None:
            self.error_message = self.error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "is_error": self.is_error,
            "error_message": self.error_message,
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
