"""Base interface, action models, and mock provider for agent systems under test."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from agentproof.tools.base import ToolDefinition


class AgentAction(BaseModel):
    """Structured action or final utterance emitted by an agent during a simulation turn."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tool_name: Optional[str] = Field(
        default=None,
        description="Identifier of the specific tool to execute, or None if providing a final response.",
    )
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="Key-value arguments conforming to the tool parameter schema.",
    )
    thought: Optional[str] = Field(
        default=None,
        description="Internal scratchpad, reasoning chain, or justification prior to action.",
    )
    is_done: bool = Field(
        default=False,
        description="Flag signaling whether the agent considers the overall episode complete.",
    )
    final_response: Optional[str] = Field(
        default=None,
        description="Natural language dialogue message targeted to the caller or user.",
    )
    cost_usd: float = Field(
        default=0.0,
        description="Estimated or observed LLM query cost in USD for this action.",
    )
    prompt_tokens: int = Field(
        default=0,
        description="Count of prompt tokens consumed in generating this action.",
    )
    completion_tokens: int = Field(
        default=0,
        description="Count of completion tokens consumed in generating this action.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Diagnostic metadata, routing decisions, and telemetry records.",
    )
    content: Optional[str] = Field(
        default=None,
        description="Text content emitted for turn-record compatibility in simulation orchestrators.",
    )
    tool_calls: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of tool calls formatted for simulation orchestrators.",
    )

    def model_post_init(self, __context: Any) -> None:
        """Synchronize compatibility fields for seamless orchestrator integration."""
        if self.content is None:
            self.content = self.final_response or self.thought or ""
        if not self.tool_calls and self.tool_name:
            self.tool_calls = [{"tool_name": self.tool_name, "arguments": self.arguments}]


class MockProvider:
    """Mock LLM provider for deterministic offline testing and scripted evaluation."""

    def __init__(
        self,
        canned_responses: Optional[List[Union[str, Dict[str, Any]]]] = None,
        handler: Optional[
            Callable[[List[Dict[str, Any]], Optional[List[Dict[str, Any]]]], Union[str, Dict[str, Any]]]
        ] = None,
    ) -> None:
        self.canned_responses = list(canned_responses) if canned_responses else []
        self.handler = handler
        self.call_count = 0
        self.history: List[Dict[str, Any]] = []

    def reset(self) -> None:
        """Reset internal call counts and recorded histories."""
        self.call_count = 0
        self.history.clear()

    def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Produce a canned or dynamically evaluated completion response."""
        self.history.append({"messages": messages, "tools": tools, "kwargs": kwargs})
        self.call_count += 1

        if self.handler is not None:
            raw_res = self.handler(messages, tools)
            if isinstance(raw_res, dict):
                return raw_res
            return {
                "choices": [{"message": {"role": "assistant", "content": str(raw_res)}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            }

        if self.canned_responses:
            idx = min(self.call_count - 1, len(self.canned_responses) - 1)
            item = self.canned_responses[idx]
            if isinstance(item, dict):
                return item
            return {
                "choices": [{"message": {"role": "assistant", "content": str(item)}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            }

        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Awaiting customer request. How can Permian Chemical assist you today?",
                    }
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        }


class AgentInterface(ABC):
    """Abstract base class establishing the contract for all systems under test."""

    name: str
    system_prompt: str

    @abstractmethod
    def step(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[ToolDefinition]] = None,
        context: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AgentAction:
        """Execute a single reasoning step or turn given conversation history and tools."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset all internal state, memory buffers, and scratchpads for a clean episode."""
        ...

    def act(
        self,
        history: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> AgentAction:
        """Alias for step adhering to alternative caller conventions."""
        return self.step(messages=history, history=history, context=context, **kwargs)

    def __call__(
        self,
        history: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> AgentAction:
        """Callable interface matching LangGraph node and simulation runner conventions."""
        return self.step(messages=history, history=history, context=context, **kwargs)


__all__ = ["AgentAction", "AgentInterface", "MockProvider"]
