"""Mock and scripted agents for deterministic testing of simulation orchestrator."""

from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class ScriptedAction(BaseModel):
    """Specification of a deterministic action taken by a scripted agent."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    content: str = Field(
        default="",
        description="Text content emitted by the agent for this turn."
    )
    tool_calls: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of tool calls to request, each containing name/tool_name and arguments."
    )
    cost_usd: float = Field(
        default=0.0,
        description="Simulated LLM cost in USD for this action."
    )
    prompt_tokens: int = Field(
        default=0,
        description="Simulated prompt token consumption."
    )
    completion_tokens: int = Field(
        default=0,
        description="Simulated completion token consumption."
    )
    is_done: bool = Field(
        default=False,
        description="Whether this action signals completion."
    )


class ScriptedAgent:
    """Deterministic agent following a scripted list of actions.

    Used to test orchestrator modes, turn-taking, tool dispatch, and permission barriers.
    """

    def __init__(
        self,
        name: str = "scripted_agent",
        actions: Optional[List[Union[ScriptedAction, Dict[str, Any], str, Callable[[Any], Any]]]] = None,
        default_completion_message: str = "Task finished. ###DONE###",
    ):
        self.name = name
        self.actions = actions or []
        self.current_step = 0
        self.default_completion_message = default_completion_message
        self.received_histories: List[Any] = []

    def reset(self) -> None:
        """Reset agent step index and received message history."""
        self.current_step = 0
        self.received_histories.clear()

    @property
    def is_done(self) -> bool:
        """Check if all scripted actions have been executed."""
        return self.current_step >= len(self.actions)

    def _normalize_action(self, raw_action: Any, history: Any) -> ScriptedAction:
        """Convert various action representations into a standard ScriptedAction."""
        if callable(raw_action):
            raw_action = raw_action(history)

        if isinstance(raw_action, ScriptedAction):
            return raw_action

        if isinstance(raw_action, str):
            return ScriptedAction(content=raw_action)

        if isinstance(raw_action, dict):
            # Normalize 'name' to 'tool_name' or vice versa if needed
            tool_calls = raw_action.get("tool_calls", [])
            normalized_calls = []
            for call in tool_calls:
                call_dict = dict(call)
                if "name" in call_dict and "tool_name" not in call_dict:
                    call_dict["tool_name"] = call_dict["name"]
                normalized_calls.append(call_dict)

            return ScriptedAction(
                content=raw_action.get("content", ""),
                tool_calls=normalized_calls,
                cost_usd=float(raw_action.get("cost_usd", 0.0)),
                prompt_tokens=int(raw_action.get("prompt_tokens", 0)),
                completion_tokens=int(raw_action.get("completion_tokens", 0)),
                is_done=bool(raw_action.get("is_done", False)),
            )

        return ScriptedAction(content=str(raw_action))

    def step(self, history: Any = None, **kwargs: Any) -> ScriptedAction:
        """Execute the next scripted action."""
        self.received_histories.append(history)

        if self.current_step < len(self.actions):
            action_item = self.actions[self.current_step]
            self.current_step += 1
            return self._normalize_action(action_item, history)

        # Fallback when all actions are exhausted
        return ScriptedAction(content=self.default_completion_message, is_done=True)

    def act(self, history: Any = None, **kwargs: Any) -> ScriptedAction:
        """Alias for step() to support diverse agent caller conventions."""
        return self.step(history=history, **kwargs)

    def __call__(self, history: Any = None, **kwargs: Any) -> ScriptedAction:
        """Callable interface matching LangGraph and callable agent standards."""
        return self.step(history=history, **kwargs)


__all__ = ["ScriptedAction", "ScriptedAgent"]
