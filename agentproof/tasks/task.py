"""Task definition and models for agentproof simulation benchmarks."""

from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class Task(BaseModel):
    """Specification of an evaluation task for ChemDesk order desk."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    task_id: str = Field(description="Unique task identifier, e.g. 'TASK-001'.")
    goal: str = Field(description="Primary objective or instructions for the task.")
    category: Optional[str] = Field(default=None, description="Task category, e.g. 'order_creation', 'hazmat', 'credit_limit'.")
    initial_user_message: Optional[str] = Field(
        default=None,
        description="Initial statement or query spoken by the user persona."
    )
    initial_db_setup: Optional[Union[List[str], str, Dict[str, Any], Callable[[Any], None]]] = Field(
        default=None,
        description="SQL statements, data dictionary, or callable for scenario setup."
    )
    gold_plan: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Reference sequence of actions or tool calls for oracle planning."
    )
    user_persona: Optional[str] = Field(
        default=None,
        description="Persona and background instructions for the user simulator."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary additional metadata for evaluation and tagging."
    )
