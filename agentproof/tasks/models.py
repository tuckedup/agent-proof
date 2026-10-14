"""Pydantic v2 data models for benchmark tasks and user personas.

These models define evaluation tasks, caller personas, dual-control simulator
instructions, and expected database assertion criteria for ChemDesk.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class TaskIntent(str, Enum):
    """Supported atomic operational intents."""

    VERIFY = "verify"
    CHECK_STOCK = "check_stock"
    PLACE_ORDER = "place_order"
    MODIFY_ORDER = "modify_order"
    CANCEL_ORDER = "cancel_order"
    SCHEDULE_DELIVERY = "schedule_delivery"
    HAZMAT_CHECK = "hazmat_check"
    CREDIT_CHECK = "credit_check"
    ESCALATE = "escalate"


class Difficulty(str, Enum):
    """Task complexity rating."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Category(str, Enum):
    """Task category reflecting compositional depth and challenge type."""

    ATOMIC = "atomic"
    COMPOSITIONAL_2STEP = "compositional_2step"
    COMPOSITIONAL_3STEP = "compositional_3step"
    COMPOSITIONAL_4STEP = "compositional_4step"
    HARD = "hard"


class DBAssertion(BaseModel):
    """Database state verification check evaluated against SQLite after task completion."""

    table: str = Field(description="Name of the SQLite database table to inspect.")
    filter: Dict[str, Any] = Field(
        default_factory=dict,
        description="Column-value conditions used to select target rows.",
    )
    expected_values: Dict[str, Any] = Field(
        default_factory=dict,
        description="Expected column values on the matched row.",
    )
    expected_count: Optional[int] = Field(
        default=None,
        description="Expected number of rows matching the filter condition.",
    )
    query: Optional[str] = Field(
        default=None,
        description="Custom SQL query for complex checks.",
    )
    description: Optional[str] = Field(
        default=None,
        description="Human readable explanation of what this assertion verifies.",
    )

    def check(self, db: Any) -> Tuple[bool, Optional[str]]:
        """Evaluate this assertion against the given DatabaseManager instance."""
        if self.query:
            rows = db.execute_query(self.query)
            if self.expected_count is not None and len(rows) != self.expected_count:
                return (
                    False,
                    f"Query returned {len(rows)} rows; expected {self.expected_count}",
                )
            if self.expected_values and rows:
                first_row = dict(rows[0])
                for col, expected in self.expected_values.items():
                    actual = first_row.get(col)
                    if actual != expected:
                        return (
                            False,
                            f"Column '{col}' expected '{expected}', found '{actual}'",
                        )
            return True, None

        clauses = []
        params = []
        for col, val in self.filter.items():
            clauses.append(f"{col} = ?")
            params.append(val)

        where_str = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM {self.table}{where_str}"
        rows = db.execute_query(sql, tuple(params))

        if self.expected_count is not None and len(rows) != self.expected_count:
            return (
                False,
                f"Table '{self.table}' had {len(rows)} matching rows; expected {self.expected_count}",
            )

        if self.expected_values:
            if not rows:
                return (
                    False,
                    f"Table '{self.table}' has no rows matching filter {self.filter}",
                )
            first_row = dict(rows[0])
            for col, expected in self.expected_values.items():
                actual = first_row.get(col)
                if actual != expected:
                    return (
                        False,
                        f"Table '{self.table}' column '{col}' expected '{expected}', found '{actual}'",
                    )

        return True, None


class UserPersona(BaseModel):
    """Profile of the field caller being simulated."""

    name: str = Field(description="Caller full name.")
    role: str = Field(description="Caller job title, such as Toolpusher or Rig Lead.")
    account_id: str = Field(description="Associated customer account identifier.")
    site_code: str = Field(description="Associated well site code.")
    phone: Optional[str] = Field(default=None, description="Contact phone number.")
    email: Optional[str] = Field(default=None, description="Contact email address.")
    has_hazmat_cert: bool = Field(
        default=False,
        description="Whether the caller holds a valid DOT hazmat certification.",
    )
    cert_expiration: Optional[str] = Field(
        default=None,
        description="Expiration date of hazmat certification if held.",
    )
    personality_traits: List[str] = Field(
        default_factory=list,
        description="Behavioral traits like pushy, cooperative, or hurried.",
    )


class UserInstructions(BaseModel):
    """Complete instructions guiding the user simulator during an interaction episode."""

    caller_name: str = Field(description="Name the simulator must introduce themselves as.")
    account_id: str = Field(description="Customer account ID.")
    site_code: str = Field(description="Destination well site code.")
    goal: str = Field(description="Primary objective the user needs accomplished.")
    initial_utterance: str = Field(description="Opening statement sent by the user.")
    persona: Optional[UserPersona] = Field(
        default=None,
        description="Persona details for role playing consistency.",
    )
    secret_facts: Dict[str, Any] = Field(
        default_factory=dict,
        description="Facts known to the user that must not be revealed unless asked.",
    )
    reveal_conditions: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping from topic or tool to conditions for disclosing information.",
    )
    permitted_user_tools: List[str] = Field(
        default_factory=list,
        description="User tools that the simulator is allowed to call during the run.",
    )
    fallback_behavior: Optional[str] = Field(
        default=None,
        description="Action to take when the agent is unhelpful or repetitive.",
    )
    hidden_instructions: Optional[str] = Field(
        default=None,
        description="Internal instructions that must never be leaked to the assistant.",
    )


class Task(BaseModel):
    """Canonical benchmark task representation."""

    id: str = Field(description="Unique task identifier, e.g. TASK-ATOMIC-001.")
    category: Category = Field(description="Task composition category.")
    intents: List[TaskIntent] = Field(description="List of intents exercised in this task.")
    difficulty: Difficulty = Field(description="Task difficulty rating.")
    description: str = Field(description="Summary of task objective and conditions.")
    user_instructions: UserInstructions = Field(
        description="Instructions and persona for user simulation.",
    )
    initial_db_setup: Dict[str, List[Dict[str, Any]]] = Field(
        default_factory=dict,
        description="Table rows to insert or update in the database prior to run.",
    )
    expected_db_assertions: List[DBAssertion] = Field(
        default_factory=list,
        description="Assertions that must hold after the task finishes.",
    )
    required_facts: List[str] = Field(
        default_factory=list,
        description="Key factual claims the agent must convey during the dialogue.",
    )
    forbidden_actions: List[str] = Field(
        default_factory=list,
        description="Tools or actions the agent is strictly prohibited from executing.",
    )
    oracle_plan: List[str] = Field(
        default_factory=list,
        description="Expected optimal sequence of tool calls or conversational milestones.",
    )
