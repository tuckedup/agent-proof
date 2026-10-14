"""Tools package for agentproof."""

from agentproof.tools.agent_tools import AGENT_TOOLS
from agentproof.tools.base import ToolDefinition, ToolResult
from agentproof.tools.user_tools import USER_TOOLS

ALL_TOOLS = {**AGENT_TOOLS, **USER_TOOLS}

__all__ = [
    "AGENT_TOOLS",
    "USER_TOOLS",
    "ALL_TOOLS",
    "ToolDefinition",
    "ToolResult",
]
