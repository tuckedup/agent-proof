"""Initial scaffold tests for agentproof."""

import sys
from pathlib import Path


def test_python_version() -> None:
    """Verify supported Python version."""
    assert sys.version_info >= (3, 11)


def test_project_structure() -> None:
    """Verify expected directory layout exists."""
    root = Path(__file__).resolve().parent.parent
    expected_dirs = [
        "agentproof",
        "agentproof/domain",
        "agentproof/tools",
        "agentproof/tasks",
        "agentproof/sim",
        "agentproof/agents",
        "agentproof/tracing",
        "agentproof/providers",
        "agentproof/scoring",
        "agentproof/judges",
        "agentproof/mast",
        "agentproof/experiments",
        "domain/seed",
        "tasks",
        "web",
        "reports",
        "docs",
        "tests",
    ]
    for rel_dir in expected_dirs:
        assert (root / rel_dir).is_dir(), f"Missing required directory: {rel_dir}"


def test_core_docs_exist() -> None:
    """Verify required governance documents exist."""
    root = Path(__file__).resolve().parent.parent
    for doc in ["PLAN.md", "CLAUDE.md", "AGENTS.md", "DECISIONS.md", "BLOCKERS.md", "README.md"]:
        assert (root / doc).is_file(), f"Missing required documentation: {doc}"
