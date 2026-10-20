# AGENTS.md: agentproof Developer Guide

## Stack and Architecture
- Python: 3.12+ managed with `uv`
- Package & Task Runner: `uv` and `poethepoet`
- Frontend: Node.js 20+, `pnpm`, Next.js App Router, TypeScript, Tailwind CSS, Lucide React
- LLM Provider Interface: `litellm` (supporting Anthropic, OpenAI, local models)
- Multi-Agent Orchestration: `langgraph`
- Observability: OpenTelemetry Python SDK with GenAI semantic conventions, Arize Phoenix
- Database: SQLite (isolated, per-episode database instances seeded deterministically)

## Environment Variables
- `AGENT_MODEL`: Model string for the agent under test (e.g. `openai/gpt-4o-mini`, `anthropic/claude-3-5-sonnet-20241022`)
- `USER_SIM_MODEL`: Model string for the user simulator
- `JUDGE_MODEL`: Model string for LLM-as-a-judge
- `MAX_RUN_USD`: Hard budget ceiling per simulation run (default: 5.0)
- `PHOENIX_COLLECTOR_ENDPOINT`: OTLP endpoint for Arize Phoenix (default: `http://localhost:6006/v1/traces`)
- `AGENTPROOF_RECORD_MOCKS`: Set to `1` when recording real API calls for mock replay

## Key Commands
```powershell
# Python environment & dependencies
uv sync --locked --extra dev

# Implemented Poe checks
uv run poe test             # Run Python unit and integration tests
uv run poe lint             # Run Ruff checks
uv run poe typecheck        # Run Mypy

# Minimal frontend scaffold test
cd web
node --test tests/smoke.test.js
```

The CLI commands `dev`, `sim`, `report`, `validate-judges`, and `gate` are placeholders that only print the selected command. Do not treat them as implemented workflows.

## Coding Conventions
1. Target Windows 11 PowerShell and Linux. Never author bash-only scripts or use Unix-specific assumptions.
2. Maintain strong typing: Pydantic v2 models for all structured payloads, tool definitions, and task specifications.
3. Conventional Commits format for git history: `feat:`, `fix:`, `test:`, `docs:`, `chore:`.
4. Keep functions modular and clean. Preserve documentation integrity and docstrings.
5. Every tool function must specify Pydantic argument schemas and clear docstrings explaining purpose, constraints, and side effects.
6. Write prose without em-dashes throughout all documentation and user-facing text.

## Honesty Rules (Section 10 - Non-Negotiable)
1. Never generate human labels with an LLM and present them as human. LLM-proposed labels must always be tagged with `source=llm` and must never be used as ground truth for judge validation.
2. Every reported number comes from a committed file in `reports/`. If something is not measured yet, write "not yet measured".
3. The domain and data are synthetic. State this clearly in the README and case study. Do not describe this as a deployment at any real company.
4. Write prose without em-dashes.
