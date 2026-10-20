# Development checkpoint

This checkpoint records CI repairs made on top of repository commit `df0dff8`. The domain and all customer and operational data are synthetic.

## Implemented scope

- ChemDesk domain models, seeded SQLite databases, and desk and site-side tools for account, order, inventory, credit, compatibility, escalation, and delivery operations.
- Task models and generator, with a committed task set.
- Scripted user simulator, leak detector, and episode orchestrator with dual, no-user, and oracle-plan modes.
- React baseline, LangGraph plan-execute and supervisor agents, plus mock LLM provider, cache, and cost tracking.
- A static Next.js page and minimal smoke tests.

## CI repair scope

- Added explicit collection annotations in the deterministic domain generator and resolved a reused loop-variable type conflict.
- Kept hard-task `forbidden_actions` and `oracle_plan` on the parent `Task`, where evaluator constraints belong. A regression test confirms they are retained without being exposed in `UserInstructions`.
- Made the plan-execute credit policy fail closed when credit verification fails, returns no payload, or omits a sufficient-credit result. A regression test covers the missing-payload case.
- Made the mock provider tolerate a partial or absent LiteLLM import without assigning `None` to an imported class type.
- Added the Next.js ESLint configuration so `pnpm lint` does not enter interactive setup in CI.

These components have unit and integration test coverage. The full Python test suite passed in the validation environment described below.

## Remaining milestones

- The CLI commands `dev`, `sim`, `report`, `validate-judges`, and `gate` only print the command name. They do not launch services, execute benchmark runs, generate reports, validate judges, or enforce a regression gate.
- OpenTelemetry/Phoenix trace integration, scoring and report generation, MAST taxonomy data, annotation UI, judge validation, and empirical experiments remain incomplete.
- The web page is static; it does not load trajectories, traces, labels, or benchmark results.
- Benchmark metrics and evaluation results are not yet measured. There are no committed benchmark reports.

## Validation for CI repairs

- Ruff: `D:\FDE\agentproof\.venv\Scripts\python.exe -m ruff check agentproof tests` passed.
- Mypy: `D:\FDE\agentproof\.venv\Scripts\python.exe -m mypy agentproof --ignore-missing-imports` reports six unrelated LiteLLM missing-attribute errors in `cost_tracker.py`, `cache.py`, `litellm_client.py`, `user_sim.py`, and `react.py`. The supplied environment has a partial LiteLLM installation, as documented for this repair; no errors remain in the four CI-repair source files. The command used `PYTHONPATH` set to this clone, `PYTHONDONTWRITEBYTECODE=1`, and workspace-local temporary paths.
- Final pytest: all 64 tests passed in 4.06 seconds with pytest plugin autoload disabled, `pytest_asyncio` explicitly enabled, `LITELLM_LOCAL_MODEL_COST_MAP=True`, and temporary files directed to the workspace.
- Frontend: `node --test tests/smoke.test.js` passed (2 tests). `pnpm lint` was not run locally because `web/node_modules` is absent; no dependencies were installed or upgraded.
- No live or paid model APIs were called.

A new clean GitHub CI run must verify the repaired type and frontend lint checks.

The `reports/` directory is reserved for future run artifacts. It currently contains no benchmark results.
