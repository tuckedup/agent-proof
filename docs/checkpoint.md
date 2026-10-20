# Development checkpoint

This checkpoint reflects repository commit `0260e3f` and records work already present in the source tree. The domain and all customer and operational data are synthetic.

## Implemented scope

- ChemDesk domain models, seeded SQLite databases, and desk and site-side tools for account, order, inventory, credit, compatibility, escalation, and delivery operations.
- Task models and generator, with a committed task set.
- Scripted user simulator, leak detector, and episode orchestrator with dual, no-user, and oracle-plan modes.
- React baseline, LangGraph plan-execute and supervisor agents, plus mock LLM provider, cache, and cost tracking.
- A static Next.js page and minimal smoke tests.

These components have unit and integration test coverage. The full Python test suite passed in the validation environment described below.

## Remaining milestones

- The CLI commands `dev`, `sim`, `report`, `validate-judges`, and `gate` only print the command name. They do not launch services, execute benchmark runs, generate reports, validate judges, or enforce a regression gate.
- OpenTelemetry/Phoenix trace integration, scoring and report generation, MAST taxonomy data, annotation UI, judge validation, and empirical experiments remain incomplete.
- The web page is static; it does not load trajectories, traces, labels, or benchmark results.
- Benchmark metrics and evaluation results are not yet measured. There are no committed benchmark reports.

## Validation at checkpoint preparation

- Python: the supplied environment reports Python 3.12.7.
- Ruff: `python -m ruff check agentproof tests` passed.
- Pytest: all 62 tests passed with pytest plugin autoload disabled, `pytest_asyncio` explicitly enabled, and temporary files directed to the workspace. The tests use mock providers; no live model calls were made. `test_mock_provider_scenarios` was corrected to read both dictionary and LiteLLM object response shapes while retaining its tool-call assertions.
- Mypy: failed with 15 errors across 9 files. Reported issues include type mismatches in the domain/task/agent code and missing LiteLLM attributes in the supplied environment.
- Frontend: `node --test tests/smoke.test.js` passed (2 tests). Those tests contain basic assertions and do not exercise the page or application behavior. The package lint command was not run because `web/node_modules` is absent.
- No live or paid model APIs were called.

The `reports/` directory is reserved for future run artifacts. It currently contains no benchmark results.
