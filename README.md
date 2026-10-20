# agentproof: Agent Reliability Lab (in progress)

> An empirical reliability laboratory for tool-calling agents in critical logistics workflows. Modeled on tau2-bench (dual-control environment), MAST (multi-agent systems failure taxonomy), and EvalGen (validating LLM judges).

## 1. Headline Result
*Benchmark execution and reporting are not implemented in this checkpoint. All headline metrics remain unmeasured.*
- Baseline Pass^1: not yet measured
- Oracle Plan Pass^1: not yet measured
- Dual-Control Performance Gap: not yet measured
- Policy Violation Rate: not yet measured
- Top Failure Category: not yet measured

## 2. Client Brief
The order desk of an oilfield chemical supplier takes requests from field crews by chat and email: check stock, place or change orders, schedule deliveries to well sites, and enforce hazmat and credit rules. Leadership wants an agent to handle routine requests and hand off the rest, but will not approve it without evidence that it follows policy and degrades safely. The deliverable is the agent and the evidence.

*Disclaimer: The domain, company context, and customer records used in this benchmark are entirely synthetic for research and evaluation purposes.*

## 3. Architecture Overview

This section describes the target architecture. The repository implements only part of it so far; tracing, the MAST taxonomy and annotation workflow, judge validation, scoring, and benchmark reporting remain planned work. See [the development checkpoint](docs/checkpoint.md) for the implemented scope.

- **ChemDesk Domain**: Isolated SQLite database per episode seeded with synthetic accounts, well sites, chemicals, orders, DOT hazard classifications, truck compatibility rules, and credit limits.
- **Dual-Control Simulation**: Modeled after tau2-bench. The agent operates desk tools (`find_account`, `create_order`, `book_delivery`, etc.) while a seeded LLM user simulator operates site-side tools (`check_site_access`, `report_receiver_on_site`, `confirm_delivery_window`).
- **Telemetry & Tracing**: OpenTelemetry GenAI semantic conventions instrumentation exporting to Arize Phoenix.
- **MAST Taxonomy**: Planned systematic error labeling across Specification, Alignment, and Verification categories.
- **LLM Judge Validation**: Planned binary judges with held-out validation sets; the pipeline and results are not implemented.

## 4. Quickstart

The repository currently contains the Python domain, tool, task, simulation, agent, and mock-provider components described in [the checkpoint](docs/checkpoint.md). The benchmark CLI and end-to-end evaluation workflow are not implemented yet. The `dev`, `sim`, `report`, `validate-judges`, and `gate` CLI commands are placeholders that only print the selected command; they do not start services or run evaluations.

```powershell
# Install dependencies
uv sync --locked --extra dev

# Run the implemented Python checks
uv run poe test
uv run poe lint
# Currently reports existing type errors; see docs/checkpoint.md
uv run poe typecheck

# Run the frontend scaffold's minimal Node.js smoke tests
cd web
node --test tests/smoke.test.js
```

Benchmark results are not yet measured. The web page is a static scaffold and does not display run data.
