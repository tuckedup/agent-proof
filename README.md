# agentproof: Agent Reliability Lab

> An empirical reliability laboratory for tool-calling agents in critical logistics workflows. Modeled on tau2-bench (dual-control environment), MAST (multi-agent systems failure taxonomy), and EvalGen (validating LLM judges).

## 1. Headline Result
*Note: Full evaluation runs across all agents and ablation configurations are underway. Results below reflect benchmark metrics measured in committed reports.*
- Baseline Pass^1: not yet measured
- Oracle Plan Pass^1: not yet measured
- Dual-Control Performance Gap: not yet measured
- Policy Violation Rate: not yet measured
- Top Failure Category: not yet measured

## 2. Client Brief
The order desk of an oilfield chemical supplier takes requests from field crews by chat and email: check stock, place or change orders, schedule deliveries to well sites, and enforce hazmat and credit rules. Leadership wants an agent to handle routine requests and hand off the rest, but will not approve it without evidence that it follows policy and degrades safely. The deliverable is the agent and the evidence.

*Disclaimer: The domain, company context, and customer records used in this benchmark are entirely synthetic for research and evaluation purposes.*

## 3. Architecture Overview
- **ChemDesk Domain**: Isolated SQLite database per episode seeded with 40 accounts, 120 well sites, 60 chemicals, historical orders, DOT hazard classifications, truck compatibility rules, and credit limits.
- **Dual-Control Simulation**: Modeled after tau2-bench. The agent operates desk tools (`find_account`, `create_order`, `book_delivery`, etc.) while a seeded LLM user simulator operates site-side tools (`check_site_access`, `report_receiver_on_site`, `confirm_delivery_window`).
- **Telemetry & Tracing**: OpenTelemetry GenAI semantic conventions instrumentation exporting to Arize Phoenix.
- **MAST Taxonomy**: Systematic error labeling mapped to the 14 multi-agent failure modes across Specification, Alignment, and Verification.
- **LLM Judge Validation**: Binary judges per failure class validated on held-out test sets with at least 40 examples per class.

## 4. Quickstart
```powershell
# Install dependencies
uv sync --extra dev

# Run unit tests
uv run poe test

# Launch development environment (Arize Phoenix and web UI)
uv run poe dev

# Run simulation benchmark
uv run poe sim --agent react_baseline --mode dual --tasks all --trials 4
```
