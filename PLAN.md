# agentproof: Development Plan (M0 to M8)

An agent reliability lab for ChemDesk, an oilfield chemical supplier order desk agent, featuring dual-control simulation, multi-agent failure taxonomy (MAST) error analysis, OpenTelemetry GenAI tracing, and LLM judge validation.

## Milestone Roadmap

### M0: Repository & Environment Scaffold
- [ ] Task 0.1: Initialize Python project using `uv` with `pyproject.toml`, dependencies (`pydantic>=2.7`, `litellm>=1.40`, `langgraph>=0.2`, `opentelemetry-api`, `opentelemetry-sdk`, `phoenix>=4.0`, `poethepoet>=0.27`, `pytest>=8.0`, `pytest-asyncio`, `ruff`, `mypy`).
- [ ] Task 0.2: Configure `poethepoet` tasks in `pyproject.toml` (`dev`, `test`, `lint`, `typecheck`, `sim`, `report`, `judges:validate`, `gate`).
- [ ] Task 0.3: Initialize web workspace with `pnpm` in `web/` (Next.js 14/15 App Router, TypeScript, Tailwind CSS, Lucide icons, Vitest/Playwright).
- [ ] Task 0.4: Set up `docker-compose.yml` for local Arize Phoenix OTLP collector and optional web service.
- [ ] Task 0.5: Set up GitHub Actions CI workflow in `.github/workflows/ci.yml` running lint, types, unit tests, and mock smoke simulation.
- **Acceptance Check**: `uv run poe test` passes (Python test suite) and `pnpm test` passes in `web/`.

### M1: Domain Model, Seed Data & Tools ("ChemDesk")
- [ ] Task 1.1: Author `domain/policy.md` (detailed operating policy covering caller verification, order state machine and modification cutoffs, DOT hazard classes 3 and 8, incompatibility matrix, credit limits and escalation thresholds, delivery window weather restrictions).
- [ ] Task 1.2: Design SQLite schema and isolation manager in `agentproof/domain/` with fresh database copies per episode.
- [ ] Task 1.3: Generate realistic deterministic seed data in `domain/seed/*.json` (40 accounts, 120 sites, 60 products, 300 orders, truck delivery slots, contact certifications, credit ledger) with fixed RNG seed.
- [ ] Task 1.4: Implement 14 agent tools with Pydantic schemas and descriptive docstrings:
  - `find_account`, `verify_caller`, `get_order`, `list_orders`, `check_stock`, `quote_price`, `create_order`, `modify_order`, `cancel_order`, `list_delivery_slots`, `book_delivery`, `check_compatibility`, `check_credit`, `escalate_to_human`.
- [ ] Task 1.5: Implement 3 user dual-control tools:
  - `check_site_access`, `confirm_delivery_window`, `report_receiver_on_site`.
- [ ] Task 1.6: Write comprehensive unit tests for all 17 tools and database state transitions.
- **Acceptance Check**: 100% of tools covered by unit tests; all tests pass; reviewer subagent validates against Section 10 rules.

### M2: Task Generator, User Simulator & Dual-Control Orchestration
- [ ] Task 2.1: Implement compositional task generator in `agentproof/tasks/` combining atomic intents (verify, check stock, order, modify, cancel, schedule, hazmat check, credit check, escalate) across 1 to 4 steps.
- [ ] Task 2.2: Generate and validate at least 150 programmatic tasks plus 20 hand-crafted hard edge-case tasks stored in `tasks/*.json`.
- [ ] Task 2.3: Build user simulator in `agentproof/sim/user_sim.py` role-playing field crew lead, interacting exclusively through user tools, emitting `###DONE###` or `###TRANSFER###`, with leak detection preventing hidden prompt leakage.
- [ ] Task 2.4: Build orchestrator in `agentproof/sim/orchestrator.py` supporting `dual`, `no_user`, and `oracle_plan` modes with configurable turn bounds.
- [ ] Task 2.5: Implement scripted fake agent and user simulator unit tests to verify deterministic execution.
- **Acceptance Check**: Scripted fake agent solves 5 easy tasks deterministically; reviewer subagent validation.

### M3: Agent Implementations & Provider Cost Control
- [ ] Task 3.1: Define common agent interface in `agentproof/agents/base.py`.
- [ ] Task 3.2: Implement `react_baseline` agent (hand-written tool-calling loop using LiteLLM directly without third-party agent framework).
- [ ] Task 3.3: Implement `langgraph_plan_execute` agent (LangGraph workflow with planner, executor, pre-write policy verification node, reflection node, and checkpointing).
- [ ] Task 3.4: Implement `langgraph_supervisor` agent (hierarchical router directing requests between Orders Specialist and Logistics Specialist).
- [ ] Task 3.5: Implement provider layer in `agentproof/providers/` wrapping LiteLLM with timeouts, exponential backoff, cost tracking, `MAX_RUN_USD` ceiling, SQLite disk cache, and `MockProvider` replay system.
- [ ] Task 3.6: Provide unit and integration tests using `MockProvider` to ensure zero API cost during CI.
- **Acceptance Check**: Each agent completes at least one episode; all agents pass mock-provider integration tests; reviewer subagent check.

### M4: OpenTelemetry Tracing & Arize Phoenix Integration
- [ ] Task 4.1: Implement OpenTelemetry GenAI semantic conventions instrumentation in `agentproof/tracing/` (`invoke_agent`, `chat`, `execute_tool` spans with `gen_ai.*` attributes, token counts, model parameters).
- [ ] Task 4.2: Configure OTLP exporter for Phoenix (Docker and local) with fallback to in-memory collector.
- [ ] Task 4.3: Embed trace identifiers and Phoenix URLs in all trajectory output files.
- [ ] Task 4.4: Write unit tests verifying span names and attribute conformance using in-memory tracer.
- **Acceptance Check**: Test asserts span names and attributes using in-memory exporter; Phoenix integration verified.

### M5: Scoring Engine, Metrics & Simulation Runner
- [ ] Task 5.1: Implement programmatic scoring in `agentproof/scoring/` (DB end-state assertions, normalized string matching for required communicated facts, forbidden action detection).
- [ ] Task 5.2: Implement pass^k metric computation (unbiased estimator over n trials).
- [ ] Task 5.3: Build run report generator producing overall and per-category pass^1..4, policy violation rates, escalation precision/recall, latency p50/p95, cost, and mode gap comparisons.
- [ ] Task 5.4: Provide CLI runner: `uv run poe sim --agent <name> --mode dual --tasks all --trials 4 --concurrency 8`.
- **Acceptance Check**: `poe sim` runs on 20 tasks x 4 trials and generates valid report JSON and summary tables.

### M6: Annotation UI, MAST Taxonomy & Judge Validation
- [ ] Task 6.1: Load MAST 14 failure mode definitions into `agentproof/mast/taxonomy.json`.
- [ ] Task 6.2: Build Next.js annotation interface in `web/` (trajectory explorer, conversation timeline, tool call inspector, DB diff viewer, Phoenix trace link).
- [ ] Task 6.3: Implement open coding (free-text failure notes, pass/fail toggling) and axial coding (LLM clustering of notes into editable failure categories with `source=llm`).
- [ ] Task 6.4: Implement MAST failure tagging interface and criterion drift tracking in `docs/criteria-drift.md`.
- [ ] Task 6.5: Implement LLM judges in `agentproof/judges/` with few-shot prompt templates and validation pipeline (`uv run poe judges:validate`) enforcing the 40-example threshold per class and split verification (train/dev/test).
- [ ] Task 6.6: Add Playwright end-to-end test verifying labeling flow and label persistence.
- **Acceptance Check**: Playwright test labels a trajectory and verifies label in DB; judge validation script operational; reviewer subagent check.

### M7: Empirical Experiments & Ablation Studies
- [ ] Task 7.1: Build and run Experiment 1: Mode gap analysis (`dual` vs `no_user` vs `oracle_plan`).
- [ ] Task 7.2: Build and run Experiment 2: Baseline fix ablations (rewritten tool descriptions, confirm-before-write, planning step, structured state notes, combined).
- [ ] Task 7.3: Build and run Experiment 3: Single-agent vs supervisor multi-agent comparison on pass^k and MAST failure distributions.
- [ ] Task 7.4: Generate publication-grade matplotlib charts (pass^k curves, failure category distributions, cost vs pass^1).
- [ ] Task 7.5: Save all experiment runs and reports into `reports/`.
- **Acceptance Check**: Experiments execute within `MAX_RUN_USD` ceiling; reports and charts generated; human label integrity verified (Section 10).

### M8: Documentation, Case Study & Delivery
- [ ] Task 8.1: Write top-level `README.md` following Section 7 specification (real numbers from reports, client brief, screenshots, architecture diagram, results tables, quickstart, evals guide, limitations, related work).
- [ ] Task 8.2: Produce system architecture diagram.
- [ ] Task 8.3: Author `docs/case-study.md` detailing the problem, methodology, findings, failure patterns, and future roadmap.
- [ ] Task 8.4: Author `docs/postmortem-<topic>.md` documenting an authentic technical challenge solved during implementation.
- [ ] Task 8.5: Provide executable demonstration script and walkthrough guide.
- **Acceptance Check**: Final reviewer subagent audits acceptance criteria, automated tests, and honesty rules.
