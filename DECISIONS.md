# Architectural & Design Decisions

### ADR 001: Isolated SQLite Database per Episode
- **Date**: 2026-10-14
- **Context**: The simulation evaluates agent tools that mutate state (orders, inventory, credits, bookings). If episodes share a database or rollbacks are incomplete, cross-episode state contamination creates nondeterministic evaluation results.
- **Decision**: Every episode creates a distinct in-memory or temporary SQLite database file initialized from canonical JSON seed files (`domain/seed/*.json`). When an episode completes, the episode database is cleanly torn down after capturing the final state diff.
- **Consequences**: Deterministic, isolated evaluation runs with zero cross-talk between concurrent trials.

### ADR 002: Dual-Control Environment Architecture
- **Date**: 2026-10-14
- **Context**: Real-world field interactions involve constraints on the customer side that the order desk agent cannot directly inspect (such as road mud conditions after rain, whether a certified hazmat receiver is currently on site).
- **Decision**: Modeled after tau2-bench dual-control architecture. The agent has 14 desk tools (`find_account`, `create_order`, etc.), while the user simulator has 3 site-side tools (`check_site_access`, `confirm_delivery_window`, `report_receiver_on_site`). The orchestrator strictly checks tool requestor permissions and prevents the agent from invoking user tools directly in `dual` mode.
- **Consequences**: Enables measuring the "dual gap" (performance degradation when information must be elicited interactively versus when the agent can inspect everything directly in `no_user` mode).

### ADR 003: Model Independence via LiteLLM and Environment Variables
- **Date**: 2026-10-14
- **Context**: Evaluation must work across Anthropic, OpenAI, or local models without modifying agent or evaluator code.
- **Decision**: All model invocations go through LiteLLM with model names configured exclusively via environment variables: `AGENT_MODEL`, `USER_SIM_MODEL`, `JUDGE_MODEL`.
- **Consequences**: Clean separation of agent logic and model selection. Zero vendor lock-in.

### ADR 004: Multi-Agent Failure Taxonomy (MAST) Integration
- **Date**: 2026-10-14
- **Context**: Failure analysis must classify multi-agent and tool-calling errors systematically rather than using ad-hoc categories.
- **Decision**: Adopt the 14 failure modes defined by MAST across Specification Issues, Inter-Agent Misalignment, and Task Verification. Store canonical taxonomy definitions in `agentproof/mast/taxonomy.json` and map error traces to these modes in the annotation UI and judge pipeline.
- **Consequences**: Standardized error analysis directly comparable to published MAS research.

### ADR 005: Zero API Cost CI with Recorded Mock Provider
- **Date**: 2026-10-14
- **Context**: GitHub Actions CI must never fail due to missing API keys or incur unpredictable cloud costs.
- **Decision**: Implement `MockProvider` which replays recorded LLM responses hashed by normalized message history and tool schemas. CI runs all tests and a 20-task smoke simulation strictly through `MockProvider`.
- **Consequences**: Fast, 100% reproducible, zero-cost CI pipeline.
