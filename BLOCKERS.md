# Project Blockers Log

This document records any blockers, missing external dependencies, or environment limitations encountered during the build.

| Date | Item | Status | Mitigation / Resolution |
| :--- | :--- | :--- | :--- |
| 2026-10-14 | Docker Desktop daemon not active | Mitigated | Arize Phoenix supports local execution in Python (`phoenix.server.start`) or containerized OTLP. Local Python runner and mock test exporters will be primary for unit testing and CI; docker-compose provided for deployment. |
| 2026-10-14 | External LLM API Keys (OpenAI / Anthropic) not present in local environment | Mitigated | `MockProvider` implemented in Milestone M3 with recorded trajectories allows 100% test coverage, smoke simulations, and CI execution without external paid APIs. |
