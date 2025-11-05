# NQ Trading Bot Technical Review

## Executive Summary
The project combines a Flask webhook receiver, a Dash-based operations dashboard, and two Python strategies to implement a near-real-time NQ futures paper-trading loop. The solution covers data persistence, logging, position tracking, and strategy orchestration, but several incomplete modules and operational gaps keep it from running reliably. Unresolved market-context hooks, inconsistent time handling, and CSV-bound state management produce runtime failures and scalability limits. Addressing the outlined gaps will materially improve stability and make a live-trading rollout safer.

## Architecture Overview
- **Ingestion & Control Plane:** A Flask application receives TradingView alerts, persists every bar, runs the strategy engine, and coordinates the position manager before replying to the webhook caller.【F:main.py†L26-L115】
- **Dashboard:** The Dash app (mounted on the same Flask server) visualizes metrics, recent signals, and price action with indicators added on the fly for each refresh.【F:dashboard/trading_dashboard.py†L1-L209】【F:dashboard/trading_dashboard.py†L210-L318】
- **Persistence:** `DataHandler` manages CSV-backed storage for bars, signals, and trades, and derives aggregate performance metrics from the on-disk files.【F:utils/data_handler.py†L15-L104】【F:utils/data_handler.py†L224-L286】
- **Risk & Execution:** `PositionManager` enforces position limits, tracks P&L, and records trade lifecycle events; strategy outputs are encoded as signal dictionaries consumed by this layer.【F:position_manager.py†L12-L204】
- **Strategies:** The currently wired strategies are Opening Range Breakout and Mean Reversion. They depend on shared indicator utilities and configuration weights to populate the signal payload that downstream components expect.【F:strategies/opening_range.py†L14-L201】【F:strategies/mean_reversion.py†L13-L150】
- **Market Context:** `MarketLevels` is intended to calculate prior-day levels, overnight ranges, and round-number zones, making this context available to strategies during evaluation.【F:utils/market_levels.py†L11-L189】

## Strengths
- **Clear orchestration path:** The webhook handler persists data, updates risk, and invokes strategies in a deterministic order, making it easy to reason about the event loop.【F:main.py†L63-L108】
- **Rich logging surface:** Dedicated rotating loggers for system, trades, strategy, ML, and webhook flows provide good observability and future auditing capability.【F:utils/logger.py†L10-L83】
- **Config-driven strategy tuning:** `config.py` centralizes strategy toggles and parameters, enabling quick experimentation without code edits.【F:config.py†L60-L101】
- **Dashboard-first operations:** The Dash interface renders trades, signals, and indicators in near real time, supporting monitoring without diving into logs.【F:dashboard/trading_dashboard.py†L36-L318】
- **Well-structured indicator utilities:** Shared helper modules expose ATR, EMA, VWAP, and range-calculation utilities, reducing duplication across strategies and simplifying future additions.【F:utils/indicators.py†L12-L189】

## Key Risks & Gaps
1. **Incomplete strategy-engine integration** – The market-levels block inside `process_new_bar` still contains placeholder comments (`# ADD THIS`) and references to an undefined `should_log_levels` flag, so the module will raise `NameError` before any strategy can run.【F:strategy_engine.py†L103-L112】 Similarly, the ORB `generate_signal` method includes pseudo-code for PDH proximity checks that references `signal_direction` and other variables that do not exist, which will crash the strategy on the first execution.【F:strategies/opening_range.py†L98-L123】
2. **Timezone mismatch in market context** – `MarketLevels.update_levels` creates timezone-aware timestamps for the overnight session boundaries, but bar indices coming from `DataHandler.get_latest_bars` are timezone-naive, producing `TypeError: Cannot compare tz-naive and tz-aware` during comparisons in a live run.【F:utils/data_handler.py†L95-L104】【F:utils/market_levels.py†L58-L68】 This prevents the overnight-range logic from ever completing.
3. **CSV-bound persistence limits throughput and durability** – Every webhook call reads and rewrites entire CSV files for bars, signals, and trades.【F:utils/data_handler.py†L63-L83】【F:utils/data_handler.py†L141-L179】 This will become a bottleneck at a one-minute cadence, increases the risk of partial writes on crash, and complicates running multiple processes.
4. **Risk calculations hard-code assumptions** – Position P&L subtracts a flat $5 commission, ignoring the more realistic cost constants defined in `config.py`, and assumes one contract per trade without respecting `MAX_POSITION_SIZE`.【F:position_manager.py†L96-L101】【F:position_manager.py†L132-L169】 It also measures daily loss in absolute P&L without session boundaries, so hitting the loss limit during overnight trading could lock the bot until the host’s local midnight.
5. **Duplicate configuration sources** – `StrategyEngine` embeds a static list of high-impact event dates instead of reusing `HIGH_IMPACT_DATES` from `config.py`, increasing drift risk between modules.【F:strategy_engine.py†L28-L36】【F:config.py†L157-L176】
6. **Testing coverage is absent** – There are no automated tests validating strategy outputs, persistence behaviors, or Flask endpoints, leaving regressions undetected.
7. **Deployment playbooks are missing** – The repository does not document how to run the bot in paper versus live modes, manage environment variables, or rotate credentials, leaving operations staff without a runbook.【F:README.md†L1-L68】

## Recommendations
1. **Finalize the market-context integration**
   - Remove placeholder comments, implement a real `should_log_levels` flag (or drop the logging), and thread computed levels/context into strategy calls. For ORB, rewrite the PDH proximity guard using existing local variables to avoid `NameError` and ensure the method still returns a signal dictionary in the happy path.【F:strategy_engine.py†L103-L135】【F:strategies/opening_range.py†L98-L196】
2. **Normalize timestamps end-to-end**
   - Make `DataHandler` persist ISO timestamps with timezone information or localize them when loading so that `MarketLevels` comparisons succeed. Alternatively, switch `MarketLevels` to operate on naive datetimes consistently, but pick a single convention and enforce it before slicing overnight sessions.【F:utils/data_handler.py†L95-L104】【F:utils/market_levels.py†L58-L83】
3. **Upgrade persistence before scaling**
   - Replace per-request CSV rewrites with an append-only store (SQLite, DuckDB, Timescale). If CSV must remain, buffer writes in memory and flush periodically to reduce I/O and race conditions.【F:utils/data_handler.py†L63-L83】【F:utils/data_handler.py†L141-L179】
4. **Align risk management with configuration**
   - Pull commission and slippage values from `config.py`, calculate trade size from `MAX_POSITION_SIZE` or signal metadata, and introduce a session-aware loss reset (e.g., reset at 16:00 ET) to reflect futures trading rules.【F:position_manager.py†L96-L169】【F:config.py†L29-L47】
5. **Consolidate shared business calendars**
   - Source the “do not trade” calendar from `config.py` inside the engine so that business users can maintain one list, and consider integrating an external economic-calendar feed for automation.【F:strategy_engine.py†L28-L36】【F:config.py†L157-L176】
6. **Add regression tests and linting**
   - Start with unit tests for each strategy’s signal generation on canned datasets and integration tests for the webhook route to ensure malformed payloads are rejected gracefully. Wire the suite into CI to gate future changes.
7. **Document deployment paths**
   - Provide a minimal operations runbook covering environment setup, credential storage, cloud deployment targets, and monitoring hooks to shorten time-to-paper/live trading adoption.【F:README.md†L1-L68】

## Issue Severity Matrix
| Area | Impact | Urgency | Notes |
| --- | --- | --- | --- |
| Strategy engine placeholders | High | Immediate | Blocks execution with `NameError` when the first bar arrives. |
| Timezone handling | High | Immediate | Prevents market-level calculations and can corrupt risk windows. |
| Persistence layer | Medium | Near-term | Limits throughput and resiliency; acceptable only for very low-frequency paper runs. |
| Risk configuration drift | Medium | Near-term | Causes inaccurate P&L and position sizing, undermining trust in results. |
| Deployment documentation | Medium | Near-term | Slows onboarding and increases operational mistakes. |
| Missing tests | High | Ongoing | Leaves regressions undetected; hinders refactoring confidence. |

## Operational Readiness Checklist
- [ ] Fix runtime errors in strategy engine and ORB strategy
- [ ] Decide on a canonical timezone and refactor data ingestion accordingly
- [ ] Migrate away from CSV or implement buffered writes
- [ ] Expand risk controls to respect configurable costs and size limits
- [ ] Add automated tests (unit + integration)
- [ ] Document deployment steps for live/paper trading
- [ ] Establish monitoring/alert thresholds for webhook latency and trading errors

## Testing Status
No automated tests are defined in the repository. Manual validation is currently required for every change.
