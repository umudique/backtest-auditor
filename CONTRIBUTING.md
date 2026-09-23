# Contributing to Backtest Auditor

## Commit Style

Use conventional commits:

| Prefix | When to use |
|---|---|
| `feat:` | new capability within the auditing scope |
| `fix:` | bug correction |
| `test:` | adding or correcting tests |
| `refactor:` | structural change with no behavior change |
| `docs:` | documentation only |
| `chore:` | tooling, dependencies, CI |

For changes that touch a specific module, scope it: `feat(engine):`, `fix(metrics):`, etc.

For significant changes, reference the relevant ADR or architecture section in the commit body.

## Before Contributing

- Read `backtest_auditor_c4_architecture.md` — scope boundaries are strict.
- The product is a **backtest auditing framework**. Changes that introduce live trading, broker connectivity, automated strategy discovery, or parameter optimization are out of scope.

## Development Setup

```bash
scripts/setup-dev.sh
```

## Quality Gates

All of the following must pass before a commit lands:

| Check | Command |
|---|---|
| Format | `uv run ruff format --check .` |
| Lint | `uv run ruff check .` |
| Type check | `uv run mypy src` |
| Tests | `uv run pytest` |

The pre-commit hook runs these automatically. Install it via `scripts/setup-dev.sh`.

## Testing Requirements

| Change type | Required tests |
|---|---|
| New calculation (metric, signal, cost) | Unit test with deterministic fixture and known expected value |
| New validation method | Reproducibility test with fixed `random_seed` |
| New data validation rule | Test for valid and invalid input |
| New reporting transformation | Test for expected output shape and fields |

## Methodological Boundary

Parameter sensitivity analysis exists to evaluate **fragility**, not to search for the best parameter set. Changes that shift this toward automated optimization are out of scope.

All stochastic procedures (bootstrap, Monte Carlo) must accept and record a `random_seed`.
