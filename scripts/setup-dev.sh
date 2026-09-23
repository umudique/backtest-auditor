#!/usr/bin/env bash
# Backtest Auditor — development environment setup
# Run once after cloning the repository.

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}  ✓${NC} $1"; }
warn() { echo -e "${YELLOW}  !${NC} $1"; }

echo "Backtest Auditor — development environment setup"
echo "═════════════════════════════════════════════════"

# ── uv ───────────────────────────────────────────────────────────────────────
echo ""
echo "→ Checking uv..."
if command -v uv &>/dev/null; then
    pass "uv present ($(uv --version))"
else
    echo "  installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    pass "uv installed"
fi

# ── Virtual environment + dependencies ───────────────────────────────────────
echo ""
echo "→ Creating virtual environment and installing dependencies..."
uv venv .venv
uv sync --extra dev
pass "dependencies installed"

# ── Git hooks ────────────────────────────────────────────────────────────────
echo ""
echo "→ Configuring git hooks..."
git config core.hooksPath .githooks
chmod +x .githooks/pre-commit
pass "git hooks path set to .githooks"

# ── Verify ───────────────────────────────────────────────────────────────────
echo ""
echo "→ Verifying setup..."
uv run ruff --version >/dev/null && pass "ruff ok"
uv run mypy --version >/dev/null && pass "mypy ok"
uv run pytest --collect-only -q 2>/dev/null | tail -1 && pass "pytest ok"

echo ""
echo "═════════════════════════════════════════════════"
echo -e "${GREEN}Setup complete.${NC}"
echo ""
echo "Quick reference:"
echo "  uv run ruff format .        — auto-format"
echo "  uv run ruff check . --fix   — auto-fix lint"
echo "  uv run mypy src             — type check"
echo "  uv run pytest               — run all tests"
echo "  uv run streamlit run app/streamlit_app.py — launch UI"
