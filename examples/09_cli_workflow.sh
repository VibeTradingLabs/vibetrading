#!/bin/bash
# Example 09: Full CLI Workflow
#
# This script demonstrates the complete vibetrading CLI workflow:
# 1. Generate a strategy from a template
# 2. Validate it
# 3. Run a backtest
#
# Usage: bash examples/09_cli_workflow.sh

set -e

echo "=== Step 1: Generate strategy from template ==="
vibetrading template momentum asset=BTC leverage=3 sma_fast=10 sma_slow=30 -o /tmp/my_strategy.py
echo ""

echo "=== Step 2: Validate the strategy ==="
vibetrading validate /tmp/my_strategy.py
echo ""

echo "=== Step 3: Run backtest ==="
vibetrading backtest /tmp/my_strategy.py \
    --interval 1h \
    --balance 10000 \
    --exchange binance \
    --start 2025-01-01 \
    --end 2025-04-01
echo ""

echo "=== Step 4: Run backtest with JSON output ==="
vibetrading backtest /tmp/my_strategy.py --interval 1h --json | python3 -m json.tool
echo ""

echo "=== Available templates ==="
vibetrading template --list
echo ""

# Cleanup
rm -f /tmp/my_strategy.py
echo "Done!"
