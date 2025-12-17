#!/usr/bin/env python3
"""
Direkter Backtest-Aufruf mit den Live-Strategien
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from jaegerbot.analysis.portfolio_simulator import run_portfolio_simulation
from datetime import date

print("=" * 60)
print("  Direkter Backtest mit Live-Strategien")
print("=" * 60)
print()

# Strategien aus settings.json
strategies_to_test = [
    "config_SOLUSDTUSDT_15m.json",
    "config_DOGEUSDTUSDT_15m.json", 
    "config_ETHUSDTUSDT_15m.json",
    "config_XRPUSDTUSDT_15m.json",
    "config_ADAUSDTUSDT_1d.json",
    "config_AAVEUSDTUSDT_1d.json"
]

print("Teste folgende Strategien:")
for s in strategies_to_test:
    print(f"  - {s}")
print()

# Führe Portfolio-Simulation aus
result = run_portfolio_simulation(
    start_date=date(2025, 11, 1),
    end_date=date(2025, 12, 17),
    start_capital=250.0,
    selected_strategies=strategies_to_test,
    export_filename='live_strategies_backtest.csv'
)

print()
print("=" * 60)
print("  Backtest abgeschlossen!")
print("=" * 60)
print()

if result:
    print(f"📊 ERGEBNISSE:")
    print(f"   Startkapital: 250.00 USDT")
    print(f"   Endkapital: {result['final_equity']:.2f} USDT")
    print(f"   PnL: {result['total_pnl']:.2f} USDT ({result['total_return_pct']:.2f}%)")
    print(f"   Anzahl Trades: {result['num_trades']}")
    print(f"   Max Drawdown: {result['max_dd_pct']:.2f}%")
    print(f"   Liquidiert: {'JA' if result['liquidated'] else 'NEIN'}")
    print()
