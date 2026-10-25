#!/bin/bash
# Regeln je Timeframe, aktuelles Team und Portfolio-Kennzahlen (ohne etwas zu aendern)
cd "$(dirname "$0")"
PY=.venv/bin/python3; [ -x "$PY" ] || PY=$(command -v python3 || command -v python)
export PYTHONPATH=src
$PY - <<'PY'
import json, os
r = json.load(open('artifacts/results/rules.json')) if os.path.exists('artifacts/results/rules.json') else {}
print("Regeln je Timeframe (IS-Wahl, OOS-Bestaetigung):")
for tf, x in r.items():
    if 'cfg' not in x:
        print(f"  {tf:3s} {x.get('status')}"); continue
    c, i, o = x['cfg'], x['is'], x['oos']
    print(f"  {tf:3s} {'BESTAETIGT ' if x['confirmed'] else 'abgelehnt  '} {c['sides']:2s} Top {100 - c['quantile'] * 100:.1f} % "
          f"Trend={'ja' if c['ema_filter'] else 'nein'} | {x['n_coins']} Coins | IS {i['pnl_pct']:+.1f} % DD {i['max_dd_pct']:.1f} | "
          f"OOS {o['pnl_pct']:+.1f} % DD {o['max_dd_pct']:.1f} n {int(o['trades'])} avgR {o['avg_R']:+.3f}")
lt = json.load(open('settings.json'))['live_trading_settings']
team = lt.get('active_strategies', [])
print(f"\nLive-Team: {len(team)} Strategien, Risiko {lt.get('risk_per_trade_pct')} % je Trade, max {lt.get('max_open_positions')} Positionen")
print('  ' + ', '.join(f"{s['symbol'].split('/')[0]} {s['timeframe']}" for s in team))
PY
echo
$PY run_portfolio_optimizer.py
