#!/usr/bin/env python3
"""Portfolio: welche bestaetigten Timeframe-Regeln zusammen, mit welchem Risiko je Trade.

Ziel wie fibot (objective "pnl"): maximales Endkapital bei Drawdown <= max_drawdown_pct -- gerechnet
ueber die GANZE Walk-Forward-Zeit seit first_trade_date (IS + OOS), nicht nur die letzten 26 Wochen,
damit der Drawdown nicht aus einer guenstigen Phase geschaetzt wird. Keine Coin-Auswahl (je Coin war
im Test reiner Auswahleffekt) und keine woechentliche Neuwahl (senkte im Test die Rendite meist).

Aufruf: python run_portfolio_optimizer.py [--write] [--capital 100]
"""
import os
import sys
import json
import argparse
import itertools
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from jaegerbot.utils.data import load_settings
from jaegerbot.analysis.backtester import run_portfolio
from jaegerbot.analysis.optimizer import RULES_FILE, events_for, oos_start

RISK_GRID = np.round(np.arange(0.5, 4.01, 0.05), 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--write', action='store_true', help='Team + Risiko in settings.json schreiben')
    p.add_argument('--capital', type=float, default=100.0)
    a = p.parse_args()
    s = load_settings()
    lt, crit, rs = s['live_trading_settings'], s['optimization_settings'], s['research']
    if not os.path.exists(RULES_FILE):
        print("Keine Regeln -- erst den Optimizer laufen lassen.")
        sys.exit(1)
    with open(RULES_FILE, 'r', encoding='utf-8') as f:
        rules = {tf: r for tf, r in json.load(f).items() if r.get('confirmed') and tf in rs['timeframes']}
    if not rules:
        print("Kein Timeframe bestaetigt -- Team bleibt unveraendert.")
        sys.exit(1)
    start = pd.Timestamp(rs['first_trade_date'], tz='UTC')
    ob = oos_start(s)
    years = (pd.Timestamp.now(tz='UTC') - start).days / 365.25
    ev_tf = {tf: events_for(r['coins'], tf, r['cfg'], start) for tf, r in rules.items()}
    print(f"Bestaetigte Regeln: {', '.join(f'{tf} ({len(r['coins'])} Coins)' for tf, r in rules.items())} | "
          f"seit {start:%Y-%m-%d}, max {lt['max_open_positions']} Positionen, DD-Limit {crit['max_drawdown_pct']} %")
    best = None
    for k in range(1, len(rules) + 1):
        for combo in itertools.combinations(sorted(rules), k):
            ev = [e for tf in combo for e in ev_tf[tf]]
            for risk in RISK_GRID:
                st = run_portfolio([], a.capital, float(risk), lt['max_open_positions'], lt['max_leverage'],
                                   events=[dict(e) for e in ev])['stats']
                if st['max_dd_pct'] > crit['max_drawdown_pct']:
                    continue
                if best is None or st['end_capital'] > best[2]['end_capital']:
                    best = (combo, float(risk), st)
    if best is None:
        print("Keine Kombination mit Drawdown im Limit -- Team bleibt unveraendert.")
        sys.exit(1)
    combo, risk, st = best
    cagr = ((st['end_capital'] / a.capital) ** (1 / years) - 1) * 100
    oos = run_portfolio([], a.capital, risk, lt['max_open_positions'], lt['max_leverage'],
                        events=[dict(e) for tf in combo for e in ev_tf[tf] if e['time'] >= ob])['stats']
    print(f"Gewaehlt: {'+'.join(combo)} mit {risk} % Risiko je Trade")
    print(f"Seit {start:%Y-%m-%d}: {cagr:+.1f} %/Jahr, DD {st['max_dd_pct']:.1f} %, {st['trades']} Trades, avgR {st['avg_R']:+.3f}")
    print(f"Davon OOS ab {ob:%Y-%m-%d}: {oos['pnl_pct']:+.1f} %, DD {oos['max_dd_pct']:.1f} %, {oos['trades']} Trades")
    if a.write:
        path = os.path.join(PROJECT_ROOT, 'settings.json')
        with open(path, 'r', encoding='utf-8') as f:
            full = json.load(f)
        full['live_trading_settings']['risk_per_trade_pct'] = risk
        full['live_trading_settings'].pop('risk_scale', None)
        full['live_trading_settings']['active_strategies'] = [
            {'symbol': f"{c}/USDT:USDT", 'timeframe': tf, 'active': True} for tf in combo for c in rules[tf]['coins']]
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(full, f, indent=2, ensure_ascii=False)
        print(f"settings.json aktualisiert ({len(full['live_trading_settings']['active_strategies'])} Strategien).")


if __name__ == '__main__':
    main()
