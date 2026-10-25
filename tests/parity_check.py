#!/usr/bin/env python3
"""Paritaetstest: Live-Entscheidung (trade_manager.decide_entry auf einem Kerzenfenster wie live)
== Backtest-Signal (backtester.signal_events auf voller Historie) -- je Kerze, Richtung, SL, TP.

Aufruf: python tests/parity_check.py [--candles 150] [--all-configs] [--keys BTC_4h ETH_2h]
Exit-Code 0 nur bei 100 % Uebereinstimmung (wird von der Quartals-Pipeline als Tor genutzt).
"""
import os
import sys
import json
import glob
import argparse
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from jaegerbot.utils.data import load_data, load_settings
from jaegerbot.utils.trade_manager import decide_entry, FETCH_CANDLES
from jaegerbot.analysis.backtester import signal_events, strategy_frame
from jaegerbot.analysis.optimizer import CONFIG_DIR


def load_configs(all_configs, keys):
    out = []
    from jaegerbot.strategy import trade_rules as tr
    active = {(s['symbol'], s['timeframe']) for s in load_settings()['live_trading_settings'].get('active_strategies', [])
              if s.get('profile', 'trend') == tr.PROFILE}
    for path in sorted(glob.glob(os.path.join(CONFIG_DIR, 'config_*.json'))):
        with open(path, 'r', encoding='utf-8') as f:
            c = json.load(f)
        sym, tf = c['market']['symbol'], c['market']['timeframe']
        key = f"{sym.split('/')[0]}_{tf}"
        if keys and key not in keys:
            continue
        if not keys and not all_configs and (sym, tf) not in active:
            continue
        out.append((key, sym, tf, c['strategy']))
    return out


def check(sym, tf, cfg, n_candles):
    coin = sym.split('/')[0]
    raw = load_data(sym, tf)
    sf = strategy_frame(coin, tf)
    if sf is None or raw.empty:
        return None, ['keine Daten/Vorhersagen']
    test_idx = sf.index[sf['pL'].notna()][-n_candles:]
    start = test_idx[0] + pd.Timedelta(tf)
    bt = {e['time'] - pd.Timedelta(tf): e for e in signal_events(coin, tf, cfg, start=start)}
    diffs, n_sig, max_dp = [], 0, 0.0
    pos = raw.index.get_indexer(test_idx)
    for ts, i in zip(test_idx, pos):
        window = raw.iloc[max(0, i + 1 - FETCH_CANDLES):i + 1]
        d = decide_entry(window, tf, cfg)
        if d.get('error'):
            diffs.append(f"{ts}: live Fehler {d['error']}")
            continue
        max_dp = max(max_dp, abs(d['pred']['pL'] - sf.at[ts, 'pL']), abs(d['pred']['pS'] - sf.at[ts, 'pS']))
        b = bt.get(ts)
        if d['side'] is None and b is None:
            continue
        n_sig += 1
        if (d['side'] is None) != (b is None) or d['side'] != b['side']:
            diffs.append(f"{ts}: live {d['side']} / Backtest {b['side'] if b else None}")
        elif not (np.isclose(d['sl'], b['sl'], rtol=1e-9) and np.isclose(d['tp'], b['tp'], rtol=1e-9)):
            diffs.append(f"{ts}: SL/TP live {d['sl']:.8g}/{d['tp']:.8g} BT {b['sl']:.8g}/{b['tp']:.8g}")
    return {'signals': n_sig, 'max_pred_diff': max_dp}, diffs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--candles', type=int, default=150)
    p.add_argument('--all-configs', action='store_true', help='alle Configs statt nur aktives Team')
    p.add_argument('--keys', nargs='*')
    a = p.parse_args()
    configs = load_configs(a.all_configs, a.keys)
    if not configs:
        print("Keine Configs zu pruefen.")
        sys.exit(1)
    failed = 0
    for key, sym, tf, cfg in configs:
        res, diffs = check(sym, tf, cfg, a.candles)
        if res is None or diffs:
            failed += 1
            print(f"FEHL {key}: {len(diffs)} Abweichung(en)")
            for d in diffs[:5]:
                print(f"     {d}")
        else:
            print(f"OK   {key}: {res['signals']} Signale gleich, max. Vorhersage-Differenz {res['max_pred_diff']:.2e}")
    print(f"\nParitaet: {len(configs) - failed}/{len(configs)} Configs identisch.")
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
