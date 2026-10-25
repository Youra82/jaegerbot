#!/usr/bin/env python3
"""Kerzen-Cache fuer alle Coins des Universums und alle Timeframes aktualisieren.

Aufruf: python download_data.py [--coins BTC ETH] [--timeframes 4h] [--jobs 4]
"""
import os
import sys
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

import json
import pandas as pd
from jaegerbot.utils.data import update_cache, symbol_of, load_settings, _get_exchange
from jaegerbot.utils.data_check import check_candles

logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(levelname)s %(message)s')


def main():
    s = load_settings()['research']
    p = argparse.ArgumentParser()
    p.add_argument('--coins', nargs='*', default=s['universe'])
    p.add_argument('--timeframes', nargs='*', default=s['timeframes'])
    p.add_argument('--start', default=s['history_start'])
    p.add_argument('--jobs', type=int, default=4)
    a = p.parse_args()
    _get_exchange()  # Maerkte einmal laden, bevor die Threads starten

    def job(task):
        coin, tf = task
        df = update_cache(symbol_of(coin), tf, a.start)
        problems = check_candles(df, tf, max_age=pd.Timedelta(hours=6) + pd.Timedelta(tf))
        rng = f"{df.index[0].date()} -> {df.index[-1]}" if not df.empty else 'keine Daten'
        print(f"{coin:8s} {tf:3s} {len(df):6d} Kerzen  {rng}  {'OK' if not problems else '; '.join(problems)}", flush=True)
        return {'coin': coin, 'timeframe': tf, 'candles': len(df), 'problems': problems,
                'first': str(df.index[0]) if len(df) else None, 'last': str(df.index[-1]) if len(df) else None}

    tasks = [(c, tf) for c in a.coins for tf in a.timeframes]
    with ThreadPoolExecutor(a.jobs) as ex:
        report = list(ex.map(job, tasks))
    bad = [r for r in report if r['problems']]
    os.makedirs(os.path.join(PROJECT_ROOT, 'artifacts', 'db'), exist_ok=True)
    with open(os.path.join(PROJECT_ROOT, 'artifacts', 'db', 'data_health.json'), 'w', encoding='utf-8') as f:
        json.dump({'checked': pd.Timestamp.now(tz='UTC').isoformat(), 'files': report}, f, indent=1)
    print(f"\nDatenpruefung: {len(report) - len(bad)}/{len(report)} Dateien in Ordnung.", flush=True)


if __name__ == '__main__':
    main()
