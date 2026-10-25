# src/jaegerbot/analysis/optimizer.py
"""Regel-Suche je Timeframe: Auswahl nur In-Sample, Bestaetigung Out-of-Sample (wie fibot, aber auf
Portfolio-Ebene statt je Coin).

Warum nicht je Coin (Test 25.10.2026): Auswahl je Coin x Timeframe bestaetigte 20 Configs, aber
Zufalls-Vorhersagen bestaetigten mit derselben Methode 4-13 und erzielten dieselben Backtest-Zahlen.
Rueckversetzt (Auswahl bis 26.04.2026, dann 26 W vorwaerts): Auswahl je Coin -2 %, Zufall Ø -19 %,
eine einheitliche Regel ueber alle etablierten Coins +22 % (DD 6 %).

IS  = first_trade_date bis OOS-Beginn (heute - oos_weeks), OOS = danach. Regel = Seiten x Schwelle x
Trendfilter, gewaehlt nach Erwartungswert je Trade (min. MIN_IS_TRADES, IS-PnL > 0), bewertet als
Portfolio aller etablierten Coins mit live-Einstellungen (Risiko, max. Positionen). Bestaetigt, wenn
OOS-PnL >= min_oos_pnl_pct, OOS-DD <= max_drawdown_pct, OOS-Trades >= min_oos_trades.

Aufruf: python -m jaegerbot.analysis.optimizer [--timeframes 4h 6h]
Schreibt je Coin eine Config (Regel ihres Timeframes) + artifacts/results/rules.json.
"""
import os
import sys
import json
import argparse
import itertools
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from jaegerbot.utils.data import load_settings, established_coins
from jaegerbot.utils.data_check import enough_history_for_config
from jaegerbot.analysis.backtester import run_portfolio, signal_events, strategy_frame
from jaegerbot.strategy import trade_rules as tr

CONFIG_DIR = tr.profile_dir(os.path.join(PROJECT_ROOT, 'src', 'jaegerbot', 'strategy', 'configs'))
RULES_FILE = tr.profile_dir(os.path.join(PROJECT_ROOT, 'artifacts', 'results', 'rules')) + '.json'
GRID = [{'sides': s, 'quantile': q, 'ema_filter': e}
        for s, q, e in itertools.product(('L', 'S', 'LS'), tr.QUANTILE_LEVELS, (False, True))]
MIN_IS_TRADES = 100


def config_path(coin, timeframe):
    return os.path.join(CONFIG_DIR, f"config_{coin}USDTUSDT_{timeframe}.json")


def oos_start(settings=None):
    rs = (settings or load_settings())['research']
    return pd.Timestamp.now(tz='UTC').normalize() - pd.Timedelta(weeks=int(rs['oos_weeks']))


def tradable_coins(timeframe, settings=None):
    """Etablierte Coins (Listing-Alter >= min_listing_days) mit Vorhersagen, aktuellen Daten und
    genug Historie. Auf jungen Altcoins hatte das ANN im Test keine Kante (~0 R)."""
    s = settings or load_settings()
    rs = s['research']
    out = []
    for coin in established_coins(rs['universe'], int(rs['min_listing_days'])):
        df = strategy_frame(coin, timeframe)
        if df is None or not df['pL'].notna().any() or not enough_history_for_config(df, timeframe):
            continue
        if df.index[-1] < pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=3):
            continue
        out.append(coin)
    return out


def events_for(coins, timeframe, cfg, start, end=None):
    ev = []
    for c in coins:
        for e in signal_events(c, timeframe, cfg, start, end):
            e['key'] = f"{c}_{timeframe}"
            ev.append(e)
    return ev


def portfolio_stats(coins, timeframe, cfg, start, end, lt):
    return run_portfolio([], 100.0, lt['risk_per_trade_pct'], lt['max_open_positions'], lt['max_leverage'],
                         events=events_for(coins, timeframe, cfg, start, end))['stats']


def optimize_timeframe(timeframe, s):
    rs, crit, lt = s['research'], s['optimization_settings'], s['live_trading_settings']
    coins = tradable_coins(timeframe, s)
    if not coins:
        return {'timeframe': timeframe, 'status': 'keine handelbaren Coins'}
    is_begin, ob = pd.Timestamp(rs['first_trade_date'], tz='UTC'), oos_start(s)
    rows = []
    for cfg in GRID:
        st = portfolio_stats(coins, timeframe, cfg, is_begin, ob, lt)
        rows.append((cfg, st))
        print(f"  {timeframe} {cfg['sides']:2s} q{cfg['quantile']:<5} ema={int(cfg['ema_filter'])}: IS {st['pnl_pct']:+8.1f} % "
              f"DD {st['max_dd_pct']:5.1f} n {st['trades']:5d} avgR {st['avg_R']:+.3f}", flush=True)
    valid = [r for r in rows if r[1]['trades'] >= MIN_IS_TRADES and r[1]['pnl_pct'] > 0]
    if not valid:
        return {'timeframe': timeframe, 'status': 'keine positive IS-Regel', 'coins': coins}
    cfg, ist = max(valid, key=lambda r: r[1]['avg_R'])   # Erwartungswert je Trade, nicht Endkapital
    ost = portfolio_stats(coins, timeframe, cfg, ob, None, lt)
    confirmed = (ist['pnl_pct'] > 0 and ost['pnl_pct'] >= crit['min_oos_pnl_pct']
                 and ost['max_dd_pct'] <= crit['max_drawdown_pct'] and ost['trades'] >= crit['min_oos_trades'])
    clean = lambda d: {k: round(float(v), 3) for k, v in d.items() if isinstance(v, (int, float))}
    meta = {'is_start': f"{is_begin:%Y-%m-%d}", 'oos_start': f"{ob:%Y-%m-%d}", 'is': clean(ist), 'oos': clean(ost),
            'risk_pct': lt['risk_per_trade_pct'], 'confirmed': bool(confirmed), 'n_coins': len(coins),
            'created': f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M}"}
    os.makedirs(CONFIG_DIR, exist_ok=True)
    for coin in coins:
        with open(config_path(coin, timeframe), 'w', encoding='utf-8') as f:
            json.dump({'market': {'symbol': f"{coin}/USDT:USDT", 'timeframe': timeframe},
                       'strategy': cfg, '_backtest': meta}, f, indent=2)
    return {'timeframe': timeframe, 'status': 'bestaetigt' if confirmed else 'nicht bestaetigt',
            'cfg': cfg, 'coins': coins, **meta}


def main():
    s = load_settings()
    p = argparse.ArgumentParser()
    p.add_argument('--timeframes', nargs='*', default=s['research']['timeframes'])
    a = p.parse_args()
    crit = s['optimization_settings']
    print(f"OOS ab {oos_start(s):%Y-%m-%d} | Kriterien: IS>0, OOS>={crit['min_oos_pnl_pct']}%, "
          f"DD<={crit['max_drawdown_pct']}%, OOS-Trades>={crit['min_oos_trades']}", flush=True)
    # Configs dieser Timeframes neu: es gelten nur die Regeln dieses Laufs
    if os.path.isdir(CONFIG_DIR):
        for f in os.listdir(CONFIG_DIR):
            if any(f.endswith(f"_{tf}.json") for tf in a.timeframes):
                os.remove(os.path.join(CONFIG_DIR, f))
    results = {}
    for tf in a.timeframes:
        r = optimize_timeframe(tf, s)
        results[tf] = r
        if 'cfg' in r:
            c = r['cfg']
            print(f"{tf}: {r['status']} | Regel {c['sides']} q{c['quantile']} ema={int(c['ema_filter'])} | {len(r['coins'])} Coins | "
                  f"IS {r['is']['pnl_pct']:+.1f} % DD {r['is']['max_dd_pct']:.1f} avgR {r['is']['avg_R']:+.3f} | "
                  f"OOS {r['oos']['pnl_pct']:+.1f} % DD {r['oos']['max_dd_pct']:.1f} n {int(r['oos']['trades'])} "
                  f"avgR {r['oos']['avg_R']:+.3f}", flush=True)
        else:
            print(f"{tf}: {r['status']}", flush=True)
    os.makedirs(os.path.dirname(RULES_FILE), exist_ok=True)
    old = {}
    if os.path.exists(RULES_FILE):
        with open(RULES_FILE, 'r', encoding='utf-8') as f:
            old = json.load(f)
    old.update(results)
    with open(RULES_FILE, 'w', encoding='utf-8') as f:
        json.dump(old, f, indent=2, default=str)


if __name__ == '__main__':
    main()
