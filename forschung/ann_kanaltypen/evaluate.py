"""Auswertung PREREG.md (ANN lernt Kanaltypen). Voraussetzungen:
  JAEGER_PROFILE=trend python forschung/seitwaerts/evaluate.py dump-trend
  JAEGER_PROFILE=geo   python -m jaegerbot.analysis.walkforward --timeframes 4h
Aufruf:
  JAEGER_PROFILE=geo python forschung/ann_kanaltypen/evaluate.py
  JAEGER_PROFILE=geo python tests/parity_check.py --all-configs      (Kriterium 5)
"""
import os
import sys
import json
import pickle
import importlib.util
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)

from jaegerbot.strategy import trade_rules as tr
from jaegerbot.utils.data import load_settings
from jaegerbot.analysis import optimizer as opt


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pairs = _load('pairs_eval', os.path.join(ROOT, 'forschung', 'pairs', 'evaluate.py'))       # R-Simulator
side = _load('side_eval', os.path.join(ROOT, 'forschung', 'seitwaerts', 'evaluate.py'))    # Kontrollgruppe

START, OOS = pairs.START, pairs.OOS
SIDE_A, SIDE_B = pairs.SIDE_A, pairs.SIDE_B


def to_R(events, risk):
    out = []
    for e in events:
        if e['exit_time'] is None:
            continue
        ef = tr.fill_price(e['signal_close'], e['side'], True)
        pnl, _ = tr.trade_pnl(e['side'], 1.0, ef, e['exit_raw'])
        out.append({'time': e['time'], 'exit_time': e['exit_time'], 'coins': (e['coin'],),
                    'R': pnl / abs(ef - e['sl']), 'risk': risk})
    return out


def best(ev):
    bb = None
    for sc in np.round(np.arange(0.1, 4.01, 0.05), 2):
        st = pairs.simulate(ev, None, scale=float(sc))
        if st['dd'] <= 30 and (bb is None or st['end'] > bb[1]['end']):
            bb = (float(sc), st)
    return bb


def main():
    assert tr.PROFILE in ('geo', 'geohtf') and tr.GEO_FEATURES
    with open(pairs.TREND_EVENTS, 'rb') as f:
        trend = pickle.load(f)
    coins = trend['coins']
    s = load_settings()
    opt.tradable_coins = lambda timeframe, settings=None: coins
    r = opt.optimize_timeframe('4h', s)
    print(f"\nRegel geo: {r['status']} | {r['cfg']} | IS {r['is']['pnl_pct']:+.1f} % DD {r['is']['max_dd_pct']:.1f} "
          f"avgR {r['is']['avg_R']:+.3f} | OOS {r['oos']['pnl_pct']:+.1f} % DD {r['oos']['max_dd_pct']:.1f} "
          f"n {int(r['oos']['trades'])} avgR {r['oos']['avg_R']:+.3f}")
    geo_ev = to_R(opt.events_for(coins, '4h', r['cfg'], START), 0.95)
    tr_ev = to_R(trend['events'], 0.95)
    now = pd.Timestamp.now(tz='UTC')
    bt, bg = best(tr_ev), best(geo_ev)
    print(f"K1 DD<=30 seit 2023: trend {pairs.cagr(bt[1], START, now):+.1f} %/J (n {bt[1]['n']}) | "
          f"geo {pairs.cagr(bg[1], START, now):+.1f} %/J (n {bg[1]['n']}, avgR {bg[1]['avgR']:+.3f}, Faktor {bg[0]})")
    to, go = pairs.simulate(tr_ev, None, start=OOS), pairs.simulate(geo_ev, None, start=OOS)
    ts, gs = pairs.simulate(tr_ev, None, start=SIDE_A, end=SIDE_B), pairs.simulate(geo_ev, None, start=SIDE_A, end=SIDE_B)
    print(f"K2 OOS 26 W (0,95 %): trend {to['pnl']:+.1f} % | geo {go['pnl']:+.1f} % (DD {go['dd']:.1f}, n {go['n']})")
    print(f"K3 Seitwaertsfenster (0,95 %): trend {ts['pnl']:+.1f} % | geo {gs['pnl']:+.1f} % (DD {gs['dd']:.1f})")
    a, na, b, nb, t = side._random_control(coins, r['cfg'])
    print(f"K4 Kontrolle: ANN {a:+.3f} R (n {na}) vs Zufall {b:+.3f} R (n {nb}), t {t:+.2f}")
    res = {'k1': r['status'] == 'bestaetigt' and bg[1]['end'] > bt[1]['end'], 'k2': go['end'] >= to['end'],
           'k3': gs['end'] > ts['end'], 'k4': t > 2}
    print('Ergebnis K1-K4:', {k: ('ok' if v else 'NICHT erfuellt') for k, v in res.items()},
          '(K5 = Paritaetstest separat)')
    with open(os.path.join(ROOT, 'forschung', 'ann_kanaltypen', f'result_{tr.PROFILE}.json'), 'w', encoding='utf-8') as f:
        json.dump({'rule': r['cfg'], 'criteria_1_4': res, 'is': r['is'], 'oos': r['oos'],
                   'cagr_trend': pairs.cagr(bt[1], START, now), 'cagr_geo': pairs.cagr(bg[1], START, now),
                   'oos_trend': to['pnl'], 'oos_geo': go['pnl'], 'side_trend': ts['pnl'], 'side_geo': gs['pnl'],
                   'control': {'ann': a, 'random': b, 't': t}}, f, indent=2, default=float)


if __name__ == '__main__':
    main()
