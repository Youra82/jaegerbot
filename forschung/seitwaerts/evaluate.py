"""Auswertung der Vorregistrierung (PREREG.md). Zwei Stufen, weil das Profil je Prozess gilt:

  JAEGER_PROFILE=trend python forschung/seitwaerts/evaluate.py dump-trend   # Live-Trend-Signale speichern
  JAEGER_PROFILE=range python forschung/seitwaerts/evaluate.py evaluate     # Kriterien 1-4
"""
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)

from jaegerbot.strategy import trade_rules as tr
from jaegerbot.utils.data import load_settings
from jaegerbot.analysis import optimizer as op
from jaegerbot.analysis.backtester import run_portfolio, strategy_frame

OUT = os.path.join(ROOT, 'forschung', 'seitwaerts')
TREND_EVENTS = os.path.join(os.environ.get('TMP', '/tmp'), 'jaeger_trend_events.pkl')
START = pd.Timestamp('2023-01-01', tz='UTC')
SIDE_A, SIDE_B = pd.Timestamp('2024-03-15', tz='UTC'), pd.Timestamp('2026-07-31', tz='UTC')
TF = '4h'


def dump_trend():
    assert tr.PROFILE == 'trend'
    lt = load_settings()['live_trading_settings']
    coins = sorted({s['symbol'].split('/')[0] for s in lt['active_strategies'] if s['timeframe'] == TF})
    with open(op.RULES_FILE, encoding='utf-8') as f:
        cfg = json.load(f)[TF]['cfg']
    ev = op.events_for(coins, TF, cfg, START)
    with open(TREND_EVENTS, 'wb') as f:
        pickle.dump({'coins': coins, 'cfg': cfg, 'events': ev}, f)
    print(f"Trend: {len(ev)} Signale aus {len(coins)} Coins gespeichert")


def _cagr(st, a, b):
    return ((st['end_capital'] / 100) ** (365.25 / (b - a).days) - 1) * 100


def _random_control(coins, cfg, n_mult=5, seed=0):
    rng = np.random.default_rng(seed)
    ann, rnd = [], []
    for c in coins:
        df = strategy_frame(c, TF)
        df = df[df.index >= START]
        o, h, l, cl, atr = (df[k].to_numpy() for k in ('open', 'high', 'low', 'close', 'atr'))
        rows = df[['pL', 'pS', 'ema20', 'ema50', 'atr', tr.thr_col('L', cfg['quantile']), tr.thr_col('S', cfg['quantile'])]]
        valid = rows['pL'].notna().to_numpy()
        sig = np.array([tr.entry_signal(r, cfg) if v else None for r, v in zip(rows.to_dict('records'), valid)], dtype=object)

        def R(i, side):
            sl, tp = tr.levels(cl[i], atr[i], side)
            ex = tr.simulate_exit(o, h, l, cl, i, side, sl, tp, tr.MAX_HOLD[TF])
            if ex is None:
                return None
            ef = tr.fill_price(cl[i], side, True)
            return tr.trade_pnl(side, 1.0, ef, ex[1])[0] / abs(ef - sl)
        idx = np.flatnonzero(sig != None)  # noqa: E711
        ann += [x for x in (R(i, sig[i]) for i in idx) if x is not None]
        # Zufall: gleiche Seitenverteilung, gleicher Trendfilter (falls aktiv)
        sides = [s for s in ('long', 'short') if s[0].upper() in cfg['sides']]
        if cfg.get('regime') == 'sideways':
            valid = valid & (np.abs(df['ema20'] - df['ema50']) < tr.SIDEWAYS_SPREAD_ATR * df['atr']).to_numpy()
        pool = np.flatnonzero(valid)
        for i in rng.choice(pool, min(len(pool), n_mult * max(1, len(idx))), replace=False):
            side = sides[rng.integers(len(sides))]
            if cfg['ema_filter'] and ((side == 'long') != (df['ema20'].iat[i] > df['ema50'].iat[i])):
                continue
            x = R(i, side)
            if x is not None:
                rnd.append(x)
    a, b = np.array(ann), np.array(rnd)
    t = (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return a.mean(), len(a), b.mean(), len(b), t


def evaluate(v2=False):
    assert tr.PROFILE == 'range'
    if v2:   # PREREG v2: nur Seitwaertskerzen, ohne Trendfilter
        import itertools
        op.GRID[:] = [{'sides': sd, 'quantile': q, 'ema_filter': False, 'regime': 'sideways'}
                      for sd, q in itertools.product(('L', 'S', 'LS'), tr.QUANTILE_LEVELS)]
    s = load_settings()
    s['live_trading_settings']['risk_per_trade_pct'] = 1.0           # laut PREREG: 1 % Risiko, max. 20 Pos.
    s['live_trading_settings']['max_open_positions'] = 20
    lt = s['live_trading_settings']
    with open(TREND_EVENTS, 'rb') as f:
        trend = pickle.load(f)
    coins = trend['coins']
    res = {}

    # Kriterium 1 (= 4, Rueckversetz-Test: Auswahl nur bis OOS-Beginn 26.04.2026, danach 26 W vorwaerts)
    import jaegerbot.analysis.optimizer as opt
    opt.tradable_coins = lambda timeframe, settings=None: coins     # gleiches Universum wie das Live-Team
    r = opt.optimize_timeframe(TF, s)
    print(f"\nK1/K4 Regel-Wahl IS, OOS-Bestaetigung: {r['status']} | Regel {r.get('cfg')} | "
          f"IS {r['is']['pnl_pct']:+.1f} % DD {r['is']['max_dd_pct']:.1f} avgR {r['is']['avg_R']:+.3f} | "
          f"OOS {r['oos']['pnl_pct']:+.1f} % DD {r['oos']['max_dd_pct']:.1f} n {int(r['oos']['trades'])} avgR {r['oos']['avg_R']:+.3f}")
    res['k1'] = r['status'] == 'bestaetigt'
    res['k4'] = r['oos']['pnl_pct'] > 0
    cfg = r['cfg']

    # Kriterium 2: Kontrollgruppe
    a, na, b, nb, t = _random_control(coins, cfg)
    print(f"K2 Kontrollgruppe: ANN {a:+.3f} R (n={na}) vs Zufall {b:+.3f} R (n={nb}), t = {t:+.2f}")
    res['k2'] = t > 2

    # Kriterium 3: gemeinsames Konto Trend + Range
    range_ev = op.events_for(coins, TF, cfg, START)
    for e in range_ev:
        e['key'] += '_R'
    tr_ev = trend['events']

    def port(evs, risk, a=None, b=None):
        sel = [dict(e) for e in evs if (a is None or e['time'] >= a) and (b is None or e['time'] <= b)]
        return run_portfolio([], 100, risk, 20, 20, events=sel)['stats']
    t_side = port(tr_ev, 0.95, SIDE_A, SIDE_B)
    c_side = port(tr_ev + range_ev, 0.95, SIDE_A, SIDE_B)
    print(f"K3a Seitwaertsfenster 15.03.2024-31.07.2026 (0,95 %): Trend {t_side['pnl_pct']:+.1f} % DD {t_side['max_dd_pct']:.1f} "
          f"| Trend+Range {c_side['pnl_pct']:+.1f} % DD {c_side['max_dd_pct']:.1f}")
    res['k3a'] = c_side['end_capital'] > t_side['end_capital']
    now = pd.Timestamp.now(tz='UTC')
    best = {}
    for name, evs in (('Trend', tr_ev), ('Trend+Range', tr_ev + range_ev), ('Range', range_ev)):
        bb = None
        for risk in np.round(np.arange(0.3, 3.01, 0.05), 2):
            st = port(evs, float(risk))
            if st['max_dd_pct'] <= 30 and (bb is None or st['end_capital'] > bb[1]['end_capital']):
                bb = (float(risk), st)
        best[name] = bb
        if bb:
            print(f"K3b {name:12s}: Risiko {bb[0]:.2f} % -> {_cagr(bb[1], START, now):+.1f} %/J DD {bb[1]['max_dd_pct']:.1f} "
                  f"n {bb[1]['trades']} avgR {bb[1]['avg_R']:+.3f}")
    res['k3b'] = bool(best['Trend+Range'] and best['Trend']
                      and best['Trend+Range'][1]['end_capital'] > best['Trend'][1]['end_capital'])
    res['k3'] = res['k3a'] and res['k3b']

    passed = all(res[k] for k in ('k1', 'k2', 'k3', 'k4'))
    print('\nErgebnis:', {k: ('ok' if v else 'NICHT erfuellt') for k, v in res.items()},
          '=> BESTANDEN' if passed else '=> NICHT BESTANDEN')
    with open(os.path.join(OUT, 'result_v2.json' if v2 else 'result.json'), 'w', encoding='utf-8') as f:
        json.dump({'rule': cfg, 'criteria': res, 'passed': passed,
                   'is': r['is'], 'oos': r['oos'], 'control': {'ann': a, 'random': b, 't': t},
                   'sideways': {'trend': t_side['pnl_pct'], 'combined': c_side['pnl_pct']},
                   'best': {k: (v[0], _cagr(v[1], START, now), v[1]['max_dd_pct']) if v else None
                            for k, v in best.items()}}, f, indent=2, default=float)


if __name__ == '__main__':
    {'dump-trend': dump_trend, 'evaluate': evaluate, 'evaluate-v2': lambda: evaluate(True)}[sys.argv[1]]()
