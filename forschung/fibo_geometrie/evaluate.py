"""Auswertung PREREG.md (fibot-Geometrie im jaegerbot). Braucht die Trend-Signale:
  JAEGER_PROFILE=trend python forschung/seitwaerts/evaluate.py dump-trend
  python forschung/fibo_geometrie/evaluate.py
"""
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FIBOT_SRC = os.path.abspath(os.path.join(ROOT, '..', 'fibot', 'src'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'forschung', 'pairs'))
os.chdir(ROOT)

from jaegerbot.strategy import trade_rules as tr
from jaegerbot.analysis.backtester import strategy_frame
from evaluate import simulate, cagr, TREND_EVENTS          # R-Simulator (gegen Bot geprueft)

START = pd.Timestamp('2023-01-01', tz='UTC')
OOS = pd.Timestamp.now(tz='UTC').normalize() - pd.Timedelta(weeks=26)
SIDE_A, SIDE_B = pd.Timestamp('2024-03-15', tz='UTC'), pd.Timestamp('2026-07-31', tz='UTC')
TF = pd.Timedelta('4h')
HOLD = 60


def fibot_geometry(coin):
    sys.path.insert(0, FIBOT_SRC)
    from fibot.strategy import confluence_logic as cl
    p = {**cl.DEFAULTS, **cl.V2}
    df = strategy_frame(coin, '4h')
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ('open', 'high', 'low', 'close'))
    lo = cl._plan_one_direction(h, l, c, p)
    sh = cl._plan_one_direction(-l, -h, -c, p)
    # Long-Seite findet nur steigende Kanaele; fallende kommen aus der gespiegelten Reihe (fibot-Short-Seite)
    up, dn = lo['ch'], sh['ch']
    rising = up['valid'] & (up['slope'] > 0)
    falling = (dn['valid'] & (dn['slope'] >= 0)) | (up['valid'] & (up['slope'] <= 0))
    state = np.where(rising & ~falling, 'steigend', np.where(falling, 'flach/fallend', 'unklar'))
    n = len(c)
    thrL, thrS = df[tr.thr_col('L', 0.9)].to_numpy(), df[tr.thr_col('S', 0.9)].to_numpy()
    pL, pS = df['pL'].to_numpy(), df['pS'].to_numpy()
    fills = []
    for side, pl, sign in (('long', lo, 1), ('short', sh, -1)):
        for i in np.flatnonzero(pl['fill'][:n]):
            if df.index[i] < START:
                continue
            limit, sl, tp = sign * pl['limit'][i], sign * pl['sl'][i], sign * pl['tp'][i]
            ef = tr.fill_price(limit, side, True)
            if pl['instant'][i]:
                ex_raw, j = sl, i
            else:
                ex = tr.simulate_exit(o, h, l, c, i, side, sl, tp, HOLD)
                if ex is None:
                    continue
                j, ex_raw = ex[0], ex[1]
            pnl, _ = tr.trade_pnl(side, 1.0, ef, ex_raw)
            prev = i - 1
            ann_ok = (np.isfinite(pL[prev]) and
                      ((side == 'long' and pL[prev] >= thrL[prev]) or (side == 'short' and pS[prev] >= thrS[prev])))
            fills.append({'coins': (coin,), 'time': df.index[i] + TF, 'exit_time': df.index[j] + TF,
                          'R': pnl / abs(ef - sl), 'risk': 1.0, 'ann': bool(ann_ok), 'side': side})
    return coin, pd.Series(state, index=df.index), fills


def tstat(x):
    x = np.asarray(x)
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def main():
    with open(TREND_EVENTS, 'rb') as f:
        trend = pickle.load(f)
    coins = trend['coins']
    res = Parallel(n_jobs=8)(delayed(fibot_geometry)(c) for c in coins)
    states = {c: s for c, s, _ in res}
    fills = [x for _, _, fl in res for x in fl]
    share = pd.concat([s[s.index >= START] for s in states.values()]).value_counts(normalize=True) * 100
    print('fibot-Kanalzustand der 4h-Kerzen seit 2023:', ', '.join(f"{k} {v:.1f} %" for k, v in share.items()))

    trend_ev = []
    for e in trend['events']:
        if e['exit_time'] is None:
            continue
        ef = tr.fill_price(e['signal_close'], e['side'], True)
        pnl, _ = tr.trade_pnl(e['side'], 1.0, ef, e['exit_raw'])
        trend_ev.append({'time': e['time'], 'exit_time': e['exit_time'], 'coins': (e['coin'],),
                         'R': pnl / abs(ef - e['sl']), 'risk': 0.95, 'coin': e['coin']})
    now = pd.Timestamp.now(tz='UTC')

    def best(ev):
        bb = None
        for sc in np.round(np.arange(0.1, 4.01, 0.05), 2):
            st = simulate(ev, None, scale=float(sc))
            if st['dd'] <= 30 and (bb is None or st['end'] > bb[1]['end']):
                bb = (float(sc), st)
        return bb
    b_trend = best(trend_ev)
    ts = simulate(trend_ev, None, start=SIDE_A, end=SIDE_B)
    to = simulate(trend_ev, None, start=OOS)

    # ---------------- G1 ----------------
    st_of = lambda e: states[e['coin']].get(e['time'] - TF, 'unklar')
    by = {}
    for e in trend_ev:
        by.setdefault(st_of(e), []).append(e['R'])
    print('\nG1 Trend-Signale nach fibot-Kanalzustand: ' +
          ' | '.join(f"{k}: n {len(v)} avgR {np.mean(v):+.3f}" for k, v in by.items()))
    kept = [e for e in trend_ev if st_of(e) != 'flach/fallend']
    dropped = [e for e in trend_ev if st_of(e) == 'flach/fallend']
    rk = np.mean([e['R'] for e in kept])
    rd = np.mean([e['R'] for e in dropped]) if dropped else float('nan')
    b_f = best(kept)
    fs, fo = simulate(kept, None, start=SIDE_A, end=SIDE_B), simulate(kept, None, start=OOS)
    print(f"G1 DD<=30 seit 2023: ohne {cagr(b_trend[1], START, now):+.1f} %/J | mit Filter {cagr(b_f[1], START, now):+.1f} %/J"
          f" | Seitwaerts ohne {ts['pnl']:+.1f} % / mit {fs['pnl']:+.1f} % | OOS ohne {to['pnl']:+.1f} % / mit {fo['pnl']:+.1f} %")
    g1 = {'k1': b_f[1]['end'] > b_trend[1]['end'], 'k2': fs['end'] > ts['end'], 'k3': fo['end'] >= to['end'],
          'k4': bool(dropped) and rd < rk}

    # ---------------- G2 ----------------
    A = np.array([x['R'] for x in fills])
    G = [x for x in fills if x['ann']]
    Rg = np.array([x['R'] for x in G])
    q = pd.Series(Rg, index=pd.DatetimeIndex([x['time'] for x in G])).groupby(pd.Grouper(freq='QE')).mean() if len(G) else pd.Series(dtype=float)
    tD = (Rg.mean() - A.mean()) / np.sqrt(Rg.var(ddof=1) / len(Rg) + A.var(ddof=1) / len(A)) if len(Rg) > 1 else 0
    print(f"\nG2 alle fibot-Fills: n {len(A)} avgR {A.mean():+.3f} (t {tstat(A):+.2f}) | mit ANN Top 10 %: n {len(Rg)} "
          f"avgR {Rg.mean() if len(Rg) else float('nan'):+.3f} (t {tstat(Rg):+.2f}), Quartale positiv "
          f"{int((q > 0).sum())}/{len(q)} | Differenz t {tD:+.2f}")
    o2 = simulate(G, 1.0, start=OOS)
    cs = simulate(trend_ev + G, None, start=SIDE_A, end=SIDE_B)
    b_c = best(trend_ev + G)
    print(f"G2 OOS 26 W: {o2['pnl']:+.1f} % DD {o2['dd']:.1f} n {o2['n']} | Seitwaerts Trend {ts['pnl']:+.1f} % / "
          f"Trend+G2 {cs['pnl']:+.1f} % | DD<=30: Trend {cagr(b_trend[1], START, now):+.1f} %/J / Trend+G2 "
          + (f"{cagr(b_c[1], START, now):+.1f} %/J" if b_c else '-'))
    g2 = {'k1': len(Rg) > 1 and Rg.mean() > 0 and tstat(Rg) > 2 and (q > 0).mean() >= 0.6, 'k2': tD > 2,
          'k3': o2['pnl'] >= 10 and o2['dd'] <= 30 and o2['n'] >= 30,
          'k4': cs['end'] > ts['end'] and bool(b_c) and b_c[1]['end'] > b_trend[1]['end']}
    for name, ok in (('G1', g1), ('G2', g2)):
        print(f"{name}:", {k: ('ok' if v else 'NICHT erfuellt') for k, v in ok.items()},
              '=> BESTANDEN' if all(ok.values()) else '=> NICHT BESTANDEN')
    with open(os.path.join(ROOT, 'forschung', 'fibo_geometrie', 'result.json'), 'w', encoding='utf-8') as f:
        json.dump({'G1': g1, 'G2': g2, 'g1_R': {k: [len(v), float(np.mean(v))] for k, v in by.items()},
                   'g2_all': [len(A), float(A.mean())], 'g2_ann': [len(Rg), float(Rg.mean()) if len(Rg) else None]},
                  f, indent=2, default=float)


if __name__ == '__main__':
    main()
