"""Auswertung PREREG.md (reine Geometrie). Braucht die Trend-Signale:
  JAEGER_PROFILE=trend python forschung/seitwaerts/evaluate.py dump-trend
  python forschung/geometrie/evaluate.py
"""
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'forschung', 'pairs'))
os.chdir(ROOT)

from jaegerbot.strategy import trade_rules as tr
from evaluate import simulate, cagr, TREND_EVENTS   # R-Simulator aus der Pairs-Auswertung (gegen Bot geprueft)

N, EDGE, SL_W, HOLD = 50, 0.15, 0.25, 12
START = pd.Timestamp('2023-01-01', tz='UTC')
OOS = pd.Timestamp.now(tz='UTC').normalize() - pd.Timedelta(weeks=26)
SIDE_A, SIDE_B = pd.Timestamp('2024-03-15', tz='UTC'), pd.Timestamp('2026-07-31', tz='UTC')
TF = pd.Timedelta('4h')
X = np.arange(N, dtype=float)
XC = X - X.mean()


def _touches(mask):
    n, last = 0, -10
    for k in np.flatnonzero(mask):
        if k - last >= 3:
            n += 1
            last = k
    return n


def geometry(coin):
    df = pd.read_pickle(os.path.join(ROOT, 'data', 'features', '4h', f"{coin}.pkl"))[['open', 'high', 'low', 'close']]
    o, h, l, c = (df[k].to_numpy() for k in ('open', 'high', 'low', 'close'))
    n = len(c)
    out = {k: np.full(n, np.nan) for k in ('side', 'mid', 'up', 'lo', 'W')}
    out['side'] = np.zeros(n, dtype=bool)
    for i in range(N - 1, n):
        sl = slice(i - N + 1, i + 1)
        cc = c[sl]
        m = (XC * (cc - cc.mean())).sum() / (XC ** 2).sum()
        center = cc.mean() + m * XC
        up = center + (h[sl] - center).max()
        lo = center + (l[sl] - center).min()
        W = up[-1] - lo[-1]
        if W <= 0:
            continue
        D = abs(m) * N
        path = np.abs(np.diff(cc)).sum()
        ER = abs(cc[-1] - cc[0]) / path if path > 0 else 0
        tu = _touches(h[sl] >= up - EDGE * W)
        tl = _touches(l[sl] <= lo + EDGE * W)
        out['side'][i] = D < 0.5 * W and ER < 0.3 and tu >= 2 and tl >= 2
        out['mid'][i], out['up'][i], out['lo'][i], out['W'][i] = center[-1], up[-1], lo[-1], W
    g = pd.DataFrame(out, index=df.index)
    return coin, df, g


def range_trades(coin, df, g, rng=None):
    """H1-Trades; rng gesetzt -> Kontrollgruppe (zufaellige Kerze im Kanal, zufaellige Seite)."""
    o, h, l, c = (df[k].to_numpy() for k in ('open', 'high', 'low', 'close'))
    idx = df.index
    side_mask = g['side'].to_numpy() & (idx >= START)
    trades, i = [], 0
    cand = np.flatnonzero(side_mask)
    if rng is not None:
        k = int(rng.integers(1, 4))   # ~ gleiche Groessenordnung, wird unten auf echte Anzahl begrenzt
    for i in cand:
        W, mid, up, lo = g['W'].iat[i], g['mid'].iat[i], g['up'].iat[i], g['lo'].iat[i]
        if rng is None:
            if c[i] <= lo + EDGE * W:
                side = 'long'
            elif c[i] >= up - EDGE * W:
                side = 'short'
            else:
                continue
        else:
            side = 'long' if rng.random() < 0.5 else 'short'
        sl = lo - SL_W * W if side == 'long' else up + SL_W * W
        tp = mid
        if (side == 'long' and not sl < c[i] < tp) or (side == 'short' and not tp < c[i] < sl):
            continue
        ex = tr.simulate_exit(o, h, l, c, i, side, sl, tp, HOLD)
        if ex is None:
            continue
        ef = tr.fill_price(c[i], side, True)
        pnl, _ = tr.trade_pnl(side, 1.0, ef, ex[1])
        trades.append({'coins': (coin,), 'time': idx[i] + TF, 'exit_time': idx[ex[0]] + TF,
                       'R': pnl / abs(ef - sl), 'risk': 1.0, 'i': i})
    return trades


def tstat(x):
    x = np.asarray(x)
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def main():
    with open(TREND_EVENTS, 'rb') as f:
        trend = pickle.load(f)
    coins = trend['coins']
    geo = {c: (df, g) for c, df, g in Parallel(n_jobs=8)(delayed(geometry)(c) for c in coins)}
    share = np.mean([g.loc[g.index >= START, 'side'].mean() for _, g in geo.values()])
    print(f"Anteil Seitwaertskanal-Kerzen seit 2023: {share * 100:.1f} %")

    # ---------------- H1 ----------------
    h1 = [t for c in coins for t in range_trades(c, *geo[c])]
    rng = np.random.default_rng(0)
    ctrl = []
    for c in coins:
        real = [t for t in h1 if t['coins'][0] == c]
        pool = range_trades(c, *geo[c], rng=rng)
        if real and pool:
            pick = rng.choice(len(pool), min(len(pool), 3 * len(real)), replace=False)
            ctrl += [pool[k] for k in pick]
    R, C = np.array([t['R'] for t in h1]), np.array([t['R'] for t in ctrl])
    q = pd.Series(R, index=pd.DatetimeIndex([t['time'] for t in h1])).groupby(pd.Grouper(freq='QE')).mean()
    tD = (R.mean() - C.mean()) / np.sqrt(R.var(ddof=1) / len(R) + C.var(ddof=1) / len(C))
    print(f"\nH1 Kanal-Rand: {len(R)} Signale, netto avgR {R.mean():+.3f} (t {tstat(R):+.2f}), WR {np.mean(R > 0) * 100:.1f} %, "
          f"Quartale positiv {int((q > 0).sum())}/{len(q)} | Kontrolle {C.mean():+.3f} R (n={len(C)}), Differenz t {tD:+.2f}")
    o1 = simulate(h1, 1.0, start=OOS)
    print(f"H1 OOS 26 W: {o1['pnl']:+.1f} % DD {o1['dd']:.1f} n {o1['n']}")
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
    b_comb = best(trend_ev + h1)
    ts = simulate(trend_ev, None, start=SIDE_A, end=SIDE_B)
    cs = simulate(trend_ev + h1, None, start=SIDE_A, end=SIDE_B)
    print(f"H1 Seitwaertsfenster: Trend {ts['pnl']:+.1f} % | Trend+Kanal {cs['pnl']:+.1f} % (DD {cs['dd']:.1f})")
    print(f"H1 DD<=30 seit 2023: Trend {cagr(b_trend[1], START, now):+.1f} %/J | Trend+Kanal "
          + (f"{cagr(b_comb[1], START, now):+.1f} %/J" if b_comb else 'kein Faktor im DD-Limit'))
    h1_ok = {'k1': R.mean() > 0 and tstat(R) > 2 and (q > 0).mean() >= 0.6, 'k2': tD > 2,
             'k3': o1['pnl'] >= 10 and o1['dd'] <= 30 and o1['n'] >= 30,
             'k4': cs['end'] > ts['end'] and bool(b_comb) and b_comb[1]['end'] > b_trend[1]['end']}

    # ---------------- H2 ----------------
    def in_channel(e):
        df, g = geo[e['coin']]
        ts_ = e['time'] - TF
        return bool(g['side'].get(ts_, False))
    flags = [in_channel(e) for e in trend_ev]
    kept = [e for e, f in zip(trend_ev, flags) if not f]
    dropped = [e for e, f in zip(trend_ev, flags) if f]
    rk, rd = np.mean([e['R'] for e in kept]), (np.mean([e['R'] for e in dropped]) if dropped else float('nan'))
    print(f"\nH2 Trend-Signale im Seitwaertskanal: {len(dropped)} von {len(trend_ev)} | avgR ausgelassen {rd:+.3f} vs behalten {rk:+.3f}")
    b_filt = best(kept)
    fs = simulate(kept, None, start=SIDE_A, end=SIDE_B)
    fo, to = simulate(kept, None, start=OOS), simulate(trend_ev, None, start=OOS)
    print(f"H2 DD<=30 seit 2023: ohne Filter {cagr(b_trend[1], START, now):+.1f} %/J | mit Filter {cagr(b_filt[1], START, now):+.1f} %/J "
          f"(Faktor {b_filt[0]})")
    print(f"H2 Seitwaertsfenster 0,95 %: ohne {ts['pnl']:+.1f} % | mit {fs['pnl']:+.1f} %   OOS 26 W: ohne {to['pnl']:+.1f} % | mit {fo['pnl']:+.1f} %")
    h2_ok = {'k1': b_filt[1]['end'] > b_trend[1]['end'], 'k2': fs['end'] > ts['end'], 'k3': fo['end'] >= to['end'],
             'k4': bool(dropped) and rd < rk}
    for name, ok in (('H1', h1_ok), ('H2', h2_ok)):
        print(f"{name}:", {k: ('ok' if v else 'NICHT erfuellt') for k, v in ok.items()},
              '=> BESTANDEN' if all(ok.values()) else '=> NICHT BESTANDEN')
    with open(os.path.join(ROOT, 'forschung', 'geometrie', 'result.json'), 'w', encoding='utf-8') as f:
        json.dump({'channel_share': share, 'H1': h1_ok, 'H2': h2_ok, 'h1_avgR': R.mean(), 'h1_t': tstat(R),
                   'h2_dropped': len(dropped), 'h2_R_dropped': rd, 'h2_R_kept': rk}, f, indent=2, default=float)


if __name__ == '__main__':
    main()
