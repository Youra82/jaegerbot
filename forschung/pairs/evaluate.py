"""Auswertung PREREG.md (Pair-Trading). Stufe 1 braucht die gespeicherten Trend-Signale:
  JAEGER_PROFILE=trend python forschung/seitwaerts/evaluate.py dump-trend
  python forschung/pairs/evaluate.py
"""
import os
import sys
import json
import pickle
import itertools
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)

from jaegerbot.utils.data import load_data
from jaegerbot.strategy import trade_rules as tr

TF = pd.Timedelta('4h')
FORM = 720            # 120 Tage Formation
T_MAX, HL_MAX, MAX_PAIRS = -3.4, 30, 20
Z_IN, Z_STOP, HOLD = 2.0, 4.0, 30
COST_SIDE = 0.0011    # je Seite und Coin
START = pd.Timestamp('2023-01-01', tz='UTC')
OOS = pd.Timestamp.now(tz='UTC').normalize() - pd.Timedelta(weeks=26)
SIDE_A, SIDE_B = pd.Timestamp('2024-03-15', tz='UTC'), pd.Timestamp('2026-07-31', tz='UTC')
TREND_EVENTS = os.path.join(os.environ.get('TMP', '/tmp'), 'jaeger_trend_events.pkl')


def closes(coins):
    df = pd.concat({c: load_data(f"{c}/USDT:USDT", '4h')['close'] for c in coins}, axis=1).sort_index()
    return np.log(df)


def formation(L, m_idx):
    """Paare fuer den Monat ab Index m_idx (nur Kerzen davor)."""
    W = L.iloc[m_idx - FORM:m_idx]
    ok = W.columns[W.notna().all()]
    out = []
    for a, b in itertools.combinations(ok, 2):
        x, y = W[b].to_numpy(), W[a].to_numpy()
        xm, ym = x.mean(), y.mean()
        beta = ((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum()
        if not 0.2 <= beta <= 5:
            continue
        s = y - beta * x
        ds, sl = np.diff(s), s[:-1]
        X = np.column_stack([np.ones_like(sl), sl])
        coef, res, *_ = np.linalg.lstsq(X, ds, rcond=None)
        bcoef = coef[1]
        if bcoef >= 0:
            continue
        resid = ds - X @ coef
        se = np.sqrt(resid.var(ddof=2) / ((sl - sl.mean()) ** 2).sum())
        t = bcoef / se
        hl = -np.log(2) / np.log(1 + bcoef) if bcoef > -1 else 0
        if t < T_MAX and hl <= HL_MAX:
            out.append((t, a, b, beta, s.mean(), s.std()))
    out.sort()
    return out[:MAX_PAIRS]


def trade_pair(L, a, b, beta, mu, sd, i0, i1, entry_mask=None):
    """Trades im Bereich [i0, i1) (Einstiege), Ausstiege duerfen spaeter liegen.
    entry_mask: optional Liste von Einstiegsindizes (Kontrollgruppe) statt |z| >= 2."""
    s = (L[a] - beta * L[b]).to_numpy()
    z = (s - mu) / sd
    idx = L.index
    trades, i = [], i0
    cost_R = COST_SIDE * 2 * (1 + beta) / (2 * sd)
    cand = set(entry_mask) if entry_mask is not None else None
    while i < i1:
        if not np.isfinite(z[i]):
            i += 1
            continue
        go = (i in cand) if cand is not None else abs(z[i]) >= Z_IN
        if not go or z[i] == 0:
            i += 1
            continue
        d = -1 if z[i] > 0 else 1          # Spread short wenn z > 0
        j = i + 1
        while j < len(z) and j - i <= HOLD:
            if not np.isfinite(z[j]):
                break
            if (d == -1 and z[j] <= 0) or (d == 1 and z[j] >= 0) or abs(z[j]) >= Z_STOP or j - i == HOLD:
                break
            j += 1
        if j >= len(z) or not np.isfinite(z[min(j, len(z) - 1)]):
            break
        R = d * (s[j] - s[i]) / (2 * sd) - cost_R
        trades.append({'pair': f"{a}/{b}", 'coins': (a, b), 'time': idx[i] + TF, 'exit_time': idx[j] + TF, 'R': R})
        i = j + 1 if cand is None else j + 1
    return trades


def simulate(events, risk, max_open=20, start=None, end=None, scale=None):
    """R-basiertes Portfolio: pnl = Equity bei Einstieg x Risiko x R; ein Coin nur einmal; max. Positionen."""
    ev = sorted((e for e in events if (start is None or e['time'] >= start) and (end is None or e['time'] <= end)),
                key=lambda e: e['time'])
    eq, peak, dd, open_, n, Rs = 100.0, 100.0, 0.0, [], 0, []
    for e in ev:
        open_.sort(key=lambda p: p['exit_time'])
        while open_ and open_[0]['exit_time'] <= e['time']:
            p = open_.pop(0)
            eq += p['pnl']
            peak = max(peak, eq)
            dd = max(dd, 1 - eq / peak)
        if len(open_) >= max_open or any(set(p['coins']) & set(e['coins']) for p in open_):
            continue
        r = (e.get('risk', risk)) * (scale or 1.0)
        open_.append({'exit_time': e['exit_time'], 'coins': e['coins'], 'pnl': eq * r / 100 * e['R']})
        n += 1
        Rs.append(e['R'])
    for p in sorted(open_, key=lambda p: p['exit_time']):
        eq += p['pnl']
        peak = max(peak, eq)
        dd = max(dd, 1 - eq / peak)
    return {'end': eq, 'pnl': eq - 100, 'dd': dd * 100, 'n': n, 'avgR': float(np.mean(Rs)) if Rs else 0.0}


def cagr(st, a, b):
    return ((st['end'] / 100) ** (365.25 / (b - a).days) - 1) * 100 if st['end'] > 0 else -100


def main():
    with open(TREND_EVENTS, 'rb') as f:
        trend = pickle.load(f)
    coins = trend['coins']
    L = closes(coins)
    idx = L.index
    months = pd.date_range(START, idx[-1], freq='MS')
    rng = np.random.default_rng(0)
    pairs_trades, ctrl = [], []
    n_sel = []
    for m in months:
        m_idx = idx.searchsorted(m)
        nxt = idx.searchsorted(m + pd.offsets.MonthBegin(1))
        sel = formation(L, m_idx)
        n_sel.append(len(sel))
        for t, a, b, beta, mu, sd in sel:
            tt = trade_pair(L, a, b, beta, mu, sd, m_idx, nxt)
            pairs_trades += tt
            if tt:   # Kontrolle: gleich viele zufaellige Einstiege im selben Paar/Monat
                pool = np.arange(m_idx, nxt)
                ctrl += trade_pair(L, a, b, beta, mu, sd, m_idx, nxt,
                                   entry_mask=sorted(rng.choice(pool, min(len(pool), 3 * len(tt)), replace=False)))
    R = np.array([t['R'] for t in pairs_trades])
    C = np.array([t['R'] for t in ctrl])
    tR = R.mean() / (R.std(ddof=1) / np.sqrt(len(R)))
    tD = (R.mean() - C.mean()) / np.sqrt(R.var(ddof=1) / len(R) + C.var(ddof=1) / len(C))
    print(f"Paare je Monat: Ø {np.mean(n_sel):.1f} | Pair-Trades {len(R)} | avgR {R.mean():+.3f} (t {tR:+.2f}) | "
          f"WR {np.mean(R > 0) * 100:.1f} % | Kontrolle {C.mean():+.3f} R (n={len(C)}), Differenz t {tD:+.2f}")

    # Trend-Ereignisse in R umrechnen (gleiche Funktionen wie der Bot)
    trend_ev = []
    for e in trend['events']:
        if e['exit_time'] is None:
            continue
        ef = tr.fill_price(e['signal_close'], e['side'], True)
        pnl, _ = tr.trade_pnl(e['side'], 1.0, ef, e['exit_raw'])
        trend_ev.append({'time': e['time'], 'exit_time': e['exit_time'], 'coins': (e['coin'],),
                         'R': pnl / abs(ef - e['sl']), 'risk': 0.95})
    for e in pairs_trades:
        e['risk'] = 1.0
    now = pd.Timestamp.now(tz='UTC')
    t_only = simulate(trend_ev, 0.95)
    print(f"Pruefung Simulator: Trend allein 0,95 % -> {cagr(t_only, START, now):+.1f} %/J DD {t_only['dd']:.1f} "
          f"(Bot-Simulator: +34,5 %/J, DD 29,9)")

    res = {}
    o = simulate(pairs_trades, 1.0, start=OOS)
    print(f"K1 OOS ab {OOS:%d.%m.%Y}: Pairs {o['pnl']:+.1f} % DD {o['dd']:.1f} n {o['n']} avgR {o['avgR']:+.3f}")
    res['k1'] = o['pnl'] >= 10 and o['dd'] <= 30 and o['n'] >= 30
    res['k2'] = tD > 2
    ts = simulate(trend_ev, 0.95, start=SIDE_A, end=SIDE_B)
    cs = simulate(trend_ev + pairs_trades, None, start=SIDE_A, end=SIDE_B)
    ps = simulate(pairs_trades, 1.0, start=SIDE_A, end=SIDE_B)
    print(f"K3a Seitwaertsfenster: Trend {ts['pnl']:+.1f} % DD {ts['dd']:.1f} | Pairs allein {ps['pnl']:+.1f} % DD {ps['dd']:.1f} "
          f"| Trend+Pairs {cs['pnl']:+.1f} % DD {cs['dd']:.1f}")
    res['k3a'] = cs['end'] > ts['end']
    best = {}
    for name, ev in (('Trend', trend_ev), ('Pairs', pairs_trades), ('Trend+Pairs', trend_ev + pairs_trades)):
        bb = None
        for sc in np.round(np.arange(0.1, 4.01, 0.05), 2):
            st = simulate(ev, None, scale=float(sc))
            if st['dd'] <= 30 and (bb is None or st['end'] > bb[1]['end']):
                bb = (float(sc), st)
        best[name] = bb
        if bb:
            print(f"K3b {name:12s}: Faktor {bb[0]:.2f} -> {cagr(bb[1], START, now):+.1f} %/J DD {bb[1]['dd']:.1f} n {bb[1]['n']}")
    res['k3b'] = bool(best['Trend+Pairs'] and best['Trend+Pairs'][1]['end'] > best['Trend'][1]['end'])
    res['k3'] = res['k3a'] and res['k3b']
    q = pd.Series(R, index=pd.DatetimeIndex([t['time'] for t in pairs_trades])).groupby(pd.Grouper(freq='QE')).mean()
    pos = float((q > 0).mean())
    print(f"K4 Robustheit: avgR {R.mean():+.3f}, t {tR:+.2f}, Quartale positiv {int((q > 0).sum())}/{len(q)} ({pos * 100:.0f} %)")
    res['k4'] = R.mean() > 0 and tR > 2 and pos >= 0.6
    passed = all(res[k] for k in ('k1', 'k2', 'k3', 'k4'))
    print('\nErgebnis:', {k: ('ok' if v else 'NICHT erfuellt') for k, v in res.items()},
          '=> BESTANDEN' if passed else '=> NICHT BESTANDEN')
    with open(os.path.join(ROOT, 'forschung', 'pairs', 'result.json'), 'w', encoding='utf-8') as f:
        json.dump({'criteria': res, 'passed': passed, 'avgR': R.mean(), 't': tR, 'control': C.mean(), 't_diff': tD,
                   'oos': o, 'quarters_positive': pos}, f, indent=2, default=float)


if __name__ == '__main__':
    main()
