# src/jaegerbot/strategy/human_geometry.py
"""Chart-Geometrie, wie ein Mensch sie liest (forschung/kanal_varianten V7): Marktstruktur, Trendlinien,
Dreiecke/Keile, horizontale Zonen, Doppel-Hoch/-Tief, Ausbruch/Retest.

Kausal: ein Pivot an Kerze j (Extrem ueber j-k..j+k) zaehlt erst ab Kerze j+k. Nur Pivots aus dem Rueckblick
`lookback` werden benutzt -> der Wert haengt nur von den letzten lookback+k Kerzen ab (live = Backtest).
"""
import numpy as np

NAMES = ['struct', 'tlu_slope', 'tlu_dist', 'tll_slope', 'tll_dist', 'converg', 'res_dist', 'res_touch',
         'sup_dist', 'sup_touch', 'dtop', 'dbot', 'brk_up', 'brk_dn', 'retest_up', 'retest_dn']
NO_ZONE = 10.0     # Abstand (in ATR), wenn es keine Zone/Linie gibt


def pivots(arr, k, mode):
    n = len(arr)
    out = []
    for j in range(k, n - k):
        win = arr[j - k:j + k + 1]
        if (mode == 'high' and arr[j] == win.max()) or (mode == 'low' and arr[j] == win.min()):
            if not out or out[-1] != j - 1 or arr[j] != arr[out[-1]]:   # gleiche Extreme direkt nacheinander nur einmal
                out.append(j)
    return np.array(out, dtype=np.int64)


def _zones(levels, tol):
    """Pivot-Preise buendeln (Abstand <= tol) -> Liste (Zonenpreis, Beruehrungen)."""
    if len(levels) == 0:
        return []
    lv = np.sort(levels)
    zones, group = [], [lv[0]]
    for x in lv[1:]:
        if x - group[-1] <= tol:
            group.append(x)
        else:
            zones.append((float(np.mean(group)), len(group)))
            group = [x]
    zones.append((float(np.mean(group)), len(group)))
    return zones


def human_geometry(h, l, c, atr, k, lookback, prefix):
    n = len(c)
    out = {f"{prefix}{name}": np.zeros(n) for name in NAMES}
    for key in ('res_dist', 'sup_dist', 'tlu_dist', 'tll_dist'):
        out[f"{prefix}{key}"][:] = NO_ZONE
    ph, pl = pivots(h, k, 'high'), pivots(l, k, 'low')
    for i in range(n):
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        lo_j, hi_j = i - lookback, i - k
        P = ph[(ph >= lo_j) & (ph <= hi_j)]
        Q = pl[(pl >= lo_j) & (pl <= hi_j)]
        o = {}
        s = 0
        if len(P) >= 2:
            s += 1 if h[P[-1]] > h[P[-2]] else -1
            su = (h[P[-1]] - h[P[-2]]) / (P[-1] - P[-2])
            line = h[P[-1]] + su * (i - P[-1])
            o['tlu_slope'], o['tlu_dist'] = su / a, (c[i] - line) / a
            o['dtop'] = float(abs(h[P[-1]] - h[P[-2]]) <= 0.5 * a)
        if len(Q) >= 2:
            s += 1 if l[Q[-1]] > l[Q[-2]] else -1
            sl = (l[Q[-1]] - l[Q[-2]]) / (Q[-1] - Q[-2])
            line = l[Q[-1]] + sl * (i - Q[-1])
            o['tll_slope'], o['tll_dist'] = sl / a, (c[i] - line) / a
            o['dbot'] = float(abs(l[Q[-1]] - l[Q[-2]]) <= 0.5 * a)
        if len(P) >= 2 and len(Q) >= 2:
            o['converg'] = o['tlu_slope'] - o['tll_slope']
        o['struct'] = s
        zones = _zones(np.concatenate([h[P], l[Q]]), 0.5 * a)
        above = [(z, t) for z, t in zones if z > c[i]]
        below = [(z, t) for z, t in zones if z <= c[i]]
        if above:
            z, t = min(above)
            o['res_dist'], o['res_touch'] = (z - c[i]) / a, t
        if below:
            z, t = max(below)
            o['sup_dist'], o['sup_touch'] = (c[i] - z) / a, t
        strong = [z for z, t in zones if t >= 2]
        if strong and i >= 10:
            prev3 = c[i - 3:i]
            prev10 = c[i - 10:i]
            o['brk_up'] = float(any(c[i] > z and prev3.min() <= z for z in strong))
            o['brk_dn'] = float(any(c[i] < z and prev3.max() >= z for z in strong))
            o['retest_up'] = float(any(prev10.min() <= z < prev10.max() and z < c[i] <= z + 0.5 * a for z in strong))
            o['retest_dn'] = float(any(prev10.min() < z <= prev10.max() and z - 0.5 * a <= c[i] < z for z in strong))
        for key, v in o.items():
            out[f"{prefix}{key}"][i] = v
    return out
