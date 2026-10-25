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


# --- Saubere Linien nach fibot-Regeln (forschung/kanal_varianten V9) ----------------------------------------
LINE_NAMES = ['lsup_ok', 'lsup_slope', 'lsup_dist', 'lsup_touch', 'lsup_age',
              'lres_ok', 'lres_slope', 'lres_dist', 'lres_touch', 'lres_age',
              'chan_width', 'chan_pos', 'brk_lsup', 'brk_lres']
LINE_RULES = {'window': 150, 'max_pivots': 8, 'touch_atr': 0.3, 'break_atr': 0.5, 'min_span': 20, 'max_dist_atr': 6.0}


def best_line(i, piv, price, closes, atr_i, k, side, rules=LINE_RULES):
    """Beste gueltige Linie an Kerze i (nur bestaetigte Pivots bis i-k), Regeln wie fibot find_channel, jede Steigung.
    side='sup': Linie durch Pivot-Tiefs, kein Schlusskurs/Pivot darunter; side='res' gespiegelt.
    Rueckgabe (p1, p2, slope, intercept, touches) oder None."""
    lo_win = i - rules['window']
    cand = piv[(piv >= lo_win) & (piv <= i - k)][-rules['max_pivots']:]
    if len(cand) < 2:
        return None
    sgn = 1.0 if side == 'sup' else -1.0           # Widerstand = Unterstuetzung der gespiegelten Reihe
    tol_b, tol_t = rules['break_atr'] * atr_i, rules['touch_atr'] * atr_i
    best = None
    for x in range(len(cand) - 1):
        p1 = cand[x]
        if i - p1 < rules['min_span']:
            break                                   # spaetere Anker sind noch juenger
        seg_c = sgn * closes[p1:i + 1]
        idx = np.arange(p1, i + 1)
        lp = cand[cand >= p1]
        for y in range(x + 1, len(cand)):
            p2 = cand[y]
            if p2 - p1 < 2 * k:
                continue
            slope = (price[p2] - price[p1]) / (p2 - p1)
            icpt = price[p1] - slope * p1
            if abs(closes[i] - (icpt + slope * i)) > rules['max_dist_atr'] * atr_i:   # zu weit weg: fuer die Lage irrelevant
                continue
            if np.any(seg_c < sgn * (icpt + slope * idx) - tol_b):    # Schlusskurs-Bruch
                continue
            dev = sgn * (price[lp] - (icpt + slope * lp))
            if np.any(dev < -tol_b):                                  # Pivot jenseits der Linie
                continue
            touches = int(np.count_nonzero(np.abs(dev) <= tol_t))
            if touches < 2:
                continue
            key = (touches, p2)
            if best is None or key > best[0]:
                best = (key, (int(p1), int(p2), float(slope), float(icpt), touches))
    return best[1] if best else None


def clean_lines(h, l, c, atr, k, prefix, rules=LINE_RULES):
    n = len(c)
    out = {f"{prefix}{name}": np.zeros(n) for name in LINE_NAMES}
    for key in ('lsup_dist', 'lres_dist'):
        out[f"{prefix}{key}"][:] = 10.0
    ph, pl = pivots(h, k, 'high'), pivots(l, k, 'low')
    prev = {'sup': None, 'res': None}
    for i in range(n):
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            prev = {'sup': None, 'res': None}
            continue
        cur = {}
        for side, piv, price in (('sup', pl, l), ('res', ph, h)):
            pv = prev[side]                         # Ausbruch: gestern gueltige Linie heute gebrochen
            if pv is not None:
                lv = pv[3] + pv[2] * i
                if (side == 'sup' and c[i] < lv - rules['break_atr'] * a) or \
                        (side == 'res' and c[i] > lv + rules['break_atr'] * a):
                    out[f"{prefix}brk_l{side}"][i] = 1.0
            ln = best_line(i, piv, price, c, a, k, side, rules)
            cur[side] = ln
            if ln is not None:
                p1, p2, slope, icpt, touches = ln
                lv = icpt + slope * i
                out[f"{prefix}l{side}_ok"][i] = 1.0
                out[f"{prefix}l{side}_slope"][i] = slope / a
                out[f"{prefix}l{side}_dist"][i] = (c[i] - lv) / a
                out[f"{prefix}l{side}_touch"][i] = touches
                out[f"{prefix}l{side}_age"][i] = (i - p1) / rules['window']
        if cur['sup'] is not None and cur['res'] is not None:
            s_ = cur['sup'][3] + cur['sup'][2] * i
            r_ = cur['res'][3] + cur['res'][2] * i
            if r_ > s_:
                out[f"{prefix}chan_width"][i] = (r_ - s_) / a
                out[f"{prefix}chan_pos"][i] = (c[i] - s_) / (r_ - s_)
        prev = cur
    return out
