"""Beispiele aus dem V7/V9-Test: echte OOS-Trades mit beiden Linienarten (V7 gestrichelt, V9 durchgezogen).

Braucht %TMP%/kv_mensch.pkl und kv_kv9.pkl aus forschung/kanal_varianten/evaluate_v9.py."""
import os, sys, pickle
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, 'src')
import numpy as np, pandas as pd, ta
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from jaegerbot.utils.data import load_data
from jaegerbot.strategy.human_geometry import pivots, best_line

T = os.environ['TMP']
SURF, INK, INK2, MUT, GRID = '#1a1a19', '#f0efec', '#c3c2b7', '#898781', '#2c2c2a'
ORANGE, AQUA, VIOLET, GOOD, CRIT = '#d95926', '#199e70', '#9085e9', '#0ca30c', '#d03b3b'
OOS = pd.Timestamp.now(tz='UTC').normalize() - pd.Timedelta(weeks=26)
TF = pd.Timedelta('4h')
rng = np.random.default_rng(7)


def pick(events, n_win, n_loss):
    ev = [e for e in events if e['time'] >= OOS]
    win = [e for e in ev if e['R'] > 0.5]
    loss = [e for e in ev if e['R'] < -0.5]
    return [win[i] for i in rng.choice(len(win), min(n_win, len(win)), replace=False)] + \
           [loss[i] for i in rng.choice(len(loss), min(n_loss, len(loss)), replace=False)]


def panel(ax, e, side, label):
    coin = e['coins'][0]
    df = load_data(f'{coin}/USDT:USDT', '4h')
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], 14).average_true_range()
    o, h, l, c, atr = (df[k].to_numpy(float) for k in ('open', 'high', 'low', 'close', 'atr'))
    i = df.index.get_loc(e['time'] - TF)
    j = df.index.get_loc(e['exit_time'] - TF)
    lo = i - 90
    x = np.arange(lo, j + 6)
    xs = x[x < len(c)]
    up = c[xs] >= o[xs]
    ax.set_facecolor(SURF)
    ax.vlines(xs, l[xs], h[xs], color=MUT, lw=0.8)
    ax.bar(xs, np.abs(c[xs] - o[xs]), bottom=np.minimum(c[xs], o[xs]), width=0.7,
           color=np.where(up, '#5b5a55', '#3a3936'), edgecolor=INK2, lw=0.3)
    k = 5
    ph, pl = pivots(h[:i + 1], k, 'high'), pivots(l[:i + 1], k, 'low')
    phc, plc = ph[(ph <= i - k) & (ph >= i - 300)], pl[(pl <= i - k) & (pl >= i - 300)]
    # V7: Linie durch die letzten zwei bestaetigten Pivots (gestrichelt)
    for P, arr, col in ((phc, h, ORANGE), (plc, l, AQUA)):
        if len(P) >= 2 and P[-2] >= lo:
            a_, b_ = P[-2], P[-1]
            s = (arr[b_] - arr[a_]) / (b_ - a_)
            ax.plot([a_, i + 4], [arr[a_], arr[b_] + s * (i + 4 - b_)], color=col, lw=1.3, ls='--', alpha=0.9)
    # V9: saubere Linien (durchgezogen) mit Beruehrungen
    for sd, piv, arr, col in (('res', ph, h, ORANGE), ('sup', pl, l, AQUA)):
        ln = best_line(i, piv, arr, c, atr[i], k, sd)
        if ln is None:
            continue
        p1, p2, s, ic, t = ln
        ax.plot([max(p1, lo), i + 4], [ic + s * max(p1, lo), ic + s * (i + 4)], color=col, lw=2.2)
        cand = piv[(piv >= p1) & (piv <= i - k)]
        hit = cand[np.abs(arr[cand] - (ic + s * cand)) <= 0.3 * atr[i]]
        ax.scatter(hit[hit >= lo], arr[hit[hit >= lo]], color=col, s=28, zorder=5)
    d = atr[i] * 2.0
    sl, tp = (c[i] + d, c[i] - 2 * d) if side == 'short' else (c[i] - d, c[i] + 2 * d)
    ax.hlines([sl], i, j, color=CRIT, lw=1.5, ls=':')
    ax.hlines([tp], i, j, color=GOOD, lw=1.5, ls=':')
    ax.scatter([i], [c[i]], marker='v' if side == 'short' else '^', s=110, color=VIOLET, edgecolor=SURF, lw=1.5, zorder=6)
    res_col = GOOD if e['R'] > 0 else CRIT
    ax.set_title(f"{label} · {coin} {side.upper()} {df.index[i]:%d.%m.%Y %H:%M} → {e['R']:+.2f} R",
                 color=res_col, fontsize=10, loc='left')
    ax.set_xlim(lo - 1, j + 7)
    ymin, ymax = l[lo:j + 6].min(), h[lo:j + 6].max()
    pad = (ymax - ymin) * 0.12
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.tick_params(colors=MUT, labelsize=8)
    ax.set_xticks([])
    for sp in ax.spines.values():
        sp.set_color(GRID)


v7 = pickle.load(open(os.path.join(T, 'kv_mensch.pkl'), 'rb'))['events']
v9 = pickle.load(open(os.path.join(T, 'kv_kv9.pkl'), 'rb'))['events']
sel = [(e, 'short', 'V7 live (Short)') for e in pick(v7, 3, 2)] + [(e, 'long', 'V9 (gewählt: Long)') for e in pick(v9, 2, 1)]
fig, axes = plt.subplots(4, 2, figsize=(15, 15), facecolor=SURF)
for ax, (e, side, lab) in zip(axes.flat, sel):
    panel(ax, e, side, lab)
for ax in list(axes.flat)[len(sel):]:
    ax.axis('off')
fig.text(0.01, 0.995, 'Beispiele aus dem Test (nur OOS, Gewinner und Verlierer zufällig gezogen)', color=INK, fontsize=14,
         fontweight='bold', va='top')
fig.text(0.01, 0.975, 'gestrichelt = V7 (live): Linie durch die letzten zwei bestätigten Hochs/Tiefs   ·   durchgezogen + Punkte = V9: '
         'saubere Linie nach fibot-Regeln   ·   orange = oben, grün = unten   ·   rot/grün gepunktet = SL/TP',
         color=INK2, fontsize=9.5, va='top')
plt.tight_layout(rect=(0, 0, 1, 0.965))
out = os.path.join(os.getcwd(), 'docs', 'images', 'examples_v7_v9.png')
plt.savefig(out, dpi=105, facecolor=SURF)
print(out)
v7o = [e['R'] for e in v7 if e['time'] >= OOS]
v9o = [e['R'] for e in v9 if e['time'] >= OOS]
print(f"OOS-Signale: V7 {len(v7o)} (Ø {np.mean(v7o):+.2f} R, Trefferquote {np.mean(np.array(v7o) > 0) * 100:.0f} %) | "
      f"V9 {len(v9o)} (Ø {np.mean(v9o):+.2f} R, Trefferquote {np.mean(np.array(v9o) > 0) * 100:.0f} %)")
