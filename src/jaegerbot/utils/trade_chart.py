# src/jaegerbot/utils/trade_chart.py
"""Telegram-Chart zu jedem Trade (Eroeffnung und Schliessung) -- Stil wie die README-Bilder.

Nur Anzeige: beeinflusst keine Handelsentscheidung. Der Aufrufer faengt jeden Fehler ab und sendet dann nur Text.
Long: Kerzen, EMA20/50, Einstieg/SL/TP, darunter ANN-Wahrscheinlichkeit gegen die Schwelle.
Short (Profil mensch): zusaetzlich Trendlinien (fibot-Regeln), Netz-Eingabe Hoch-/Tief-Abfolge, Zonen, Kerzenform.
"""
import os
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jaegerbot.strategy import trade_rules as tr
from jaegerbot.strategy import model as mdl
from jaegerbot.strategy.features import FEATURES
from jaegerbot.strategy.human_geometry import pivots, best_line, _zones
from jaegerbot.strategy.candle_patterns import candle_patterns

SURFACE, INK, INK2, MUTED, GRID = '#1a1a19', '#f0efec', '#c3c2b7', '#898781', '#2c2c2a'
BLUE, ORANGE, AQUA, VIOLET = '#3987e5', '#d95926', '#199e70', '#9085e9'
GOOD, CRITICAL, WARNING = '#0ca30c', '#d03b3b', '#fab219'
BEFORE = 70          # Kerzen vor dem Signal im Bild


def _px(v):
    if v >= 1000:
        return f"{v:,.0f}".replace(',', '.')
    return f"{v:.4g}" if v >= 1 else f"{v:.5g}"


def render_trade_chart(feats, timeframe, cfg, side, signal_ts, sl, tp, symbol, exit_ts=None, exit_price=None,
                       exit_reason=None, path=None):
    """feats: compute_features(...) der geschlossenen Kerzen (wie live). signal_ts: Eroeffnungszeit der Signalkerze.
    Gibt den Pfad der PNG-Datei zurueck."""
    idx = feats.index
    i0 = idx.get_loc(pd.Timestamp(signal_ts))
    i1 = idx.get_loc(pd.Timestamp(exit_ts)) if exit_ts is not None else len(idx) - 1
    lo = max(0, i0 - BEFORE)
    hi = min(len(idx), max(i1, i0) + 4)
    win = feats.iloc[lo:hi]
    o, h, l, c = (feats[k].to_numpy(float) for k in ('open', 'high', 'low', 'close'))
    atr = feats['atr'].to_numpy(float)
    x = np.arange(len(win))
    pos = lambda j: j - lo

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 6.6), facecolor=SURFACE, sharex=True,
                                  gridspec_kw={'height_ratios': [3, 1.1], 'hspace': 0.08})
    fig.subplots_adjust(top=0.86, bottom=0.07, left=0.08, right=0.9)
    for a in (ax, ax2):
        a.set_facecolor(SURFACE)
        a.grid(axis='y', color=GRID, lw=0.8)
        a.tick_params(colors=MUTED, length=0, labelsize=9)
        for k, sp in a.spines.items():
            sp.set_visible(k == 'bottom')
            sp.set_color(GRID)
    up = win['close'] >= win['open']
    ax.vlines(x, win['low'], win['high'], color=MUTED, lw=1)
    ax.bar(x, (win['close'] - win['open']).abs(), bottom=np.minimum(win['open'], win['close']), width=0.62,
           color=np.where(up, '#5b5a55', '#3a3936'), edgecolor=np.where(up, INK2, MUTED), lw=0.6)

    found = []
    if tr.PROFILE == 'trend':
        ax.plot(x, win['ema20'], color=BLUE, lw=2, label='EMA20')
        ax.plot(x, win['ema50'], color=ORANGE, lw=2, label='EMA50')
    else:
        k = 5
        ph_all, pl_all = pivots(h[:i0 + 1], k, 'high'), pivots(l[:i0 + 1], k, 'low')
        ph = ph_all[(ph_all <= i0 - k) & (ph_all >= i0 - 300)]
        pl = pl_all[(pl_all <= i0 - k) & (pl_all >= i0 - 300)]
        for z, t in _zones(np.concatenate([h[ph], l[pl]]), 0.5 * atr[i0]):
            if t >= 2 and win['low'].min() <= z <= win['high'].max():
                ax.axhline(z, color=WARNING, lw=1, ls=':', alpha=0.8)
                ax.text(len(win) - 0.5, z, f" Zone ({t}×)", color=WARNING, fontsize=8, va='bottom', ha='right')
        for sd, piv, arr, col, lab in (('res', ph_all, h, ORANGE, 'Widerstand'), ('sup', pl_all, l, AQUA, 'Unterstützung')):
            ln = best_line(i0, piv, arr, c, atr[i0], k, sd)
            if ln is None:
                continue
            p1, p2, s_, ic, t = ln
            a0 = max(p1, lo)
            xs = np.array([pos(a0), pos(i0) + 3])
            ax.plot(xs, ic + s_ * (xs + lo), color=col, lw=2.2, label=f"{lab} ({t} Berührungen)")
            cand = piv[(piv >= p1) & (piv <= i0 - k)]
            hit = cand[np.abs(arr[cand] - (ic + s_ * cand)) <= 0.3 * atr[i0]]
            hit = hit[hit >= lo]
            ax.scatter(pos(hit), arr[hit], color=col, s=36, zorder=4)
        for P, arr, col, lab in ((ph, h, ORANGE, 'Netz-Eingabe: jüngste Hochs'), (pl, l, AQUA, 'Netz-Eingabe: jüngste Tiefs')):
            if len(P) >= 2 and P[-2] >= lo:
                a_, b_ = P[-2], P[-1]
                s_ = (arr[b_] - arr[a_]) / (b_ - a_)
                xs = np.array([pos(a_), pos(i0) + 3])
                ax.plot(xs, arr[a_] + s_ * (xs + lo - a_), color=col, lw=1.3, ls='--', alpha=0.85, label=lab)
        cp = candle_patterns(o[:i0 + 1], h[:i0 + 1], l[:i0 + 1], c[:i0 + 1], atr[:i0 + 1], '')
        found = [n.replace('_', ' ') for n, v in cp.items() if v[-1] > 0]

    end = pos(i1) if exit_ts is not None else len(win) - 1
    ax.hlines(sl, pos(i0), max(end, pos(i0) + 3), color=CRITICAL, lw=2, ls='--')
    ax.hlines(tp, pos(i0), max(end, pos(i0) + 3), color=GOOD, lw=2, ls='--')
    ax.text(len(win) + 0.3, sl, f"SL {_px(sl)}", color=CRITICAL, ha='left', va='center', fontsize=9, fontweight='bold')
    ax.text(len(win) + 0.3, tp, f"TP {_px(tp)}", color=GOOD, ha='left', va='center', fontsize=9, fontweight='bold')
    entry = c[i0]
    ax.scatter([pos(i0)], [entry], s=130, marker='^' if side == 'long' else 'v', color=VIOLET, edgecolor=SURFACE,
               lw=2, zorder=5)
    note = f"{'Long' if side == 'long' else 'Short'}-Einstieg {_px(entry)}" + (f"\nKerze: {', '.join(found)}" if found else '')
    ax.annotate(note, (pos(i0), entry), xytext=(-110, -60 if side == 'long' else 50), textcoords='offset points',
                ha='right', color=INK, fontsize=9, arrowprops=dict(arrowstyle='-', color=MUTED, lw=1),
                bbox=dict(boxstyle='round,pad=0.3', facecolor=SURFACE, edgecolor=GRID))
    R = None
    if exit_ts is not None and exit_price is not None:
        R = (exit_price - entry) / (entry - sl) if side == 'long' else (entry - exit_price) / (sl - entry)
        col = GOOD if R > 0 else CRITICAL
        ax.scatter([pos(i1)], [exit_price], s=110, marker='o', color=col, edgecolor=SURFACE, lw=2, zorder=5)
    ax.legend(loc='upper left', frameon=False, labelcolor=INK2, fontsize=8.5)
    ax.set_ylabel('Preis (USDT)', color=INK2)
    ax.set_xlim(-1, len(win) + 7)

    # ANN-Wahrscheinlichkeit des aktuellen Modells ueber das Bildfenster (Anzeige)
    ens = mdl.model_for(timeframe, idx[i0])
    if ens is not None:
        rows = win.dropna(subset=FEATURES)
        if len(rows):
            pr = mdl.predict_frame(ens, rows).reindex(win.index)
            key = 'pL' if side == 'long' else 'pS'
            thr = tr.thr_col('L' if side == 'long' else 'S', cfg['quantile'])
            ax2.plot(x, pr[key].to_numpy(), color=VIOLET, lw=2,
                     label=f"ANN: Wahrscheinlichkeit „{'TP vor SL' if side == 'long' else 'Short-Ziel'}“")
            ax2.plot(x, pr[thr].to_numpy(), color=WARNING, lw=1.5, ls=':', label='Schwelle')
            ax2.scatter([pos(i0)], [pr[key].iloc[pos(i0)]], color=AQUA, s=50, zorder=5)
            ax2.legend(loc='upper left', frameon=False, labelcolor=INK2, fontsize=8, ncol=2)
    ax2.set_ylabel('ANN', color=INK2)
    step = max(1, len(win) // 8)
    ax2.set_xticks(x[::step], [f"{d:%d.%m}" for d in win.index[::step]])

    coin = symbol.split('/')[0]
    net = 'Long-Netz' if tr.PROFILE == 'trend' else f'Short-Netz „{tr.PROFILE}“'
    if exit_ts is None:
        title = f"{coin} {timeframe} — {side.upper()} eröffnet"
        sub = f"{net} · Signal {pd.Timestamp(signal_ts):%d.%m.%Y %H:%M} UTC · SL {_px(sl)} · TP {_px(tp)}"
    else:
        why = {'tp': 'Ziel erreicht', 'sl': 'Stop ausgelöst', 'time': 'Zeitausstieg'}.get(exit_reason, 'geschlossen')
        title = f"{coin} {timeframe} — {side.upper()} geschlossen: {why}" + (f" ({R:+.1f} R brutto)" if R is not None else '')
        sub = f"{net} · Einstieg {pd.Timestamp(signal_ts):%d.%m. %H:%M} · Ausstieg {pd.Timestamp(exit_ts):%d.%m. %H:%M} UTC"
    fig.text(0.02, 0.965, title, fontsize=14, fontweight='bold', color=INK, va='top')
    fig.text(0.02, 0.915, sub, fontsize=9.5, color=INK2, va='top')
    path = path or os.path.join(tempfile.gettempdir(), f"jaeger_{coin}_{timeframe}_{tr.PROFILE}_{pd.Timestamp.now():%H%M%S%f}.png")
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    return path


def exit_from_candles(feats, timeframe, side, signal_ts, sl, tp):
    """Ausstieg (Kerze, Kurs, Grund) aus den Kerzen -- dieselbe Funktion wie der Backtest (trade_rules.simulate_exit).
    None, wenn die Kerzen keinen Ausstieg zeigen (z. B. Positionsabbruch von Hand)."""
    o, h, l, c = (feats[k].to_numpy(float) for k in ('open', 'high', 'low', 'close'))
    i0 = feats.index.get_loc(pd.Timestamp(signal_ts))
    ex = tr.simulate_exit(o, h, l, c, i0, side, sl, tp, tr.MAX_HOLD[timeframe])
    if ex is None:
        return None
    return feats.index[ex[0]], ex[1], ex[2]
