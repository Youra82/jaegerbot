"""README-Illustrationen aus echten Daten (docs/images/*.png).

  python docs/make_illustrations.py            # alle
  python docs/make_illustrations.py equity     # einzelne: pipeline trade equity control methods timeframes safety

Braucht die lokalen Walk-Forward-Vorhersagen (data/predictions) und artifacts/results/rules.json.
Farben: Referenzpalette (dataviz) dunkel, unveraendert auf ihrer validierten Flaeche #1a1a19.
"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd

OUT = os.path.join(ROOT, 'docs', 'images')
SURFACE, INK, INK2, MUTED, GRID = '#1a1a19', '#f0efec', '#c3c2b7', '#898781', '#2c2c2a'
BLUE, ORANGE, AQUA, VIOLET = '#3987e5', '#d95926', '#199e70', '#9085e9'
GOOD, CRITICAL, WARNING = '#0ca30c', '#d03b3b', '#fab219'
START = pd.Timestamp('2023-01-01', tz='UTC')

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'text.color': INK,
                     'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED})


def _fig(w, h, rows=1, **kw):
    fig, axes = plt.subplots(rows, 1, figsize=(w, h), facecolor=SURFACE, **kw)
    for ax in np.atleast_1d(axes):
        ax.set_facecolor(SURFACE)
        for side, sp in ax.spines.items():
            sp.set_visible(side == 'bottom')
            sp.set_color(GRID)
        ax.grid(axis='y', color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
    return fig, axes


def _title(fig, title, sub):
    fig.text(0.02, 0.965, title, fontsize=15, fontweight='bold', color=INK, va='top')
    fig.text(0.02, 0.905, sub, fontsize=10, color=INK2, va='top')


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print('geschrieben:', name)


def _rules():
    with open(os.path.join(ROOT, 'artifacts', 'results', 'rules.json'), encoding='utf-8') as f:
        return json.load(f)


def _team():
    with open(os.path.join(ROOT, 'settings.json'), encoding='utf-8') as f:
        lt = json.load(f)['live_trading_settings']
    return lt


def _px(v):
    """Preis lesbar: 84.820 statt 8.48e+04, kleine Preise mit genug Stellen."""
    if v >= 1000:
        return f"{v:,.0f}".replace(',', '.')
    return f"{v:.4g}" if v >= 1 else f"{v:.5g}"


def _box(ax, x, y, w, h, title, sub, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.06',
                                facecolor='#232321', edgecolor=color, linewidth=2))
    ax.add_patch(Rectangle((x, y + h - 0.07), w, 0.07, facecolor=color, edgecolor='none'))
    ax.text(x + w / 2, y + h * 0.58, title, ha='center', va='center', fontsize=11, fontweight='bold', color=INK)
    ax.text(x + w / 2, y + h * 0.27, sub, ha='center', va='center', fontsize=8.5, color=INK2, linespacing=1.4)


def _arrow(ax, x0, y0, x1, y1, color=MUTED):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=1.8, shrinkA=0, shrinkB=0))


# --- 1 Ablauf ---------------------------------------------------------------
def pipeline():
    fig = plt.figure(figsize=(12, 4.2), facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 12); ax.set_ylim(0, 4.2); ax.axis('off')
    ax.text(0.3, 3.85, 'JaegerBot v5', fontsize=22, fontweight='bold', color=INK, va='top')
    ax.text(0.3, 3.3, 'ANN-Ensemble · Walk-Forward · eine Logik für Live und Backtest', fontsize=11, color=INK2, va='top')
    steps = [('Kerzen', 'Bitget 4h\nDatenprüfung', BLUE), ('46 Features', 'Rolling-Fenster\nlive = Backtest', BLUE),
             ('ANN-Ensemble', '3 MLPs je Seite\nQuartalsmodell', VIOLET), ('Signal', 'Top 1 % + Trend\nnur Long', AQUA),
             ('Order', 'Market + SL/TP\nan der Order', ORANGE), ('Selbstcheck', 'täglich Telegram\nLive vs. Backtest', ORANGE)]
    w, gap, y = 1.62, 0.33, 0.55
    for i, (t, s, c) in enumerate(steps):
        x = 0.3 + i * (w + gap)
        _box(ax, x, y, w, 1.85, t, s, c)
        if i:
            _arrow(ax, x - gap + 0.04, y + 0.92, x - 0.04, y + 0.92)
    _save(fig, 'pipeline.png')


# --- 2 Echter Trade ---------------------------------------------------------
def trade():
    from jaegerbot.analysis.backtester import run_portfolio, strategy_frame
    from jaegerbot.analysis.optimizer import oos_start, events_for
    from jaegerbot.strategy import trade_rules as tr
    r = _rules()['4h']
    ob = oos_start()
    res = run_portfolio([], 100, 1.0, 10, 20, events=events_for(r['coins'], '4h', r['cfg'], ob))
    t = res['trades']
    t = t[t['reason'] == 'tp'].sort_values('time').iloc[-1]
    df = strategy_frame(t['coin'], '4h')
    tf = pd.Timedelta('4h')
    i0, i1 = df.index.get_loc(t['time'] - tf), df.index.get_loc(t['exit_time'] - tf)
    win = df.iloc[max(0, i0 - 40): i1 + 8]
    thr = df[tr.thr_col('L', r['cfg']['quantile'])]

    fig, (ax, ax2) = _fig(12, 6.4, rows=2, gridspec_kw={'height_ratios': [3, 1.2], 'hspace': 0.08}, sharex=True)
    fig.subplots_adjust(top=0.84, bottom=0.08, left=0.07, right=0.97)
    x = np.arange(len(win))
    up = win['close'] >= win['open']
    ax.vlines(x, win['low'], win['high'], color=MUTED, linewidth=1)
    ax.bar(x, (win['close'] - win['open']).abs(), bottom=np.minimum(win['open'], win['close']), width=0.62,
           color=np.where(up, '#5b5a55', '#3a3936'), edgecolor=np.where(up, INK2, MUTED), linewidth=0.6)
    ax.plot(x, win['ema20'], color=BLUE, linewidth=2, label='EMA20')
    ax.plot(x, win['ema50'], color=ORANGE, linewidth=2, label='EMA50')
    e = i0 - max(0, i0 - 40)
    xe = i1 - max(0, i0 - 40)
    ax.hlines(t['sl'], e, xe, color=CRITICAL, linewidth=2, linestyles='--')
    ax.hlines(t['tp'], e, xe, color=GOOD, linewidth=2, linestyles='--')
    ax.text(e - 0.6, t['sl'], f"SL {_px(t['sl'])}", color=CRITICAL, ha='right', va='center', fontsize=9, fontweight='bold')
    ax.text(e - 0.6, t['tp'], f"TP {_px(t['tp'])}", color=GOOD, ha='right', va='center', fontsize=9, fontweight='bold')
    ax.scatter([e], [t['entry_fill']], s=120, marker='^', color=AQUA, edgecolor=SURFACE, linewidth=2, zorder=5)
    ax.scatter([xe], [t['exit_fill']], s=120, marker='o', color=GOOD, edgecolor=SURFACE, linewidth=2, zorder=5)
    ax.annotate(f"Einstieg Long {_px(t['entry_fill'])}", (e, t['entry_fill']), xytext=(-14, 34), textcoords='offset points',
                color=INK, fontsize=9, ha='right', arrowprops=dict(arrowstyle='-', color=MUTED, lw=1),
                bbox=dict(boxstyle='round,pad=0.3', facecolor=SURFACE, edgecolor=GRID))
    ax.annotate(f"Ziel erreicht  {t['R']:+.1f} R", (xe, t['exit_fill']), xytext=(-12, 12), textcoords='offset points',
                color=INK, fontsize=9, ha='right', bbox=dict(boxstyle='round,pad=0.3', facecolor=SURFACE, edgecolor=GRID))
    ax.legend(loc='upper left', frameon=False, labelcolor=INK2, fontsize=9)
    ax.set_ylabel('Preis (USDT)')

    p = win['pL'].to_numpy()
    ax2.plot(x, p, color=VIOLET, linewidth=2, label='ANN: Wahrscheinlichkeit „TP vor SL“')
    ax2.plot(x, thr.loc[win.index].to_numpy(), color=WARNING, linewidth=1.5, linestyle=':', label='Schwelle (Top 1 %)')
    ax2.scatter([e], [p[e]], s=60, color=AQUA, zorder=5)
    ax2.legend(loc='upper left', frameon=False, labelcolor=INK2, fontsize=8.5, ncol=2)
    ax2.set_ylabel('ANN')
    ticks = x[::8]
    ax2.set_xticks(ticks, [f"{d:%d.%m}" for d in win.index[::8]])
    _title(fig, f"Echter Trade aus dem OOS: {t['coin']} 4h",
           f"Signal {t['time'] - tf:%d.%m.%Y %H:%M} UTC · Modell kannte diese Kerzen nicht (Walk-Forward) · "
           f"Ausstieg {t['exit_time']:%d.%m. %H:%M}")
    _save(fig, 'trade.png')


# --- 3 Equity + Drawdown ----------------------------------------------------
def equity():
    from jaegerbot.analysis.backtester import run_portfolio
    from jaegerbot.analysis.optimizer import oos_start, events_for
    lt = _team()
    r = _rules()['4h']
    res = run_portfolio([], 100, lt['risk_per_trade_pct'], lt['max_open_positions'], lt['max_leverage'],
                        events=events_for(r['coins'], '4h', r['cfg'], START))
    t = res['trades'].sort_values('exit_time')
    eq = pd.concat([pd.Series([100.0], index=[START]), t.set_index('exit_time')['equity_after']])
    dd = (eq / eq.cummax() - 1) * 100
    ob = oos_start()
    years = (eq.index[-1] - START).days / 365.25
    cagr = ((eq.iloc[-1] / 100) ** (1 / years) - 1) * 100

    fig, (ax, ax2) = _fig(12, 6, rows=2, gridspec_kw={'height_ratios': [3, 1.1], 'hspace': 0.1}, sharex=True)
    fig.subplots_adjust(top=0.84, bottom=0.08, left=0.07, right=0.97)
    for a in (ax, ax2):
        a.axvspan(ob, eq.index[-1], color=AQUA, alpha=0.10, linewidth=0)
    ax.step(eq.index, eq.values, where='post', color=BLUE, linewidth=2)
    ax.text(ob - pd.Timedelta(days=8), eq.min() + (eq.max() - eq.min()) * 0.06, 'OOS: von keiner\nAuswahl gesehen  →',
            color=AQUA, fontsize=9, ha='right', va='bottom')
    ax.annotate(f"{eq.iloc[-1]:.0f}", (eq.index[-1], eq.iloc[-1]), xytext=(-4, 6), textcoords='offset points',
                ha='right', color=INK, fontweight='bold')
    ax.set_ylabel('Kapital (Start 100)')
    ax2.fill_between(dd.index, dd.values, 0, step='post', color=CRITICAL, alpha=0.55, linewidth=0)
    ax2.set_ylabel('Drawdown %')
    ax2.annotate(f"max {dd.min():.0f} %", (dd.idxmin(), dd.min()), xytext=(6, 2), textcoords='offset points',
                 color=INK2, fontsize=9)
    _title(fig, f"Portfolio seit 2023: {cagr:+.0f} % pro Jahr, max. Drawdown {-dd.min():.0f} %",
           f"4h-Regel auf {len(r['coins'])} Coins · {lt['risk_per_trade_pct']} % Risiko je Trade · max {lt['max_open_positions']} Positionen · "
           f"{len(t)} Trades · Gebühren + Schlupf 0,22 % je Runde")
    _save(fig, 'equity.png')


# --- 4 ANN gegen Zufall -----------------------------------------------------
def control():
    from jaegerbot.analysis.backtester import strategy_frame
    from jaegerbot.strategy import trade_rules as tr
    r = _rules()
    rng = np.random.default_rng(0)
    rows = []
    for tf in ('1h', '4h', '6h'):
        if tf not in r or 'cfg' not in r[tf]:
            continue
        cfg = r[tf]['cfg']
        ann, rnd = [], []
        for c in r[tf]['coins']:
            df = strategy_frame(c, tf)
            df = df[df.index >= START]
            o, h, l, cl, atr = (df[k].to_numpy() for k in ('open', 'high', 'low', 'close', 'atr'))
            ok = (df['pL'].notna() & (df['ema20'] > df['ema50'])).to_numpy()
            sig = ok & (df['pL'] >= df[tr.thr_col('L', cfg['quantile'])]).to_numpy()

            def R(i):
                sl, tp = tr.levels(cl[i], atr[i], 'long')
                ex = tr.simulate_exit(o, h, l, cl, i, 'long', sl, tp, tr.MAX_HOLD[tf])
                if ex is None:
                    return None
                ef = tr.fill_price(cl[i], 'long', True)
                return tr.trade_pnl('long', 1.0, ef, ex[1])[0] / abs(ef - sl)
            ann += [x for x in map(R, np.flatnonzero(sig)) if x is not None]
            pool = np.flatnonzero(ok)
            rnd += [x for x in map(R, rng.choice(pool, min(len(pool), 5 * max(1, sig.sum())), replace=False)) if x is not None]
        rows.append((tf, np.mean(ann), np.std(ann) / np.sqrt(len(ann)), len(ann),
                     np.mean(rnd), np.std(rnd) / np.sqrt(len(rnd))))
    fig, ax = _fig(10, 5.4)
    fig.subplots_adjust(top=0.78, bottom=0.12, left=0.09, right=0.97)
    x = np.arange(len(rows))
    w = 0.36
    ax.bar(x - w / 2 - 0.01, [r_[1] for r_ in rows], w, yerr=[1.96 * r_[2] for r_ in rows], color=VIOLET,
           error_kw=dict(ecolor=INK2, lw=1.2, capsize=4), label='ANN-Signale (Top-Schwelle + Trend)')
    ax.bar(x + w / 2 + 0.01, [r_[4] for r_ in rows], w, yerr=[1.96 * r_[5] for r_ in rows], color='#5b5a55',
           error_kw=dict(ecolor=INK2, lw=1.2, capsize=4), label='Zufällige Longs im Aufwärtstrend')
    ax.axhline(0, color=MUTED, linewidth=1)
    for i, r_ in enumerate(rows):
        ax.text(i - w / 2, r_[1] + 1.96 * r_[2] + 0.01, f"{r_[1]:+.2f} R", ha='center', va='bottom', color=INK, fontweight='bold')
        ax.text(i + w / 2, min(0, r_[4]) - 1.96 * r_[5] - 0.012, f"{r_[4]:+.2f} R", ha='center', va='top', color=INK2)
    ax.set_xticks(x, [f"{r_[0]}  (n={r_[3]})" for r_ in rows])
    ax.set_ylabel('Ø Ergebnis je Trade (R, netto)')
    ax.legend(loc='upper left', frameon=False, labelcolor=INK2)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.04, hi + 0.06)
    _title(fig, 'Das ANN wählt besser als der Zufall',
           'Gleiche Ausstiege, gleiche Kosten, gleicher Trend-Filter seit 2023 · Fehlerbalken = 95 %-Bereich')
    _save(fig, 'control.png')


# --- 5 Regel gegen fibot-Methode -------------------------------------------
def methods():
    # Rueckversetz-Test 25.10.2026: Auswahl + Bestaetigung + Risiko bis 26.04.2026, dann 26 Wochen vorwaerts
    data = [('Regel je Timeframe\n(ANN, alle Coins)', 22.2, AQUA), ('Auswahl je Coin\n(ANN, fibot-Art)', -2.2, BLUE),
            ('Zufall 1', -23.3, '#5b5a55'), ('Zufall 2', -52.0, '#5b5a55'), ('Zufall 3', 27.1, '#5b5a55'),
            ('Zufall 4', -38.7, '#5b5a55'), ('Zufall 5', -7.9, '#5b5a55')]
    fig, ax = _fig(11, 5.4)
    fig.subplots_adjust(top=0.78, bottom=0.2, left=0.08, right=0.97)
    x = np.arange(len(data))
    ax.bar(x, [d[1] for d in data], 0.62, color=[d[2] for d in data])
    ax.axhline(0, color=MUTED, linewidth=1)
    for i, (_, v, _c) in enumerate(data):
        ax.text(i, v + (1.5 if v >= 0 else -1.5), f"{v:+.1f} %", ha='center', va='bottom' if v >= 0 else 'top',
                color=INK if i < 2 else INK2, fontweight='bold' if i < 2 else 'normal')
    ax.set_xticks(x, [d[0] for d in data], fontsize=9)
    ax.set_ylabel('Ergebnis in den 26 Wochen danach (%)')
    ax.set_ylim(-64, 44)
    ax.axvspan(1.5, 6.5, color='#232321', zorder=0)
    ax.text(4, 38, 'fibot-Methode mit Zufalls-Vorhersagen', ha='center', color=MUTED, fontsize=9)
    _title(fig, 'Vorwärts-Test: eine Regel schlägt die Auswahl je Coin',
           'Alles bis 26.04.2026 ausgewählt und bestätigt, danach 26 Wochen ohne jede Anpassung gerechnet')
    _save(fig, 'methods.png')


# --- 6 Timeframe-Pruefung ---------------------------------------------------
def timeframes():
    r = _rules()
    tfs = [tf for tf in ('1h', '2h', '4h', '6h', '1d') if tf in r and 'oos' in r[tf]]
    team = {s['timeframe'] for s in _team().get('active_strategies', [])}
    fig, ax = _fig(10, 5.2)
    fig.subplots_adjust(top=0.78, bottom=0.14, left=0.09, right=0.97)
    x = np.arange(len(tfs))
    vals = [r[tf]['oos']['pnl_pct'] for tf in tfs]
    cols = [AQUA if tf in team else (BLUE if r[tf]['confirmed'] else '#5b5a55') for tf in tfs]
    ax.bar(x, vals, 0.6, color=cols)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.axhline(10, color=WARNING, linewidth=1.2, linestyle=':')
    ax.text(len(tfs) - 0.5, 10.6, 'Kriterium OOS ≥ +10 %', color=WARNING, ha='right', fontsize=9)
    for i, tf in enumerate(tfs):
        c = r[tf]['cfg']
        state = 'LIVE' if tf in team else ('bestätigt' if r[tf]['confirmed'] else 'abgelehnt')
        v = vals[i]
        ax.text(i, v + (0.8 if v >= 0 else -0.8), f"{v:+.1f} %\n{state}", ha='center', va='bottom' if v >= 0 else 'top',
                color=INK if tf in team else INK2, fontweight='bold' if tf in team else 'normal', fontsize=9)
    ax.set_xticks(x, [f"{tf}\n{r[tf]['cfg']['sides']} · Top {100 - r[tf]['cfg']['quantile'] * 100:.1f} %" for tf in tfs])
    ax.set_ylabel('OOS-Ergebnis 26 Wochen (%)')
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 6, hi + 3)
    _title(fig, 'Jedes Quartal neu geprüft: welche Timeframe-Regel trägt?',
           'Regel im IS gewählt, im OOS bestätigt · LIVE = vom Portfolio-Optimizer gewählt (max. Endkapital bei DD ≤ 30 %)')
    _save(fig, 'timeframes.png')


# --- 7 Sicherung und Pruef-Tor ---------------------------------------------
def safety():
    fig = plt.figure(figsize=(12, 4.6), facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 12); ax.set_ylim(0, 4.6); ax.axis('off')
    ax.text(0.3, 4.3, 'Quartals-Pipeline: gesichert und selbst geprüft', fontsize=15, fontweight='bold', color=INK, va='top')
    ax.text(0.3, 3.85, 'Jeder Schritt muss bestehen, sonst wird automatisch der alte Stand zurückgespielt',
            fontsize=10, color=INK2, va='top')
    steps = [('Sicherung', 'Modelle, Configs,\nsettings.json', BLUE), ('Daten', 'Lücken? veraltet?\nzu kurz?', BLUE),
             ('Neues Modell', 'nur Vergangenheit\nPlausibilität', VIOLET), ('Regeln', 'IS-Wahl\nOOS-Bestätigung', VIOLET),
             ('Paritätstest', 'Live-Code =\nBacktest', AQUA), ('Live', 'neues Team\n+ Telegram', GOOD)]
    w, gap, y = 1.62, 0.33, 1.75
    for i, (t, s, c) in enumerate(steps):
        x = 0.3 + i * (w + gap)
        _box(ax, x, y, w, 1.55, t, s, c)
        if i:
            _arrow(ax, x - gap + 0.04, y + 0.78, x - 0.04, y + 0.78)
    # Ruecksicherungs-Pfad
    ax.add_patch(FancyBboxPatch((1.3, 0.35), 9.5, 0.8, boxstyle='round,pad=0.02,rounding_size=0.06',
                                facecolor='#2a1d1d', edgecolor=CRITICAL, linewidth=2))
    ax.text(6.02, 0.75, '✖ Fehler in einem Schritt  →  Rücksicherung + Telegram-Alarm  →  Bot handelt mit Vorquartal-Modell weiter',
            ha='center', va='center', color=INK, fontsize=10)
    for i in range(1, 5):
        xm = 0.3 + i * (w + gap) + w / 2
        ax.plot([xm, xm], [y - 0.02, 1.17], color=CRITICAL, linewidth=1.2, linestyle=':')
    _save(fig, 'safety.png')


ALL = {'pipeline': pipeline, 'trade': trade, 'equity': equity, 'control': control, 'methods': methods,
       'timeframes': timeframes, 'safety': safety}

if __name__ == '__main__':
    for name in (sys.argv[1:] or ALL):
        ALL[name]()
