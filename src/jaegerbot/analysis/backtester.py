# src/jaegerbot/analysis/backtester.py
"""Backtest einzelner Configs und ganzer Portfolios -- ein Ereignis-Durchlauf fuer beides.

Ablauf wie live: zum Schluss jeder Kerze erst Ausstiege, dann Einstiege. Eine Strategie, die
flach ist, nimmt das naechste Signal; ein uebersprungenes Signal (max. Positionen, Symbol belegt,
Mindestgroesse) laesst sie flach, genau wie den Livebot.
"""
import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from jaegerbot.strategy import trade_rules as tr

FEAT_DIR = tr.profile_dir(os.path.join(PROJECT_ROOT, 'data', 'features'))
PRED_DIR = tr.profile_dir(os.path.join(PROJECT_ROOT, 'data', 'predictions'))
_frames = {}
_NEVER = pd.Timestamp('2262-01-01', tz='UTC')


def strategy_frame(coin, timeframe):
    """Kerzen + ATR/EMA + Walk-Forward-Vorhersagen einer Coin (gecacht pro Prozess)."""
    key = (coin, timeframe)
    if key not in _frames:
        fp = os.path.join(FEAT_DIR, timeframe, f"{coin}.pkl")
        pp = os.path.join(PRED_DIR, timeframe, f"{coin}.csv")
        if not (os.path.exists(fp) and os.path.exists(pp)):
            _frames[key] = None
        else:
            f = pd.read_pickle(fp)[['open', 'high', 'low', 'close', 'atr', 'ema20', 'ema50']]
            p = pd.read_csv(pp, index_col=0, parse_dates=True)
            p.index = pd.to_datetime(p.index, utc=True)
            _frames[key] = f.join(p, how='left')
    return _frames[key]


def signal_events(coin, timeframe, cfg, start=None, end=None):
    """Alle Signal-Kerzen einer Config mit vorab simuliertem Ausstieg.

    Einstieg zum Schluss der Signalkerze; SL/TP aus deren Schlusskurs; Ausstieg ueber
    trade_rules.simulate_exit. Zeiten = Kerzenschluss (Index + Timeframe).
    """
    df = strategy_frame(coin, timeframe)
    if df is None:
        return []
    tf = pd.Timedelta(timeframe)
    idx = df.index
    o, h, l, c = (df[k].to_numpy() for k in ('open', 'high', 'low', 'close'))
    atr = df['atr'].to_numpy()
    hold = tr.MAX_HOLD[timeframe]
    m = df['pL'].notna().to_numpy()
    if start is not None:
        m &= np.asarray(idx + tf >= pd.Timestamp(start))
    if end is not None:
        m &= np.asarray(idx + tf <= pd.Timestamp(end))
    cols = ['pL', 'pS', 'ema20', 'ema50', 'atr', tr.thr_col('L', cfg['quantile']), tr.thr_col('S', cfg['quantile'])]
    sub = df.loc[m, cols]
    events = []
    pos = np.flatnonzero(m)
    for i, row in zip(pos, sub.to_dict('records')):
        side = tr.entry_signal(row, cfg)
        if side is None:
            continue
        sl, tp = tr.levels(c[i], atr[i], side)
        ex = tr.simulate_exit(o, h, l, c, i, side, sl, tp, hold)
        events.append({
            'time': idx[i] + tf, 'coin': coin, 'timeframe': timeframe, 'side': side,
            'signal_close': c[i], 'sl': sl, 'tp': tp,
            'exit_time': (idx[ex[0]] + tf) if ex else None,
            'exit_raw': ex[1] if ex else None, 'reason': ex[2] if ex else 'open',
        })
    return events


def run_portfolio(strategies, start_capital=100.0, risk_pct=1.0, max_open=10, max_leverage=20,
                  start=None, end=None, events=None, risk_scale=1.0):
    """strategies: Liste von dicts mit coin, timeframe, cfg (Config-Parameter).

    Risiko je Trade: cfg['risk_pct'] der Strategie x risk_scale (Portfolio-Faktor), sonst risk_pct --
    wie live (trade_manager.trade_risk_pct).

    Liefert dict mit trades (DataFrame), stats und offenen Positionen am Ende.
    """
    if events is None:
        events = []
        for s in strategies:
            for e in signal_events(s['coin'], s['timeframe'], s['cfg'], start, end):
                e['key'] = f"{s['coin']}_{s['timeframe']}"
                if 'risk_pct' in s['cfg']:
                    e['risk_pct'] = s['cfg']['risk_pct']
                events.append(e)
    # wie live (master_runner): zum gleichen Kerzenschluss erst Profil 'trend' (prio 0), dann weitere Profile
    events.sort(key=lambda e: (e['time'], e.get('prio', 0), e['key']))

    equity = start_capital
    peak, max_dd = equity, 0.0
    open_pos = []    # dicts
    trades = []
    skipped = {'max_open': 0, 'symbol_busy': 0, 'min_size': 0, 'margin': 0}

    def close_until(t):
        nonlocal equity, peak, max_dd
        open_pos.sort(key=lambda p: p['exit_time'] if p['exit_time'] is not None else _NEVER)
        while open_pos and open_pos[0]['exit_time'] is not None and open_pos[0]['exit_time'] <= t:
            p = open_pos.pop(0)
            pnl, exit_fill = tr.trade_pnl(p['side'], p['qty'], p['entry_fill'], p['exit_raw'])
            equity += pnl
            peak = max(peak, equity)
            max_dd = max(max_dd, (peak - equity) / peak if peak > 0 else 1.0)
            risk_usd = abs(p['entry_fill'] - p['sl']) * p['qty']
            trades.append({**{k: p[k] for k in ('key', 'coin', 'timeframe', 'side', 'time', 'exit_time',
                                                 'entry_fill', 'sl', 'tp', 'qty', 'reason', 'leverage')},
                           'exit_fill': exit_fill, 'pnl': pnl, 'R': pnl / risk_usd if risk_usd else 0.0,
                           'equity_after': equity})

    for e in events:
        close_until(e['time'])
        if equity <= 0:
            break
        if any(p['key'] == e['key'] for p in open_pos):
            continue                       # Strategie noch in Position -> Signal existiert live nicht
        if any(p['coin'] == e['coin'] for p in open_pos):
            skipped['symbol_busy'] += 1
            continue
        if len(open_pos) >= max_open:
            skipped['max_open'] += 1
            continue
        entry_fill = tr.fill_price(e['signal_close'], e['side'], opening=True)
        risk = e['risk_pct'] * risk_scale if 'risk_pct' in e else risk_pct
        qty = tr.position_size(equity, risk, entry_fill, e['sl'])
        if qty is None:
            skipped['min_size'] += 1
            continue
        lev = tr.leverage_for(entry_fill, e['sl'], max_leverage)
        used = sum(p['qty'] * p['entry_fill'] / p['leverage'] for p in open_pos)
        if used + qty * entry_fill / lev > equity:
            skipped['margin'] += 1
            continue
        open_pos.append({**e, 'entry_fill': entry_fill, 'qty': qty, 'leverage': lev})

    close_until(_NEVER - pd.Timedelta(days=1))  # nur Positionen mit bekanntem Ausstieg
    still_open = [p for p in open_pos]

    t = pd.DataFrame(trades)
    n = len(t)
    stats = {
        'start_capital': start_capital, 'end_capital': equity,
        'pnl_pct': (equity / start_capital - 1) * 100, 'max_dd_pct': max_dd * 100,
        'trades': n, 'win_rate': float((t['pnl'] > 0).mean() * 100) if n else 0.0,
        'avg_R': float(t['R'].mean()) if n else 0.0, 'skipped': skipped, 'open_at_end': len(still_open),
    }
    return {'trades': t, 'stats': stats, 'open_positions': still_open}


def run_backtest(coin, timeframe, cfg, start=None, end=None, start_capital=100.0, risk_pct=1.0, max_leverage=20):
    s = [{'coin': coin, 'timeframe': timeframe, 'cfg': cfg}]
    return run_portfolio(s, start_capital, risk_pct, max_open=1, max_leverage=max_leverage, start=start, end=end)
