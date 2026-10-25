# src/jaegerbot/strategy/trade_rules.py
"""Einzige Quelle fuer Einstieg, SL/TP, Positionsgroesse, Ausstieg und Kosten.

Live (trade_manager), Backtester und Portfolio-Simulator rufen nur diese Funktionen auf.
Der Ausstieg ist genau das, was das ANN gelernt hat (Label): TP = RR x SL vor SL, sonst
Zeitausstieg nach MAX_HOLD Kerzen.
"""
import math

SL_ATR_MULT = 2.0
RISK_REWARD = 2.0
MAX_HOLD = {'1h': 16, '2h': 16, '4h': 12, '6h': 12, '1d': 8}

FEE_PCT = 0.0006       # Bitget Taker je Seite
SLIPPAGE_PCT = 0.0005  # Markt-/Stop-Ausfuehrung je Seite
MIN_NOTIONAL_USDT = 5.0
LIQ_SAFETY = 0.8       # Hebel so, dass die Liquidation erst nach 1/0.8 SL-Abstand kommt

QUANTILE_LEVELS = (0.9, 0.95, 0.975, 0.99)


def thr_col(side, q):
    return f"thr{side}_{int(round(q * 1000))}"


def entry_signal(row, cfg):
    """row: Mapping mit pL, pS, thrL_*/thrS_*, ema20, ema50. Liefert 'long', 'short' oder None."""
    q = cfg['quantile']
    sides = cfg['sides']
    cand = []
    if 'L' in sides and row['pL'] >= row[thr_col('L', q)]:
        cand.append(('long', row['pL'] - row[thr_col('L', q)]))
    if 'S' in sides and row['pS'] >= row[thr_col('S', q)]:
        cand.append(('short', row['pS'] - row[thr_col('S', q)]))
    if not cand:
        return None
    side = max(cand, key=lambda x: x[1])[0]
    if cfg.get('ema_filter'):
        if side == 'long' and not row['ema20'] > row['ema50']:
            return None
        if side == 'short' and not row['ema20'] < row['ema50']:
            return None
    return side


def levels(signal_close, atr, side):
    """SL/TP relativ zum Schlusskurs der Signalkerze (live und Backtest gleich)."""
    d = atr * SL_ATR_MULT
    if side == 'long':
        return signal_close - d, signal_close + d * RISK_REWARD
    return signal_close + d, signal_close - d * RISK_REWARD


def fill_price(price, side, opening):
    """Schlupf gegen uns: Kauf teurer, Verkauf billiger."""
    buy = (side == 'long') == opening
    return price * (1 + SLIPPAGE_PCT) if buy else price * (1 - SLIPPAGE_PCT)


def leverage_for(entry, sl, max_leverage):
    sl_pct = abs(entry - sl) / entry
    return max(1, min(int(max_leverage), int(math.floor(LIQ_SAFETY / sl_pct)))) if sl_pct > 0 else 1


def position_size(equity, risk_pct, entry, sl):
    """Menge (Coins), sodass ein SL-Treffer risk_pct vom Equity kostet. None unter Mindestgroesse."""
    dist = abs(entry - sl)
    if dist <= 0 or equity <= 0:
        return None
    qty = equity * risk_pct / 100.0 / dist
    if qty * entry < MIN_NOTIONAL_USDT:
        return None
    return qty


def simulate_exit(o, h, l, c, i, side, sl, tp, max_hold):
    """Ausstieg auf Kerzen ab i+1 (i = Signalkerze). Arrays o/h/l/c.

    Liefert (exit_index, roher Ausstiegskurs, Grund) oder None, wenn die Daten vor dem
    Ausstieg enden (Position noch offen). SL vor TP in derselben Kerze (pessimistisch);
    oeffnet eine Kerze schon jenseits des SL, Fuellung zur Eroeffnung.
    """
    n = len(c)
    last = i + max_hold
    for j in range(i + 1, min(last, n - 1) + 1):
        if side == 'long':
            if o[j] <= sl:
                return j, o[j], 'sl'
            if l[j] <= sl:
                return j, sl, 'sl'
            if o[j] >= tp:
                return j, o[j], 'tp'
            if h[j] >= tp:
                return j, tp, 'tp'
        else:
            if o[j] >= sl:
                return j, o[j], 'sl'
            if h[j] >= sl:
                return j, sl, 'sl'
            if o[j] <= tp:
                return j, o[j], 'tp'
            if l[j] <= tp:
                return j, tp, 'tp'
        if j == last:
            return j, c[j], 'time'
    return None


def trade_pnl(side, qty, entry_fill, exit_raw):
    """Netto-PnL in USDT inkl. Gebuehren beider Seiten und Ausstiegs-Schlupf."""
    exit_fill = fill_price(exit_raw, side, opening=False)
    direction = 1 if side == 'long' else -1
    gross = (exit_fill - entry_fill) * qty * direction
    fees = (entry_fill + exit_fill) * qty * FEE_PCT
    return gross - fees, exit_fill
