# src/jaegerbot/utils/trade_manager.py
"""Live-Handelszyklus einer Strategie (Coin/Timeframe). Logik ausschliesslich aus trade_rules --
dieselben Funktionen wie Backtester und Portfolio-Simulator.

Je Lauf (Cron alle 15 Min):
1. eigene Position verwalten: weg -> SL/TP hat ausgeloest; Haltedauer abgelaufen -> Market-Close
2. letzte geschlossene Kerze genau einmal bewerten (Signal nicht aelter als MAX_SIGNAL_AGE)
3. bei Signal: Portfolio-Regeln wie Simulator, dann Market-Einstieg mit angehaengtem SL/TP
"""
import os
import json
import pandas as pd

from jaegerbot.utils.telegram import send_message
from jaegerbot.strategy.features import compute_features, FEATURES, WARMUP_CANDLES
from jaegerbot.strategy import trade_rules as tr
from jaegerbot.strategy import model as mdl
from jaegerbot.utils.data_check import check_candles

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
# je Profil eigene Zustaende (trend: artifacts/db, geohtf: artifacts/db_geohtf) -- Long und Short derselben Coin
# duerfen sich keinen Zustand teilen
STATE_DIR = tr.profile_dir(os.path.join(PROJECT_ROOT, 'artifacts', 'db'))
BOT = 'jaegerbot' if tr.PROFILE == 'trend' else f'jaegerbot[{tr.PROFILE}]'
MAX_SIGNAL_AGE = pd.Timedelta(minutes=20)
FETCH_CANDLES = WARMUP_CANDLES + 300


def strategy_key(symbol, timeframe):
    return f"{symbol.replace('/', '').replace(':', '')}_{timeframe}"


def _state_path(key):
    return os.path.join(STATE_DIR, f"state_{key}.json")


def load_state(key):
    try:
        with open(_state_path(key), 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_state(key, state):
    os.makedirs(STATE_DIR, exist_ok=True)
    tmp = _state_path(key) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, _state_path(key))


def _notify(telegram, text, logger):
    try:
        if telegram.get('bot_token') and telegram.get('chat_id'):
            send_message(telegram['bot_token'], telegram['chat_id'], text)
    except Exception as e:
        logger.warning(f"Telegram fehlgeschlagen: {e}")


def manage_position(exchange, state, symbol, timeframe, last_ts, telegram, logger):
    """Prueft die eigene Position. Gibt True zurueck, wenn die Strategie danach flach ist."""
    pos = state.get('position')
    if not pos:
        return True
    live = [p for p in exchange.fetch_open_positions_strict(symbol) if p.get('side') == pos['side']]
    if not live:
        logger.info(f"Position {pos['side']} geschlossen (SL/TP ausgeloest). Raeume Rest-Orders auf.")
        exchange.cancel_all_orders_for_symbol(symbol)
        _notify(telegram, f"{BOT} {symbol} {timeframe}: {pos['side']} durch SL/TP geschlossen.", logger)
        state['position'] = None
        return True
    due = pd.Timestamp(pos['signal_ts']) + tr.MAX_HOLD[timeframe] * pd.Timedelta(timeframe)
    if last_ts >= due:
        amount = abs(float(live[0].get('contracts') or 0))
        logger.info(f"Haltedauer abgelaufen (Kerze {due}) -> schliesse {pos['side']} {amount} per Market.")
        exchange.cancel_all_orders_for_symbol(symbol)
        exchange.close_position_market(symbol, pos['side'], amount)
        _notify(telegram, f"{BOT} {symbol} {timeframe}: {pos['side']} nach {tr.MAX_HOLD[timeframe]} Kerzen geschlossen (Zeitausstieg).", logger)
        state['position'] = None
        return True
    return False


def trade_risk_pct(cfg, live_settings):
    """Risiko je Trade wie backtester.run_portfolio: Strategie-Risiko x Portfolio-Faktor."""
    if 'risk_pct' in cfg:
        return float(cfg['risk_pct']) * float(live_settings.get('risk_scale', 1.0))
    return float(live_settings.get('risk_per_trade_pct', 1.0))


def decide_entry(df, timeframe, cfg):
    """Einstiegsentscheidung fuer die letzte geschlossene Kerze in df -- live und im Paritaetstest
    (tests/parity_check.py) dieselbe Funktion. Liefert side/sl/tp/row/pred oder error."""
    problems = check_candles(df, timeframe, min_candles=WARMUP_CANDLES + 2)
    if problems:
        return {'error': f"Daten fehlerhaft ({'; '.join(problems)})"}
    feats = compute_features(df)
    row = feats.iloc[-1]
    if row[FEATURES + ['atr']].isna().any():
        return {'error': 'Features unvollstaendig'}
    ens = mdl.model_for(timeframe, df.index[-1])
    if ens is None:
        return {'error': f"kein Modell fuer {timeframe} (artifacts/models/{timeframe})"}
    pred = mdl.predict_frame(ens, feats.iloc[[-1]]).iloc[0].to_dict()
    pred.update(ema20=row['ema20'], ema50=row['ema50'], atr=row['atr'])
    side = tr.entry_signal(pred, cfg)
    sl, tp = tr.levels(float(row['close']), float(row['atr']), side) if side else (None, None)
    return {'side': side, 'sl': sl, 'tp': tp, 'row': row, 'pred': pred}


def full_trade_cycle(exchange, config, live_settings, telegram, logger, now=None, df=None, manage_only=False):
    """manage_only=True: nur Ausstiege (master_runner macht erst fuer ALLE Strategien die Ausstiege,
    dann die Einstiege -- wie der Simulator, der zu jedem Kerzenschluss erst schliesst, dann eroeffnet).
    Gibt die geladenen Kerzen zurueck, damit der zweite Durchgang sie wiederverwendet."""
    symbol = config['market']['symbol']
    timeframe = config['market']['timeframe']
    cfg = config['strategy']
    key = strategy_key(symbol, timeframe)
    now = now or pd.Timestamp.now(tz='UTC')
    state = load_state(key)

    if df is None:
        df = exchange.fetch_recent_ohlcv(symbol, timeframe, limit=FETCH_CANDLES)
    if df is None or len(df) < WARMUP_CANDLES + 2:
        logger.warning(f"Nur {0 if df is None else len(df)} Kerzen geladen -- ueberspringe Lauf.")
        return None
    last_ts = df.index[-1]   # Eroeffnungszeit der letzten GESCHLOSSENEN Kerze

    flat = manage_position(exchange, state, symbol, timeframe, last_ts, telegram, logger)
    if manage_only:
        save_state(key, state)
        return df

    if state.get('last_evaluated') == str(last_ts):
        save_state(key, state)
        return
    state['last_evaluated'] = str(last_ts)
    if not flat:
        logger.info(f"Kerze {last_ts}: Strategie in Position -- kein neues Signal.")
        save_state(key, state)
        return
    age = now - (last_ts + pd.Timedelta(timeframe))
    if age > MAX_SIGNAL_AGE:
        logger.info(f"Kerze {last_ts}: Signal zu alt ({age}) -- nicht gehandelt.")
        save_state(key, state)
        return

    d = decide_entry(df, timeframe, cfg)
    if d.get('error'):
        logger.warning(f"Kerze {last_ts}: {d['error']} -- kein Einstieg.")
        save_state(key, state)
        return None
    side, row, pred = d['side'], d['row'], d['pred']
    q = cfg['quantile']
    logger.info(f"Kerze {last_ts}: pL {pred['pL']:.4f}/{pred[tr.thr_col('L', q)]:.4f} "
                f"pS {pred['pS']:.4f}/{pred[tr.thr_col('S', q)]:.4f} Modell {pred['model']} -> {side or 'kein Signal'}")
    if side is None:
        save_state(key, state)
        return

    # Portfolio-Regeln (identisch zu backtester.run_portfolio)
    positions = exchange.fetch_open_positions_strict()
    if any(p.get('symbol') == symbol for p in positions):
        logger.info("Signal verworfen: Symbol bereits belegt.")
        save_state(key, state)
        return
    if len(positions) >= int(live_settings.get('max_open_positions', 10)):
        logger.info("Signal verworfen: max. offene Positionen erreicht.")
        save_state(key, state)
        return
    close = float(row['close'])
    sl, tp = tr.levels(close, float(row['atr']), side)
    entry_ref = tr.fill_price(close, side, opening=True)
    equity = exchange.fetch_realized_equity_usdt()
    risk = trade_risk_pct(cfg, live_settings)
    qty = tr.position_size(equity, risk, entry_ref, sl)
    if qty is None:
        logger.info(f"Signal verworfen: Mindestgroesse ({tr.MIN_NOTIONAL_USDT} USDT) nicht erreicht, Equity {equity:.2f}.")
        save_state(key, state)
        return
    lev = tr.leverage_for(entry_ref, sl, live_settings.get('max_leverage', 20))
    used = sum(abs(float(p.get('initialMargin') or 0)) for p in positions)
    if used + qty * entry_ref / lev > equity:
        logger.info(f"Signal verworfen: Margin reicht nicht (gebunden {used:.2f}, Equity {equity:.2f}).")
        save_state(key, state)
        return
    last_price = float(exchange.fetch_ticker(symbol)['last'])
    if (side == 'long' and not sl < last_price < tp) or (side == 'short' and not tp < last_price < sl):
        logger.info(f"Signal verworfen: Kurs {last_price} schon ausserhalb SL/TP ({sl:.6g}/{tp:.6g}).")
        save_state(key, state)
        return

    exchange.set_margin_mode(symbol, live_settings.get('margin_mode', 'isolated'))
    exchange.set_leverage(symbol, lev)
    order = exchange.place_market_order_with_tpsl(symbol, 'buy' if side == 'long' else 'sell', qty, sl, tp)
    state['position'] = {'side': side, 'signal_ts': str(last_ts), 'sl': sl, 'tp': tp, 'qty': qty,
                         'entry_ref': entry_ref, 'leverage': lev, 'order_id': (order or {}).get('id')}
    save_state(key, state)
    logger.info(f"ERÖFFNET {side} {qty:.6g} @~{close} | SL {sl:.6g} | TP {tp:.6g} | Hebel {lev}x | Equity {equity:.2f}")
    _notify(telegram, f"{BOT} {symbol} {timeframe}: {side.upper()} eröffnet @~{close:.6g}\n"
                      f"SL {sl:.6g} | TP {tp:.6g} | Hebel {lev}x | Risiko {risk:.2f} %", logger)
