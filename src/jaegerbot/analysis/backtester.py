# src/jaegerbot/analysis/backtester.py
import os
import pandas as pd
import numpy as np
from datetime import timedelta
import json
import sys
import ta # Import für ATR/ADX benötigt
import math # Import für math.ceil

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from jaegerbot.utils.exchange import Exchange
from jaegerbot.utils.ann_model import prepare_data_for_ann, load_model_and_scaler, create_ann_features
from jaegerbot.utils.supertrend_indicator import SuperTrendLocal
from jaegerbot.utils.signal_scorer import SignalScorer, DEFAULT_WEIGHTS, DEFAULT_MIN_SIGNAL_SCORE

# --- load_data und get_higher_timeframe sind unverändert ---
def load_data(symbol, timeframe, start_date_str, end_date_str):
    cache_dir = os.path.join(PROJECT_ROOT, 'data', 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    symbol_filename = symbol.replace('/', '-').replace(':', '-')
    cache_file = os.path.join(cache_dir, f"{symbol_filename}_{timeframe}.csv")
    if os.path.exists(cache_file):
        data = pd.read_csv(cache_file, index_col='timestamp', parse_dates=True)
        try:
            if data.index.min() <= pd.to_datetime(start_date_str, utc=True) and data.index.max() >= pd.to_datetime(end_date_str, utc=True):
                return data.loc[start_date_str:end_date_str]
        except Exception:
            pass
    print(f"Starte Download für {symbol} ({timeframe}) von der Börse...")
    try:
        with open(os.path.join(PROJECT_ROOT, 'secret.json'), "r") as f: secrets = json.load(f)
        api_setup = secrets.get('jaegerbot')[0]
        exchange = Exchange(api_setup)
        full_data = exchange.fetch_historical_ohlcv(symbol, timeframe, start_date_str, end_date_str)
        if not full_data.empty:
            full_data.to_csv(cache_file)
            return full_data
    except Exception as e:
        print(f"Fehler beim Daten-Download: {e}")
    return pd.DataFrame()

def get_higher_timeframe(tf):
    """Wählt einen passenden höheren Zeitrahmen für den Filter."""
    # --- KORREKTUR: ADX/HTF-Logik entfernt, Funktion bleibt nur als Platzhalter
    if 'm' in tf: return '1h'
    if tf == '1h': return '4h'
    if tf in ['2h', '4h', '6h']: return '1d'
    if tf == '1d': return None
    return '1d'
    # ---

# *** NEUE HELFERFUNKTION: SuperTrend-Richtung berechnen ***
def calculate_supertrend_direction(data):
    """Berechnet die SuperTrend-Richtung für den Trendfilter."""
    st_indicator = SuperTrendLocal(data['high'], data['low'], data['close'], window=10, multiplier=3.0)
    # 1.0 für Long-Trend, -1.0 für Short-Trend
    return st_indicator.get_supertrend_direction().shift(1) # Shift, um den ST der VORHERIGEN Kerze zu verwenden

# *** KORRIGIERTE BACKTESTER FUNKTION (SuperTrend-Filter) ***
def run_ann_backtest(data, params, model_paths, start_capital=1000, use_macd_filter=False, htf_data=None, timeframe=None, verbose=False, params_for_htf_load=None, precomputed=False):

    if not timeframe:
        raise ValueError("Backtester benötigt ein 'timeframe' Argument für die Daten-Vorbereitung!")

    if precomputed:
        # Predictions wurden vom Optimizer vorberechnet — Modell-Load + predict überspringen
        data_with_features = data
        if data_with_features.empty or len(data_with_features) < 10:
            return {"total_pnl_pct": 0, "trades_count": 0, "win_rate": 0, "max_drawdown_pct": 1.0, "end_capital": start_capital}
    else:
        model, scaler = load_model_and_scaler(model_paths['model'], model_paths['scaler'])
        if not model or not scaler: raise Exception("Modell/Scaler nicht gefunden!")

        data_with_features = create_ann_features(data.copy())
        data_with_features.dropna(inplace=True)

        if data_with_features.empty or len(data_with_features) < 10:
            return {"total_pnl_pct": 0, "trades_count": 0, "win_rate": 0, "max_drawdown_pct": 1.0, "end_capital": start_capital}

        # SuperTrend Richtung der VORHERIGEN Kerze vorberechnen
        data_with_features['supertrend_direction'] = calculate_supertrend_direction(data_with_features)
        # Rolling-Mittel für Scorer vorberechnen (spart Loop-Overhead)
        data_with_features['avg_atr_normalized'] = data_with_features['atr_normalized'].rolling(50).mean()
        data_with_features.dropna(inplace=True)

        if data_with_features.empty:
            return {"total_pnl_pct": 0, "trades_count": 0, "win_rate": 0, "max_drawdown_pct": 1.0, "end_capital": start_capital}

        feature_cols = [
            'bb_width', 'bb_pband', 'obv', 'rsi', 'macd_diff', 'macd',
            'atr_normalized', 'adx', 'adx_pos', 'adx_neg',
            'volume_ratio', 'mfi', 'cmf',
            'price_to_ema20', 'price_to_ema50',
            'stoch_k', 'stoch_d', 'williams_r', 'roc', 'cci',
            'price_to_resistance', 'price_to_support',
            'high_low_range', 'close_to_high', 'close_to_low',
            'day_of_week', 'hour_of_day',
            'returns_lag1', 'returns_lag2', 'returns_lag3', 'hist_volatility',
            'body_to_atr', 'upper_wick_ratio', 'lower_wick_ratio',
            'candle_direction', 'body_midpoint_ratio',
            'bull_streak', 'bear_streak',
            'dist_to_struct_high', 'dist_to_struct_low',
            'price_in_range_20', 'price_in_range_50',
            'fib_position', 'in_fib_zone',
            'volume_direction', 'buying_pressure', 'selling_pressure',
        ]
        missing_cols = [col for col in feature_cols if col not in data_with_features.columns]
        if missing_cols:
            raise ValueError(f"Fehlende Spalten in data_with_features: {missing_cols}")

        features_scaled = scaler.transform(data_with_features[feature_cols])
        predictions = model.predict(features_scaled, verbose=0).flatten()
        data_with_features['prediction'] = pd.Series(predictions, index=data_with_features.index)

    pred_threshold = params.get('prediction_threshold', 0.6)
    risk_reward_ratio = params.get('risk_reward_ratio', 1.5)
    risk_per_trade_pct = params.get('risk_per_trade_pct', 1.0) / 100

    # TSL-Parameter aus Configs
    activation_rr = params.get('trailing_stop_activation_rr', 2.0)
    callback_rate = params.get('trailing_stop_callback_rate_pct', 1.0) / 100

    # Dynamische ATR-basierte SL-Parameter (konsistent mit trade_manager.py)
    atr_multiplier_sl = params.get('atr_multiplier_sl', 2.0)
    min_sl_pct = params.get('min_sl_pct', 0.5) / 100.0

    # Signal-Scoring Parameter
    min_signal_score = params.get('min_signal_score', DEFAULT_MIN_SIGNAL_SCORE)
    scorer_weights = {
        'ann_weight':        params.get('ann_weight',       DEFAULT_WEIGHTS['ann_weight']),
        'volume_weight':     params.get('volume_weight',    DEFAULT_WEIGHTS['volume_weight']),
        'structure_weight':  params.get('structure_weight', DEFAULT_WEIGHTS.get('structure_weight', 1.0)),
        'st_weight':         DEFAULT_WEIGHTS['st_weight'],
        'adx_weight':        DEFAULT_WEIGHTS['adx_weight'],
        'volatility_weight': DEFAULT_WEIGHTS['volatility_weight'],
    }
    scorer = SignalScorer(weights=scorer_weights)

    leverage = params.get('leverage', 10)
    fee_pct = 0.05 / 100

    current_capital, trades_count, wins_count = start_capital, 0, 0
    peak_capital, max_drawdown_pct = start_capital, 0.0
    position = None
    trades = []           # Sammle alle Trades für Chart-Darstellung
    equity_snapshots = [] # Kapitalverlauf für Equity-Kurve

    # --- KORREKTUR: ADX / HTF-Filter-Initialisierung entfernt ---
    # Entferne die Lade-Logik für HTF-Daten, da der ADX-Filter entfernt wurde
    # ---

    for i in range(len(data_with_features)):
        current = data_with_features.iloc[i]

        if position:
            exit_price, reason = None, None

            # *** Liquidation check (vor SL/TSL — tritt bei höherem Leverage früher ein) ***
            if position['side'] == 'long' and current['low'] <= position['liq_price']:
                exit_price = position['liq_price']
                reason = 'liquidation'
            elif position['side'] == 'short' and current['high'] >= position['liq_price']:
                exit_price = position['liq_price']
                reason = 'liquidation'

            # *** SL / TP Logik ***
            if not reason:
                if position['side'] == 'long':
                    if current['low'] <= position['stop_loss']:
                        exit_price = position['stop_loss']
                    elif current['high'] >= position['take_profit']:
                        exit_price = position['take_profit']
                elif position['side'] == 'short':
                    if current['high'] >= position['stop_loss']:
                        exit_price = position['stop_loss']
                    elif current['low'] <= position['take_profit']:
                        exit_price = position['take_profit']
            # *** Ende SL / TP Logik ***

            if exit_price:
                if reason == 'liquidation':
                    # Gesamte Margin verloren + Fees
                    net_pnl = -(position['margin_used']) - (position['notional_value'] * fee_pct * 2)
                else:
                    pnl_pct = (exit_price / position['entry_price'] - 1) if position['side'] == 'long' else (1 - exit_price / position['entry_price'])
                    notional_value = position['margin_used'] * leverage
                    pnl_usd = notional_value * pnl_pct
                    total_fees = notional_value * fee_pct * 2
                    risk_amount_usd = current_capital * risk_per_trade_pct
                    net_pnl = pnl_usd - total_fees
                    if net_pnl < -risk_amount_usd:
                        net_pnl = -risk_amount_usd
                    
                current_capital += net_pnl
                
                # *** SAMMLE TRADE FÜR CHART-ANZEIGE ***
                if position['side'] == 'long':
                    trades.append({
                        'entry_long': {'time': position['entry_time'], 'price': position['entry_price']},
                        'exit_long': {'time': data_with_features.index[i], 'price': exit_price}
                    })
                else:  # short
                    trades.append({
                        'entry_short': {'time': position['entry_time'], 'price': position['entry_price']},
                        'exit_short': {'time': data_with_features.index[i], 'price': exit_price}
                    })
                # *** ENDE TRADE-SAMMLUNG ***
                
                if net_pnl > 0: wins_count += 1
                trades_count += 1
                position = None
                equity_snapshots.append({'timestamp': data_with_features.index[i], 'equity': current_capital})
                peak_capital = max(peak_capital, current_capital)
                if peak_capital > 0:
                    drawdown = (peak_capital - current_capital) / peak_capital
                    max_drawdown_pct = max(max_drawdown_pct, drawdown)
                if current_capital <= 0: break
                continue  # Kein Re-Entry auf derselben Kerze nach Trade-Schluss

        if not position:
            side = 'long' if current['prediction'] >= pred_threshold else 'short' if current['prediction'] <= (1 - pred_threshold) else None

            if side:
                # ── EMA Bias Filter (stbot MTF-inspired) ─────────────────────────
                # Nur Longs wenn EMA20 > EMA50 (Aufwärtstrend), nur Shorts umgekehrt.
                # Verhindert Gegentrend-Trades → verbessert Win-Rate deutlich.
                _ema20 = float(current.get('ema20', 0.0))
                _ema50 = float(current.get('ema50', 0.0))
                if _ema20 > 0 and _ema50 > 0:
                    if side == 'long' and _ema20 < _ema50:
                        continue
                    if side == 'short' and _ema20 > _ema50:
                        continue

                # ── Candle Body Quality Gate (vbot-inspired) ──────────────────────
                # Doji/Indecision-Kerzen (body < 25% ATR) → kein Trade.
                if float(current.get('body_to_atr', 0.5)) < 0.25:
                    continue

                # --- SIGNAL SCORING (konsistent mit trade_manager.py) ---
                _body_to_atr     = float(current.get('body_to_atr',     0.5))
                _upper_wick      = float(current.get('upper_wick_ratio', 0.3))
                _lower_wick      = float(current.get('lower_wick_ratio', 0.3))
                _in_fib          = float(current.get('in_fib_zone',      0.0))
                _dist_struct     = float(current.get('dist_to_struct_high', 2.0)) if side == 'long' else float(current.get('dist_to_struct_low', 2.0))
                breakdown = scorer.score(
                    prediction=current['prediction'],
                    pred_threshold=pred_threshold,
                    side=side,
                    st_direction=current['supertrend_direction'],
                    adx=current.get('adx', 0.0),
                    volume_ratio=current.get('volume_ratio', 1.0),
                    atr_normalized=current.get('atr_normalized', 0.0),
                    avg_atr_normalized=current.get('avg_atr_normalized', current.get('atr_normalized', 0.0)),
                    body_to_atr=_body_to_atr,
                    upper_wick_ratio=_upper_wick,
                    lower_wick_ratio=_lower_wick,
                    dist_to_struct=_dist_struct,
                    in_fib_zone=_in_fib,
                )
                trade_allowed = breakdown.total >= min_signal_score
                # --- ENDE SIGNAL SCORING ---

                # Minimum ANN gate
                _min_ann = params.get('min_ann_score', 1.0)
                if breakdown.ann < _min_ann:
                    continue

                if trade_allowed:
                    entry_price = current['close']
                    entry_time = data_with_features.index[i]
                    risk_amount_usd = current_capital * risk_per_trade_pct

                    # Structure-aware ATR-based SL calculation
                    _atr_val    = float(current.get('atr', entry_price * 0.01))
                    if _atr_val <= 0:
                        continue
                    _sl_atr     = _atr_val * atr_multiplier_sl
                    _min_sl     = entry_price * min_sl_pct
                    _max_sl_pct = params.get('max_sl_pct', 4.0) / 100.0

                    if side == 'long':
                        _pivot_low = float(current.get('pivot_low_live', 0.0))
                        _sl_struct = (entry_price - _pivot_low + 0.25 * _atr_val) if _pivot_low > 0 else _sl_atr
                    else:
                        _pivot_high = float(current.get('pivot_high_live', 0.0))
                        _sl_struct = (_pivot_high - entry_price + 0.25 * _atr_val) if _pivot_high > 0 else _sl_atr

                    sl_distance = max(_sl_atr, _sl_struct, _min_sl)
                    sl_distance = min(sl_distance, entry_price * _max_sl_pct)
                    if sl_distance == 0:
                        continue

                    notional_value = risk_amount_usd / (sl_distance / entry_price)
                    margin_used = notional_value / leverage
                    if margin_used > current_capital:
                        continue

                    stop_loss = entry_price - sl_distance if side == 'long' else entry_price + sl_distance
                    take_profit = entry_price + sl_distance * risk_reward_ratio if side == 'long' else entry_price - sl_distance * risk_reward_ratio
                    activation_price = entry_price + sl_distance * activation_rr if side == 'long' else entry_price - sl_distance * activation_rr

                    # Liquidation price (isolated margin)
                    _mmr = params.get('maintenance_margin_rate', 0.004)
                    if side == 'long':
                        liq_price = entry_price * (1.0 - (1.0 / leverage) + _mmr)
                    else:
                        liq_price = entry_price * (1.0 + (1.0 / leverage) - _mmr)

                    position = {
                        'side': side,
                        'entry_price': entry_price,
                        'entry_time': entry_time,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'liq_price': liq_price,
                        'margin_used': margin_used,
                        'notional_value': notional_value,
                        'trailing_active': False,
                        'activation_price': activation_price,
                        'peak_price': entry_price,
                        'callback_rate': callback_rate,
                    }

    win_rate = (wins_count / trades_count * 100) if trades_count > 0 else 0
    final_pnl_pct = ((current_capital - start_capital) / start_capital) * 100 if start_capital > 0 else 0
    return {"total_pnl_pct": final_pnl_pct, "trades_count": trades_count, "win_rate": win_rate, "max_drawdown_pct": max_drawdown_pct, "end_capital": current_capital, "trades": trades, "equity_snapshots": equity_snapshots}
