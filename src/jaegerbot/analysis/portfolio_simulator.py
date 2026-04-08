# src/jaegerbot/analysis/portfolio_simulator.py (Version 9 - SuperTrend-Filter & PnL-Cap-Fix)
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import sys
from datetime import datetime
import math # Import für math.ceil

# --- NEUE PFAD-DEFINITION ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from jaegerbot.utils.ann_model import prepare_data_for_ann, create_ann_features
from jaegerbot.analysis.backtester import load_data, calculate_supertrend_direction
from jaegerbot.utils.signal_scorer import SignalScorer, DEFAULT_WEIGHTS, DEFAULT_MIN_SIGNAL_SCORE
# --- ENDE PFAD-DEFINITION ---


# --- HELPER-FUNKTIONEN (ADX entfernt) ---
def get_higher_timeframe(tf):
    return None # Nicht mehr benötigt

# Die Funktion calculate_supertrend_direction wird aus backtester importiert
# --- ENDE HELPER-FUNKTIONEN ---


def run_portfolio_simulation(start_capital, strategies_data, start_date, end_date):
    """
    Führt eine chronologische Portfolio-Simulation mit mehreren Strategien durch.
    """
    print("\n--- Starte Portfolio-Simulation... ---")

    all_signals = []

    # --- START: SuperTrend Trenddaten laden und generieren ---
    print("1/4: Trenddaten (SuperTrend) laden und generieren...")
    st_data_cache = {}
    for key, strat in strategies_data.items():
        data_with_features = strat['data'].copy()
        if not data_with_features.empty:
            st_direction = calculate_supertrend_direction(data_with_features)
            st_data_cache[strat['symbol']] = st_direction
    # --- ENDE SuperTrend Trenddaten laden ---

    print("2/4: Generiere Handelssignale für alle Strategien...")
    for key, strat in strategies_data.items():
        data = strat['data']
        model = strat['model']
        scaler = strat['scaler']
        params = strat['params']
        timeframe = strat['timeframe']
        symbol = strat['symbol']
        
        # Risk/Reward Ratio für die Simulation speichern
        rr_ratio = params.get('risk_reward_ratio', 2.0)

        # Führe Indikator-Berechnung durch
        data_with_features = create_ann_features(data.copy())
        data_with_features.dropna(inplace=True)
        if data_with_features.empty: continue

        # Feature-Spalten definieren (MUSS EXAKT mit ann_model.py übereinstimmen!)
        feature_cols = [
            # Basis-Features
            'bb_width', 'bb_pband', 'obv', 'rsi', 'macd_diff', 'macd',
            'atr_normalized', 'adx', 'adx_pos', 'adx_neg',

            # Volume-Features
            'volume_ratio', 'mfi', 'cmf',

            # Trend-Features
            'price_to_ema20', 'price_to_ema50',

            # Momentum-Features
            'stoch_k', 'stoch_d', 'williams_r', 'roc', 'cci',

            # Support/Resistance
            'price_to_resistance', 'price_to_support',

            # Price Action
            'high_low_range', 'close_to_high', 'close_to_low',

            # Zeitliche Features
            'day_of_week', 'hour_of_day',

            # Returns & Volatilität
            'returns_lag1', 'returns_lag2', 'returns_lag3', 'hist_volatility',

            # Candle DNA Features
            'body_to_atr', 'upper_wick_ratio', 'lower_wick_ratio',
            'candle_direction', 'body_midpoint_ratio',
            'bull_streak', 'bear_streak',

            # Pivot / Structure Features
            'dist_to_struct_high', 'dist_to_struct_low',
            'price_in_range_20', 'price_in_range_50',

            # Fibonacci Zone Features
            'fib_position', 'in_fib_zone',

            # Volume Direction Features
            'volume_direction', 'buying_pressure', 'selling_pressure',
        ]

        # Skalieren und Vorhersage
        features_scaled = scaler.transform(data_with_features[feature_cols])
        predictions = model.predict(features_scaled, verbose=0).flatten()
        data_with_features['prediction'] = predictions
        
        # SuperTrend Direction für den Filter hinzufügen
        st_direction_series = st_data_cache.get(symbol)
        if st_direction_series is None or st_direction_series.empty: continue

        data_with_features['st_direction'] = st_direction_series
        data_with_features.dropna(subset=['st_direction'], inplace=True) 

        pred_threshold = params.get('prediction_threshold', 0.65)
        min_signal_score = params.get('min_signal_score', DEFAULT_MIN_SIGNAL_SCORE)
        scoring_weights = params.get('scoring', DEFAULT_WEIGHTS)
        scorer = SignalScorer(weights=scoring_weights)
        avg_atr_series = data_with_features['atr_normalized'].rolling(50).mean()

        # Sammle alle Signale mit vollem Signal-Scoring (identisch zu trade_manager.py)
        for index, row in data_with_features.iterrows():
            pred = row['prediction']
            st_dir = row.get('st_direction', 0.0)

            if pred >= pred_threshold:
                side = 'long'
            elif pred <= (1 - pred_threshold):
                side = 'short'
            else:
                continue

            # ── EMA Bias Filter (identisch zu backtester.py) ──────────────
            _ema20 = float(row.get('ema20', 0.0))
            _ema50 = float(row.get('ema50', 0.0))
            if _ema20 > 0 and _ema50 > 0:
                if side == 'long' and _ema20 < _ema50:
                    continue
                if side == 'short' and _ema20 > _ema50:
                    continue

            # ── Candle Body Quality Gate (identisch zu backtester.py) ─────
            if float(row.get('body_to_atr', 0.5)) < 0.25:
                continue

            # Signal Scoring — identisch zu trade_manager.py
            avg_atr_norm = avg_atr_series.get(index, data_with_features['atr_normalized'].mean())
            breakdown = scorer.score(
                prediction=pred,
                pred_threshold=pred_threshold,
                side=side,
                st_direction=float(st_dir),
                adx=float(row.get('adx', 0.0)),
                volume_ratio=float(row.get('volume_ratio', 1.0)),
                atr_normalized=float(row.get('atr_normalized', 0.0)),
                avg_atr_normalized=float(avg_atr_norm) if pd.notna(avg_atr_norm) else 0.0,
                body_to_atr=float(row.get('body_to_atr', 0.5)),
                upper_wick_ratio=float(row.get('upper_wick_ratio', 0.3)),
                lower_wick_ratio=float(row.get('lower_wick_ratio', 0.3)),
                dist_to_struct=float(row.get('dist_to_struct_high', 2.0)) if side == 'long' else float(row.get('dist_to_struct_low', 2.0)),
                in_fib_zone=float(row.get('in_fib_zone', 0.0)),
            )
            if breakdown.total < min_signal_score:
                continue

            all_signals.append({
                'timestamp': index, 'symbol': symbol, 'timeframe': timeframe, 'side': side,
                'entry_price': row['close'], 'params': params, 'config_key': key,
                'risk_per_trade_pct': params.get('risk_per_trade_pct', 1.0) / 100,
                'risk_reward_ratio': rr_ratio,
                'atr':               float(row.get('atr', 0.0)),
                'pivot_low_live':    float(row.get('pivot_low_live', 0.0)),
                'pivot_high_live':   float(row.get('pivot_high_live', 0.0)),
                'body_to_atr':       float(row.get('body_to_atr', 0.5)),
                'upper_wick_ratio':  float(row.get('upper_wick_ratio', 0.3)),
                'lower_wick_ratio':  float(row.get('lower_wick_ratio', 0.3)),
                'dist_to_struct_high': float(row.get('dist_to_struct_high', 2.0)),
                'dist_to_struct_low':  float(row.get('dist_to_struct_low', 2.0)),
                'in_fib_zone':       float(row.get('in_fib_zone', 0.0)),
            })

    # Signale auf den gewünschten Zeitraum einschränken (Warmup-Daten herausfiltern)
    start_dt = pd.to_datetime(start_date, utc=True)
    all_signals = [s for s in all_signals if s['timestamp'] >= start_dt]

    if not all_signals:
        print("Keine Handelssignale im gewählten Zeitraum gefunden.")
        return {
            "start_capital": start_capital, "end_capital": start_capital, "total_pnl_pct": 0.0,
            "trade_count": 0, "win_rate": 0.0, "max_drawdown_pct": 0.0,
            "max_drawdown_date": None, "min_equity": start_capital, "liquidation_date": None,
            "equity_curve": pd.DataFrame({'timestamp': [datetime.strptime(start_date, "%Y-%m-%d")], 'equity': [start_capital], 'drawdown_pct': [0.0]})
        }

    all_signals.sort(key=lambda x: x['timestamp'])

    # Kombiniere alle Timeframes für die Equity Curve
    all_timestamps = set()
    for key, strat in strategies_data.items():
        all_timestamps.update(strat['data'].index)
    sorted_timestamps = sorted(list(all_timestamps))

    # --- 3. Chronologische Simulation (mit TitanBot TSL-Logik) ---
    print("3/4: Führe chronologische Backtests durch...")

    equity = start_capital
    peak_equity = start_capital
    max_drawdown_pct = 0.0
    max_drawdown_date = None
    min_equity_ever = start_capital
    liquidation_date = None

    open_positions = {}
    trade_history = []
    equity_curve = []

    fee_pct = 0.05 / 100
    min_notional = 5.0 # Bitget Minimum Notional Value

    signal_idx = 0
    # Wir iterieren über die kombinierten Timestamps für die Positionsbewertung
    for ts in tqdm(sorted_timestamps, desc="Simuliere Portfolio"):
        if liquidation_date: break

        current_total_equity = equity
        unrealized_pnl = 0

        # --- 3a. Offene Positionen managen (Trailing-Stop-Logik wie in backtester.py) ---
        positions_to_close = []
        for key in list(open_positions.keys()):
            pos = open_positions[key]
            # Hier müssen wir über den Symbol-Key auf die Strategie zugreifen
            strat_data = strategies_data.get(pos['symbol_key'])
            if not strat_data or ts not in strat_data['data'].index: continue

            current_candle = strat_data['data'].loc[ts]
            pos['last_known_price'] = current_candle['close']
            exit_price = None
            reason = None

            # Liquidation check (identisch zu backtester.py)
            if pos['side'] == 'long' and current_candle['low'] <= pos.get('liq_price', 0):
                exit_price = pos['liq_price']
                reason = 'liquidation'
            elif pos['side'] == 'short' and current_candle['high'] >= pos.get('liq_price', float('inf')):
                exit_price = pos['liq_price']
                reason = 'liquidation'

            if not reason and pos['side'] == 'long':
                # 1. TSL-Aktivierung prüfen
                if not pos['trailing_active'] and current_candle['high'] >= pos['activation_price']:
                    pos['trailing_active'] = True

                # 2. TSL aktiv: Trailing Stop-Loss anheben
                if pos['trailing_active']:
                    pos['peak_price'] = max(pos['peak_price'], current_candle['high'])
                    trailing_sl = pos['peak_price'] * (1 - pos['callback_rate'])
                    pos['stop_loss'] = max(pos['stop_loss'], trailing_sl)

                # 3. Exit prüfen (Stop-Loss oder Take-Profit)
                if current_candle['low'] <= pos['stop_loss']: exit_price = pos['stop_loss']
                elif not pos['trailing_active'] and current_candle['high'] >= pos['take_profit']: exit_price = pos['take_profit']

            elif not reason and pos['side'] == 'short':
                # 1. TSL-Aktivierung prüfen
                if not pos['trailing_active'] and current_candle['low'] <= pos['activation_price']:
                    pos['trailing_active'] = True

                # 2. TSL aktiv: Trailing Stop-Loss anheben
                if pos['trailing_active']:
                    pos['peak_price'] = min(pos['peak_price'], current_candle['low'])
                    trailing_sl = pos['peak_price'] * (1 + pos['callback_rate'])
                    pos['stop_loss'] = min(pos['stop_loss'], trailing_sl)

                # 3. Exit prüfen (Stop-Loss oder Take-Profit)
                if current_candle['high'] >= pos['stop_loss']: exit_price = pos['stop_loss']
                elif not pos['trailing_active'] and current_candle['low'] <= pos['take_profit']: exit_price = pos['take_profit']


            if exit_price:
                if reason == 'liquidation':
                    # Gesamte Margin verloren + Fees (identisch zu backtester.py)
                    net_pnl = -pos['margin_used'] - (pos['notional_value'] * fee_pct * 2)
                else:
                    pnl_pct = (exit_price / pos['entry_price'] - 1) if pos['side'] == 'long' else (1 - exit_price / pos['entry_price'])
                    pnl_usd = pos['notional_value'] * pnl_pct
                    total_fees = pos['notional_value'] * fee_pct * 2
                    net_pnl = pnl_usd - total_fees
                    # Max. Verlust = riskierter Betrag (identisch zu backtester.py)
                    risk_amount_usd = equity * pos['risk_per_trade_pct']
                    if net_pnl < -risk_amount_usd:
                        net_pnl = -risk_amount_usd
                    
                equity += net_pnl
                
                # Speichere den Config-Key für die Gruppierung
                trade_history.append({
                    'config_key': pos['config_key'],
                    'ts': ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
                    'entry_time': pos['entry_time'].isoformat() if hasattr(pos.get('entry_time', ts), 'isoformat') else str(pos.get('entry_time', ts)),
                    'symbol': pos['symbol_key'],
                    'timeframe': pos.get('timeframe', ''),
                    'direction': pos['side'],
                    'entry': pos['entry_price'],
                    'exit': exit_price,
                    'pnl': net_pnl,
                    'leverage': pos.get('leverage', 0),
                    'margin_used': round(pos.get('margin_used', 0), 4),
                    'min_sl_pct': pos.get('min_sl_pct', 0.0),
                    'max_sl_pct': pos.get('max_sl_pct', 0.0),
                    'tsl_activation_rr': pos.get('activation_rr', 0.0),
                    'tsl_callback_pct': round(pos.get('callback_rate', 0.0) * 100, 3),
                })
                
                positions_to_close.append(key)
            else:
                pnl_mult = 1 if pos['side'] == 'long' else -1
                unrealized_pnl += pos['notional_value'] * (current_candle['close'] / pos['entry_price'] -1) * pnl_mult

        for key in positions_to_close:
            del open_positions[key]

        # --- 3b. Neue Signale prüfen und Positionen eröffnen ---
        while signal_idx < len(all_signals) and all_signals[signal_idx]['timestamp'] == ts:
            signal = all_signals[signal_idx]
            symbol_key = signal['symbol']
            config_key = signal['config_key']
            
            # Schlüssel für die offene Position
            pos_key = f"{symbol_key}_{signal['timeframe']}"

            # Prüfe, ob die Strategie bereits eine offene Position hat
            if pos_key in open_positions:
                signal_idx += 1
                continue

            # Die SuperTrend-Filterung ist bereits in Schritt 2/4 erfolgt (long_signals_filtered)
            trade_allowed = True
            
            if equity <= 0: # Kein Kapital mehr
                signal_idx += 1
                continue

            # Risiko- und Positionsberechnung
            # params ist ein flaches Dict ({**strategy, **risk}) — kein 'risk'-Unterkey
            params = signal['params']
            risk_params = params

            risk_per_trade_pct = signal.get('risk_per_trade_pct', 1.0)
            risk_reward_ratio = signal.get('risk_reward_ratio', 2.0) # Nutze den Wert, den wir im Signal gespeichert haben
            initial_sl_pct = risk_params.get('initial_sl_pct', 1.0) / 100.0
            leverage = risk_params.get('leverage', 10)
            activation_rr = risk_params.get('trailing_stop_activation_rr', 2.0)
            callback_rate = risk_params.get('trailing_stop_callback_rate_pct', 1.0) / 100.0
            
            entry_price = signal['entry_price']
            risk_amount_usd = equity * risk_per_trade_pct

            # Structure-aware SL calculation
            _atr_val    = float(signal.get('atr', entry_price * 0.01))
            _sl_atr     = _atr_val * risk_params.get('atr_multiplier_sl', params.get('atr_multiplier_sl', 2.0))
            _min_sl     = entry_price * (risk_params.get('min_sl_pct', params.get('min_sl_pct', 0.5)) / 100.0)
            _max_sl_pct = risk_params.get('max_sl_pct', params.get('max_sl_pct', 4.0)) / 100.0

            if signal['side'] == 'long':
                _pivot_low  = float(signal.get('pivot_low_live', 0.0))
                _sl_struct  = (entry_price - _pivot_low + 0.25 * _atr_val) if _pivot_low > 0 and _pivot_low < entry_price else _sl_atr
            else:
                _pivot_high = float(signal.get('pivot_high_live', 0.0))
                _sl_struct  = (_pivot_high - entry_price + 0.25 * _atr_val) if _pivot_high > 0 and _pivot_high > entry_price else _sl_atr

            sl_distance = max(_sl_atr, _sl_struct, _min_sl)
            sl_distance = min(sl_distance, entry_price * _max_sl_pct)

            if sl_distance <= 0:
                signal_idx += 1
                continue

            # Berechnung der Positionsgröße basierend auf sl_distance
            notional_value = risk_amount_usd / (sl_distance / entry_price)

            if notional_value < min_notional:
                signal_idx += 1
                continue

            margin_used = notional_value / leverage
            current_total_margin = sum(p['margin_used'] for p in open_positions.values())
            if current_total_margin + margin_used > equity:
                signal_idx += 1
                continue

            # Berechnung der Preislevel
            if signal['side'] == 'long':
                stop_loss = entry_price - sl_distance
                take_profit = entry_price + sl_distance * risk_reward_ratio
                activation_price = entry_price + sl_distance * activation_rr
            else: # short
                stop_loss = entry_price + sl_distance
                take_profit = entry_price - sl_distance * risk_reward_ratio
                activation_price = entry_price - sl_distance * activation_rr

            # Liquidation price (identisch zu backtester.py)
            _mmr = params.get('maintenance_margin_rate', 0.004)
            if signal['side'] == 'long':
                liq_price = entry_price * (1.0 - (1.0 / leverage) + _mmr)
            else:
                liq_price = entry_price * (1.0 + (1.0 / leverage) - _mmr)

            open_positions[pos_key] = {
                'side': signal['side'], 'entry_price': entry_price,
                'stop_loss': stop_loss, 'take_profit': take_profit,
                'liq_price': liq_price,
                'notional_value': notional_value, 'margin_used': margin_used,
                'leverage': leverage,
                'min_sl_pct': risk_params.get('min_sl_pct', 0.0),
                'max_sl_pct': risk_params.get('max_sl_pct', 0.0),
                'activation_rr': activation_rr,
                'callback_rate': callback_rate,
                'trailing_active': False, 'activation_price': activation_price,
                'peak_price': entry_price,
                'last_known_price': entry_price,
                'symbol_key': symbol_key,
                'config_key': config_key,
                'risk_per_trade_pct': risk_per_trade_pct,
                'risk_reward_ratio': risk_reward_ratio,
                'entry_time': ts,
                'timeframe': signal['timeframe'],
            }

            signal_idx += 1

        # --- 3c. Equity Curve und Drawdown aktualisieren ---
        current_total_equity = equity + unrealized_pnl
        equity_curve.append({'timestamp': ts, 'equity': current_total_equity})

        peak_equity = max(peak_equity, current_total_equity)
        drawdown = (peak_equity - current_total_equity) / peak_equity if peak_equity > 0 else 0
        if drawdown > max_drawdown_pct:
            max_drawdown_pct = drawdown
            max_drawdown_date = ts

        min_equity_ever = min(min_equity_ever, current_total_equity)
        if current_total_equity <= 0 and not liquidation_date:
            liquidation_date = ts

    # 4/4: Ergebnisse vorbereiten
    print("4/4: Bereite Analyse-Ergebnisse vor...")
    final_equity = equity_curve[-1]['equity'] if equity_curve else start_capital
    total_pnl_pct = (final_equity / start_capital - 1) * 100 if start_capital > 0 else 0
    wins = sum(1 for t in trade_history if t['pnl'] > 0)
    trade_count = len(trade_history)
    win_rate = (wins / trade_count * 100) if trade_count else 0

    trade_df = pd.DataFrame(trade_history)
    strategy_key_col = 'config_key' 

    pnl_per_strategy = trade_df.groupby(strategy_key_col)['pnl'].sum().reset_index() if not trade_df.empty else pd.DataFrame(columns=[strategy_key_col, 'pnl'])
    trades_per_strategy = trade_df.groupby(strategy_key_col).size().reset_index(name='trades') if not trade_df.empty else pd.DataFrame(columns=[strategy_key_col, 'trades'])

    equity_df = pd.DataFrame(equity_curve)
    if not equity_df.empty:
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown_pct'] = ((equity_df['peak'] - equity_df['equity']) / equity_df['peak'].replace(0, np.nan)).fillna(0)
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
        equity_df.set_index('timestamp', inplace=True, drop=False)

    print("Analyse abgeschlossen.")

    return {
        "start_capital": start_capital, "end_capital": final_equity, "total_pnl_pct": total_pnl_pct,
        "trade_count": trade_count, "win_rate": win_rate, "max_drawdown_pct": max_drawdown_pct * 100,
        "max_drawdown_date": max_drawdown_date, "min_equity": min_equity_ever, "liquidation_date": liquidation_date,
        "pnl_per_strategy": pnl_per_strategy, "trades_per_strategy": trades_per_strategy,
        "equity_curve": equity_df,
        "trade_history": trade_history,
    }
