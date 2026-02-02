# src/jaegerbot/utils/trade_manager.py (KORRIGIERTE VERSION - Bereinigt)
import logging
import time
import ccxt
import os
import json
import pandas as pd
import ta
import math

from jaegerbot.utils.telegram import send_message
from jaegerbot.utils.ann_model import create_ann_features
from jaegerbot.utils.exchange import Exchange
from jaegerbot.utils.supertrend_indicator import SuperTrendLocal

# Pfade für die Lock-Datei definieren
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
LOCK_FILE_PATH = os.path.join(PROJECT_ROOT, 'artifacts', 'db', 'trade_lock.json')

# --------------------------------------------------------------------------- #
# Trade-Lock-Hilfsfunktionen (Unverändert)
# --------------------------------------------------------------------------- #
def get_trade_lock(strategy_id):
    """Liest den Zeitstempel des letzten Trades für eine Strategie aus der Lock-Datei."""
    if not os.path.exists(LOCK_FILE_PATH):
        return None
    try:
        with open(LOCK_FILE_PATH, 'r') as f:
            locks = json.load(f)
        # NEU: Der gespeicherte Wert ist der Zeitstempel der Kerze, die gehandelt wurde
        return locks.get(strategy_id)
    except (json.JSONDecodeError, FileNotFoundError):
        return None

def set_trade_lock(strategy_id, candle_timestamp):
    """Setzt eine Sperre für eine Strategie, um erneutes Handeln auf derselben Kerze zu verhindern."""
    os.makedirs(os.path.dirname(LOCK_FILE_PATH), exist_ok=True)
    locks = {}
    if os.path.exists(LOCK_FILE_PATH):
        try:
            with open(LOCK_FILE_PATH, 'r') as f:
                locks = json.load(f)
        except json.JSONDecodeError:
            locks = {}
    # Speichere nur den Zeitstempel der Kerze
    locks[strategy_id] = candle_timestamp.strftime('%Y-%m-%d %H:%M:%S')
    with open(LOCK_FILE_PATH, 'w') as f:
        json.dump(locks, f, indent=4)


# --------------------------------------------------------------------------- #
# Housekeeper & Helper (Unverändert)
# --------------------------------------------------------------------------- #

def housekeeper_routine(exchange, symbol, logger):
    """Storniert alle offenen Orders FÜR EIN SYMBOL und versucht, die Position zu schließen, falls verwaist."""
    logger.info(f"Starte Aufräum-Routine für {symbol}...")

    # 1. Alle ORDERS stornieren (robust)
    try:
        cancelled_count = exchange.cleanup_all_open_orders(symbol)
        if cancelled_count > 0:
            logger.info(f"{cancelled_count} verwaiste Order(s) gefunden und storniert.")
    except Exception as e:
        logger.error(f"Fehler während der Order-Aufräumung: {e}")

    # 2. Position prüfen und schließen (wichtig für Teardown/Fallback)
    try:
        position = exchange.fetch_open_positions(symbol)
        if position:
            pos_info = position[0]
            close_side = 'sell' if pos_info['side'] == 'long' else 'buy'
            contracts = float(pos_info['contracts'])

            logger.warning(f"Housekeeper: Schließe verwaiste Position ({pos_info['side']} {contracts:.6f})...")
            exchange.create_market_order(symbol, close_side, contracts, {'reduceOnly': True})
            time.sleep(2)

            if exchange.fetch_open_positions(symbol):
                logger.error("Housekeeper: Position konnte nicht geschlossen werden!")
                return False
            else:
                logger.info(f"Housekeeper: {symbol} ist jetzt sauber.")
                return True
        else:
            logger.info(f"Housekeeper: {symbol} ist jetzt sauber.")
            return True
    except Exception as e:
        logger.error(f"Housekeeper-Fehler beim Positions-Management: {e}", exc_info=True)
        return False


# --------------------------------------------------------------------------- #
# Hauptfunktion: Trade öffnen (mit dynamischem SL)
# --------------------------------------------------------------------------- #
def check_and_open_new_position(exchange: Exchange, model, scaler, params, telegram_config, logger):

    symbol = params['market']['symbol']
    timeframe = params['market']['timeframe']
    strategy_id = f"{symbol.replace('/', '').replace(':', '')}_{timeframe}"
    account_name = exchange.account.get('name', 'Standard-Account')
    
    logger.info("Suche nach neuen Signalen...")
    data = exchange.fetch_recent_ohlcv(symbol, timeframe, limit=500)

    if len(data) < 2:
        logger.warning("Nicht genug Daten geladen. Überspringe.")
        return

    last_candle_timestamp = data.index[-2]
    last_trade_timestamp_str = get_trade_lock(strategy_id)
    if last_trade_timestamp_str and last_trade_timestamp_str == last_candle_timestamp.strftime('%Y-%m-%d %H:%M:%S'):
        logger.info(f"Signal für Kerze {last_candle_timestamp} wurde bereits gehandelt. Überspringe bis zur nächsten Kerze.")
        return

    data_with_features = create_ann_features(data.copy())

    # --- SuperTrend Richtung hinzufügen ---
    st_indicator = SuperTrendLocal(data_with_features['high'], data_with_features['low'], data_with_features['close'], window=10, multiplier=3.0)
    # Verwende die Richtung der VORHERIGEN Kerze
    st_direction = st_indicator.get_supertrend_direction().iloc[-2]
    # ---

    # *** ERWEITERTE FEATURE-LISTE ***
    # Feature 'ema_cross_20_50' entfernt (konsistent mit ann_model.py)
    feature_cols = [
        'bb_width', 'bb_pband', 'obv', 'rsi', 'macd_diff', 'macd', 
        'atr_normalized', 'adx', 'adx_pos', 'adx_neg',
        'volume_ratio', 'mfi', 'cmf',
        'price_to_ema20', 'price_to_ema50',
        'stoch_k', 'stoch_d', 'williams_r', 'roc', 'cci',
        'price_to_resistance', 'price_to_support',
        'high_low_range', 'close_to_high', 'close_to_low',
        'day_of_week', 'hour_of_day',
        'returns_lag1', 'returns_lag2', 'returns_lag3', 'hist_volatility'
    ]
    # ---

    latest_features = data_with_features.iloc[-2:-1][feature_cols]

    if latest_features.isnull().values.any():
        logger.warning("Neueste Feature-Daten sind unvollständig, überspringe diesen Zyklus.")
        return

    scaled_features = scaler.transform(latest_features)
    prediction = model.predict(scaled_features, verbose=0)[0][0]
    pred_threshold = params['strategy']['prediction_threshold']
    side = None
    signal_reason = None

    # --- ANN-Signal prüfen und loggen ---
    if prediction >= pred_threshold and params.get('behavior', {}).get('use_longs', True):
        side = 'buy'
        signal_reason = f"Modell-Vorhersage {prediction:.3f} >= Schwelle {pred_threshold:.3f} (LONG)"
    elif prediction <= (1 - pred_threshold) and params.get('behavior', {}).get('use_shorts', True):
        side = 'sell'
        signal_reason = f"Modell-Vorhersage {prediction:.3f} <= Schwelle {1 - pred_threshold:.3f} (SHORT)"
    else:
        signal_reason = f"Modell-Vorhersage {prediction:.3f} -> Kein gültiges Signal (LONG/SHORT) für Schwellen {pred_threshold:.3f}/{1 - pred_threshold:.3f}"

    logger.info(f"Signal-Entscheidung für {symbol} @ {last_candle_timestamp}: {side if side else 'NEUTRAL'} | Grund: {signal_reason}")

    # --- SUPER TREND FILTER: Trendbestätigung und Richtung erzwingen ---
    trade_allowed = True
    if side == 'buy':
        if st_direction != 1.0:
            trade_allowed = False
            logger.info(f"Signal (LONG) für {symbol} @ {last_candle_timestamp} deaktiviert durch SuperTrend-Filter: Kein Long-Trend (st_direction={st_direction})")
    elif side == 'sell':
        if st_direction != -1.0:
            trade_allowed = False
            logger.info(f"Signal (SHORT) für {symbol} @ {last_candle_timestamp} deaktiviert durch SuperTrend-Filter: Kein Short-Trend (st_direction={st_direction})")
    # --- ENDE SUPER TREND FILTER ---

    # *** NEUE FILTER: ADX & VOLUME & VOLATILITÄT ***
    if side and trade_allowed:
        last_candle = data_with_features.iloc[-2]
        # ADX-Filter: Nur bei ausreichender Trendstärke traden
        current_adx = last_candle.get('adx', 0)
        if current_adx < 20:
            trade_allowed = False
            logger.info(f"Signal ({side.upper()}) für {symbol} @ {last_candle_timestamp} deaktiviert durch ADX-Filter: ADX zu niedrig ({current_adx:.1f} < 20)")
        # Volume-Filter: Mindestens 80% des Average Volume
        if 'volume' in data_with_features.columns:
            current_volume = last_candle['volume']
            avg_volume = data_with_features['volume'].rolling(20).mean().iloc[-2]
            if current_volume < avg_volume * 0.8:
                trade_allowed = False
                logger.info(f"Signal ({side.upper()}) für {symbol} @ {last_candle_timestamp} deaktiviert durch Volumen-Filter: Volume zu niedrig ({current_volume:.0f} < {avg_volume*0.8:.0f})")
        # Volatilitäts-Filter: Keine extremen Spikes
        current_atr_norm = last_candle.get('atr_normalized', 0)
        avg_atr_norm = data_with_features['atr_normalized'].rolling(50).mean().iloc[-2]
        if current_atr_norm > avg_atr_norm * 2.0:
            trade_allowed = False
            logger.info(f"Signal ({side.upper()}) für {symbol} @ {last_candle_timestamp} deaktiviert durch Volatilitäts-Filter: ATR {current_atr_norm:.2f}% > {avg_atr_norm*2:.2f}%")
    # *** ENDE NEUE FILTER ***

    if side and trade_allowed:
        logger.info(f"Gültiges Signal '{side.upper()}' für {symbol} @ {last_candle_timestamp} erkannt (Grund: {signal_reason}). Trade-Eröffnung wird gestartet.")
        p = params['risk']
        # Basis-Risk aus Config
        base_risk_pct = p['risk_per_trade_pct']
        applied_risk_pct = base_risk_pct

        risk_per_trade_pct = applied_risk_pct / 100.0
        risk_reward_ratio = p['risk_reward_ratio']
        # --- NEU: Dynamische SL-Parameter lesen ---
        # Nutze 0.5% als Fallback für min_sl_pct, falls es in alten Configs fehlt
        min_sl_pct = p.get('min_sl_pct', 0.5) / 100.0
        atr_multiplier_sl = p.get('atr_multiplier_sl', 2.0)
        # --- ENDE NEU ---
        leverage = p['leverage']
        activation_rr = p.get('trailing_stop_activation_rr', 2.0)
        callback_rate_pct = p.get('trailing_stop_callback_rate_pct', 1.0) / 100.0

        current_balance = exchange.fetch_balance_usdt()
        if current_balance <= 0: logger.error("Kein Guthaben zum Eröffnen."); return

        risk_amount_usd = current_balance * risk_per_trade_pct
        ticker = exchange.fetch_ticker(symbol)
        entry_price = ticker['last']
        
        # --- NEU: DYNAMISCHE SL-DISTANZ-BERECHNUNG (wie TitanBot) ---
        last_candle = data_with_features.iloc[-2] # Nimm die abgeschlossene Kerze
        current_atr = last_candle.get('atr', 0.0)
        
        if current_atr <= 0:
            logger.error("ATR ist Null oder ungültig. Kann dynamischen SL nicht setzen."); return

        sl_distance_atr = current_atr * atr_multiplier_sl
        sl_distance_min = entry_price * min_sl_pct
        sl_distance = max(sl_distance_atr, sl_distance_min)
        # --- ENDE DYNAMISCHE SL-DISTANZ-BERECHNUNG ---

        if sl_distance == 0: logger.error("SL-Distanz Null."); return

        # Neuberechnung der Positionsgröße basierend auf der DYNAMISCHEN SL-Distanz
        notional_value = risk_amount_usd / (sl_distance / entry_price)

        # Override: Für Tests/CI kann ein fixes Notional (in USDT) per ENV oder params erzwungen werden
        # Priorität: params['risk']['force_notional_usdt'] > ENV JAEGER_PEPE_NOTIONAL_USDT
        force_notional = None
        try:
            force_notional = p.get('force_notional_usdt') if isinstance(p, dict) else None
        except Exception:
            force_notional = None

        if not force_notional:
            env_force = os.getenv('JAEGER_PEPE_NOTIONAL_USDT') or os.getenv('JAEGER_FORCE_NOTIONAL_USDT')
            if env_force:
                try:
                    force_notional = float(env_force)
                except Exception:
                    force_notional = None

        if force_notional and 'PEPE' in symbol.upper():
            # Verwende das erzwungene Notional (USDT) statt Risiko-basiertem Notional
            notional_value = float(force_notional)

        amount = notional_value / entry_price

        # Berechnung der Trigger-Preise
        stop_loss_price = entry_price - sl_distance if side == 'buy' else entry_price + sl_distance
        activation_price = entry_price + sl_distance * activation_rr if side == 'buy' else entry_price - sl_distance * activation_rr

        tsl_side = 'sell' if side == 'buy' else 'buy'
        # Der fixe TP wird nur für die Message benötigt, aber nicht als Order platziert
        take_profit_price = entry_price + sl_distance * risk_reward_ratio if side == 'buy' else entry_price - sl_distance * risk_reward_ratio

        # --- Trade-Eröffnung ---
        try:
            # Setze zuerst Margin Mode (isolated) bevor Hebel, damit Bitget den Modus korrekt übernimmt
            if not exchange.set_margin_mode(symbol, p.get('margin_mode', 'isolated')): return
            if not exchange.set_leverage(symbol, leverage, p.get('margin_mode', 'isolated')): return
                # Circuit Breaker deaktiviert – kein Risk-Scaling

            time.sleep(2)

            final_position = exchange.fetch_open_positions(symbol)
            if not final_position: raise Exception("Position konnte nicht bestätigt werden.")
            final_amount = float(final_position[0]['contracts'])

            sl_rounded = float(exchange.exchange.price_to_precision(symbol, stop_loss_price))
            activation_price_rounded = float(exchange.exchange.price_to_precision(symbol, activation_price))

            # --- 1. Fixen SL setzen (PRIORITÄT) ---
            logger.info(f"Platziere FIXEN SL @ {sl_rounded} (kritische Sicherheit).")
            exchange.place_trigger_market_order(
                symbol, tsl_side, final_amount, sl_rounded, {'reduceOnly': True}
            )

            # --- 2. TSL als dynamischen TP setzen (Sekundär) ---
            tsl_placed = False
            try:
                logger.info(f"Platziere TSL (ersetzt TP): Aktivierung @ {activation_price_rounded}, Callback @ {callback_rate_pct*100:.2f}%")
                tsl_order = exchange.place_trailing_stop_order(
                    symbol,
                    tsl_side,
                    final_amount,
                    activation_price_rounded,
                    callback_rate_pct,
                    {'reduceOnly': True}
                )
                if tsl_order:
                    tsl_placed = True
            except Exception as inner_e:
                logger.warning(f"WARNUNG: TSL-Platzierung fehlgeschlagen. Fixer SL sollte aktiv sein. Fehler: {inner_e}")

            # --- Erfolgsnachricht senden ---
            set_trade_lock(strategy_id, last_candle_timestamp)

            tsl_status = f"TSL aktiv (Aktivierung @ ${activation_price_rounded:.4f})" if tsl_placed else "KEIN TSL aktiv (nur fixer SL)"

            # Erweiterte Telegram-Nachricht (ähnlich der gewünschten Vorlage)
            message = (
                f"🧠 ANN Signal für *{account_name}* ({symbol}, {side.upper()})\n"
                f"- Entry @ Market (≈${entry_price:.4f})\n"
                f"- Positionswert: ${notional_value:.2f} | Menge: {final_amount:.4f} Contracts\n"
                f"- SL: ${sl_rounded:.4f} (DYNAMISCHE SICHERHEIT)\n"
                f"- TP: , {tsl_status}\n"
                f"- Hebel: {leverage}x | Margin Mode: {p.get('margin_mode', 'isolated')}"
            )
            sent = send_message(telegram_config.get('bot_token'), telegram_config.get('chat_id'), message)
            if sent:
                logger.info("Telegram-Benachrichtigung: Trade-Eröffnung gesendet.")
            else:
                logger.warning("Telegram-Benachrichtigung konnte nicht gesendet werden.")
            logger.info(f"Trade-Eröffnungsprozess abgeschlossen (SL gesetzt{tsl_status}).")


        except Exception as e:
            # KRITISCHER FEHLERFALL: Position ist möglicherweise offen, aber ungeschützt.
            logger.error(f"FEHLER beim Eröffnen/SL-Platzierung: {e}")

            final_position_after_error = exchange.fetch_open_positions(symbol)
            if final_position_after_error:
                # Schließen der Position, da wir nicht garantieren können, dass der SL sitzt
                logger.critical("Position konnte nicht geschützt werden! Starte Notfallschließung.")
                housekeeper_routine(exchange, symbol, logger) # Schließt die Position

                fallback_msg = (f"❌ *Kritisch: Position geschlossen*\n"
                                f"SL-Platzierung/Trade-Eröffnung fehlgeschlagen für *{symbol}*. "
                                f"Die Position wurde ZUR SICHERHEIT geschlossen.")
                sent = send_message(telegram_config.get('bot_token'), telegram_config.get('chat_id'), fallback_msg)
                if sent:
                    logger.info("Telegram-Benachrichtigung (Fallback): Nachricht gesendet.")
                else:
                    logger.warning("Telegram-Benachrichtigung (Fallback) konnte nicht gesendet werden.")
            else:
                logger.info("Keine Position nach Fehler gefunden. Alles geschlossen.")


def full_trade_cycle(exchange, model, scaler, params, telegram_config, logger):
    """Der Haupt-Handelszyklus für eine einzelne Strategie."""
    symbol = params['market']['symbol']
    try:
        position = exchange.fetch_open_positions(symbol)
        position = position[0] if position else None

        if not position:
            if not housekeeper_routine(exchange, symbol, logger):
                logger.error("Housekeeper konnte die Umgebung nicht säubern. Breche ab.")
                return
            check_and_open_new_position(exchange, model, scaler, params, telegram_config, logger)
        else:
            logger.info(f"Offene Position für {symbol} gefunden. Warte auf SL/TSL/TP-Trigger.")

    except ccxt.InsufficientFunds as e:
        logger.error(f"Fehler: Nicht genügend Guthaben. {e}")
    except Exception as e:
        logger.error(f"Unerwarteter Fehler im Handelszyklus: {e}", exc_info=True)
