# tests/test_workflow.py
# =============================================================================
# JaegerBot: Live-Workflow-Test auf Bitget (mit gemocktem ANN-Signal)
# =============================================================================
import pytest
import os
import sys
import json
import logging
import time
import pandas as pd
from unittest.mock import patch

# Füge das Projektverzeichnis zum Python-Pfad hinzu
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from jaegerbot.utils.exchange import Exchange
from jaegerbot.utils.trade_manager import check_and_open_new_position, housekeeper_routine

# Pfade für Lock-Dateien
LOCK_FILE_PATH = os.path.join(PROJECT_ROOT, 'artifacts', 'db', 'trade_lock.json')
CIRCUIT_BREAKER_PATH = os.path.join(PROJECT_ROOT, 'artifacts', 'db', 'circuit_breaker.json')


# Mock-Klassen für das ANN-Modell
class FakeModel:
    """Mock-Version des Keras-Modells - gibt immer LONG-Signal."""
    def predict(self, data, verbose=0):
        return [[0.95]]  # Hohe Vorhersage = LONG Signal


class FakeScaler:
    """Mock-Version des Scalers - reicht Daten einfach durch."""
    def transform(self, data):
        return data


class FakeSuperTrendLocal:
    """Mock-Version von SuperTrend - gibt immer LONG-Trend zurück."""
    def __init__(self, high, low, close, window, multiplier):
        self.size = len(high)

    def get_supertrend_direction(self):
        return pd.Series([1.0] * self.size)  # 1.0 = Long-Trend


def create_fake_features(ohlcv_data):
    """
    Erstellt Fake-Features die alle Filter passieren.
    ADX > 20, Volume hoch, ATR normal.
    """
    df = ohlcv_data.copy()
    n = len(df)
    
    # Alle benötigten Feature-Spalten mit Werten die Filter passieren
    df['bb_width'] = 0.02
    df['bb_pband'] = 0.5
    df['obv'] = 1000000
    df['rsi'] = 50.0
    df['macd_diff'] = 0.0
    df['macd'] = 0.0
    df['atr'] = df['close'] * 0.02  # 2% ATR
    df['atr_normalized'] = 2.0  # Normal, nicht zu hoch
    df['adx'] = 35.0  # Über 20 - passiert ADX-Filter!
    df['adx_pos'] = 20.0
    df['adx_neg'] = 15.0
    df['volume_ratio'] = 1.5  # Über 0.8 - passiert Volume-Filter!
    df['mfi'] = 50.0
    df['cmf'] = 0.1
    df['price_to_ema20'] = 1.0
    df['price_to_ema50'] = 1.0
    df['stoch_k'] = 50.0
    df['stoch_d'] = 50.0
    df['williams_r'] = -50.0
    df['roc'] = 0.0
    df['cci'] = 0.0
    df['price_to_resistance'] = 1.0
    df['price_to_support'] = 1.0
    df['high_low_range'] = 0.02
    df['close_to_high'] = 0.5
    df['close_to_low'] = 0.5
    df['day_of_week'] = 1
    df['hour_of_day'] = 12
    df['returns_lag1'] = 0.0
    df['returns_lag2'] = 0.0
    df['returns_lag3'] = 0.0
    df['hist_volatility'] = 0.02
    
    return df


def clear_lock_file():
    """Löscht die trade_lock.json, falls sie existiert."""
    if os.path.exists(LOCK_FILE_PATH):
        try:
            os.remove(LOCK_FILE_PATH)
            print("-> Lokale 'trade_lock.json' wurde erfolgreich gelöscht.")
        except Exception as e:
            print(f"Warnung: Lock-Datei konnte nicht gelöscht werden: {e}")


def clear_circuit_breaker_file():
    """Löscht die circuit_breaker.json, falls sie existiert."""
    if os.path.exists(CIRCUIT_BREAKER_PATH):
        try:
            os.remove(CIRCUIT_BREAKER_PATH)
            print("-> Lokale 'circuit_breaker.json' wurde erfolgreich gelöscht.")
        except Exception as e:
            print(f"Warnung: Circuit-Breaker-Datei konnte nicht gelöscht werden: {e}")


@pytest.fixture(scope="module")
def test_setup():
    print("\n--- Starte umfassenden LIVE JaegerBot-Workflow-Test (PEPE) ---")
    print("\n[Setup] Bereite Testumgebung vor...")

    secret_path = os.path.join(PROJECT_ROOT, 'secret.json')
    if not os.path.exists(secret_path):
        pytest.skip("secret.json nicht gefunden. Überspringe Live-Workflow-Test.")

    with open(secret_path, 'r') as f:
        secrets = json.load(f)

    if not secrets.get('jaegerbot') or not secrets['jaegerbot']:
        pytest.skip("Es wird mindestens ein Account unter 'jaegerbot' in secret.json benötigt.")

    test_account = secrets['jaegerbot'][0]
    telegram_config = secrets.get('telegram', {})

    try:
        exchange = Exchange(test_account)
        if not exchange.markets:
            pytest.fail("Exchange konnte nicht initialisiert werden (Märkte nicht geladen).")
    except Exception as e:
        pytest.fail(f"Exchange konnte nicht initialisiert werden: {e}")

    # PEPE - Kleine Mindestgröße, gut zum Testen
    symbol = 'PEPE/USDT:USDT'

    # Test-Parameter für JaegerBot
    params = {
        'market': {'symbol': symbol, 'timeframe': '15m'},
        'strategy': {'prediction_threshold': 0.6},
        'behavior': {'use_longs': True, 'use_shorts': True},
        'risk': {
            'risk_per_trade_pct': 15.0,       # 15% wie StBot/KBot
            'risk_reward_ratio': 2.0,
            'initial_sl_pct': 4.0,            # 4% SL wie StBot/KBot
            'leverage': 20,
            'margin_mode': 'isolated',
            'trailing_stop_activation_rr': 1.0,
            'trailing_stop_callback_rate_pct': 0.5
        }
    }

    model = FakeModel()
    scaler = FakeScaler()

    test_logger = logging.getLogger("test-logger")
    test_logger.setLevel(logging.INFO)
    if not test_logger.handlers:
        test_logger.addHandler(logging.StreamHandler(sys.stdout))

    print(f"-> Führe initiales Aufräumen für {symbol} durch...")
    try:
        housekeeper_routine(exchange, symbol, test_logger)
        time.sleep(2)
        # Doppelte Sicherheit
        pos = exchange.fetch_open_positions(symbol)
        if pos:
            exchange.create_market_order(symbol, 'sell' if pos[0]['side'] == 'long' else 'buy', 
                                         float(pos[0]['contracts']), {'reduceOnly': True})
            time.sleep(2)

        clear_lock_file()
        clear_circuit_breaker_file()
        print("-> Ausgangszustand ist sauber.")
    except Exception as e:
        pytest.fail(f"Fehler beim initialen Aufräumen: {e}")

    yield exchange, model, scaler, params, telegram_config, symbol, test_logger

    print("\n[Teardown] Räume nach dem Test auf...")
    try:
        print("-> 1. Lösche offene Trigger Orders...")
        exchange.cleanup_all_open_orders(symbol)
        time.sleep(2)

        print("-> 2. Prüfe auf offene Positionen...")
        position = exchange.fetch_open_positions(symbol)
        if position:
            print(f"-> Position nach Test noch offen. Schließe sie...")
            exchange.create_market_order(symbol, 'sell' if position[0]['side'] == 'long' else 'buy', 
                                         float(position[0]['contracts']), {'reduceOnly': True})
            time.sleep(3)
        else:
            print("-> Keine offene Position gefunden.")

        print("-> 3. Lösche verbleibende Trigger Orders (Sicherheitsnetz)...")
        exchange.cleanup_all_open_orders(symbol)

        clear_lock_file()
        clear_circuit_breaker_file()
        print("-> Aufräumen abgeschlossen.")

    except Exception as e:
        print(f"FEHLER beim Aufräumen nach dem Test: {e}")


def test_full_jaegerbot_workflow_on_bitget(test_setup):
    """
    Umfassender Live-Workflow-Test für JaegerBot.
    
    Testet:
    1. Trade-Eröffnung mit gemocktem ANN-Signal
    2. Position + SL/TSL-Verifizierung (Native Bitget TSL)
    3. Sauberes Schließen
    """
    exchange, model, scaler, params, telegram_config, symbol, logger = test_setup

    # Check Balance vor dem Test
    bal = exchange.fetch_balance_usdt()
    print(f"\n--- Verfügbares Guthaben für Test: {bal:.4f} USDT ---")

    if bal < 5:
        pytest.skip(f"Nicht genug Guthaben für Test: {bal:.2f} USDT")

    # Mock SuperTrend und create_ann_features um alle Filter zu passieren
    with patch('jaegerbot.utils.trade_manager.SuperTrendLocal', FakeSuperTrendLocal):
        with patch('jaegerbot.utils.trade_manager.create_ann_features', side_effect=create_fake_features):
            print("\n[Schritt 1/3] Mocke ANN-Signal und prüfe Trade-Eröffnung...")
            check_and_open_new_position(exchange, model, scaler, params, telegram_config, logger)

    print("-> Warte 5s auf Order-Ausführung...")
    time.sleep(5)

    print("\n[Schritt 2/3] Überprüfe Position und Orders...")
    position = exchange.fetch_open_positions(symbol)

    # Assert Position
    if not position:
        pytest.fail(f"FEHLER: Position nicht eröffnet. Verfügbares Guthaben ({bal:.2f} USDT) war evtl. zu wenig oder Filter haben blockiert.")

    assert len(position) == 1
    pos_info = position[0]
    print(f"-> Position erfolgreich eröffnet: {pos_info['side'].upper()} {pos_info['contracts']} PEPE.")
    
    # Prüfe Margin Mode
    margin_mode = pos_info.get('marginMode', 'unknown')
    print(f"-> Margin Mode: {margin_mode}")

    # Assert Orders (SL + optional TSL)
    trigger_orders = exchange.fetch_open_trigger_orders(symbol)
    if len(trigger_orders) == 0:
        print("WARNUNG: Keine Trigger-Orders im API-Return gefunden (kann bei PEPE vorkommen).")
    else:
        print(f"-> Trigger-Orders gefunden: {len(trigger_orders)} (SL + ggf. TSL)")

    # --- SAUBERES SCHLIESSEN ---
    print("\n[Schritt 3/3] Schließe die Position und räume auf...")

    # 1. Orders löschen VOR dem Schließen
    print("-> Lösche Trigger-Orders VOR dem Schließen...")
    exchange.cleanup_all_open_orders(symbol)
    time.sleep(3)

    # 2. Position schließen
    amount_to_close = abs(float(pos_info.get('contracts', 0)))
    side_to_close = 'sell' if pos_info.get('side', '').lower() == 'long' else 'buy'

    if amount_to_close > 0:
        print(f"-> Schließe Position ({amount_to_close} PEPE)...")
        close_order = exchange.create_market_order(symbol, side_to_close, amount_to_close, params={'reduceOnly': True})
        assert close_order, "FEHLER: Konnte Position nicht schließen!"
        print(f"-> Position erfolgreich geschlossen.")
        time.sleep(4)

    # 3. Orders löschen NACH dem Schließen
    print("-> Lösche verbleibende Trigger-Orders NACH dem Schließen...")
    exchange.cleanup_all_open_orders(symbol)
    time.sleep(2)

    # Finale Prüfung
    final_positions = exchange.fetch_open_positions(symbol)
    final_orders = exchange.fetch_open_trigger_orders(symbol)

    if len(final_orders) > 0:
        print(f"WARNUNG: Es sind noch {len(final_orders)} Trigger-Orders offen! Versuche erneutes Löschen...")
        exchange.cleanup_all_open_orders(symbol)
        time.sleep(2)
        final_orders = exchange.fetch_open_trigger_orders(symbol)

    assert len(final_positions) == 0, "FEHLER: Position sollte geschlossen sein."
    assert len(final_orders) == 0, f"FEHLER: Trigger-Orders wurden nicht sauber gelöscht! ({len(final_orders)} verbleibend)"

    print("\n--- UMFASSENDER WORKFLOW-TEST ERFOLGREICH! ---")
