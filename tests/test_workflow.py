# tests/test_workflow.py (FINAL KORRIGIERTE VERSION 6 - Anpassung auf BTC/USDT)
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
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

# Korrekte Imports
from jaegerbot.utils.exchange import Exchange
from jaegerbot.utils.trade_manager import check_and_open_new_position, housekeeper_routine
from jaegerbot.utils.supertrend_indicator import SuperTrendLocal 
from jaegerbot.utils.ann_model import create_ann_features 

# Definition der Pfade und Mocks

LOCK_FILE_PATH = os.path.join(PROJECT_ROOT, 'artifacts', 'db', 'trade_lock.json')

# Erstelle eine "Fake"-KI, die wir für den Test kontrollieren können
class FakeModel:
    """Eine Mock-Version des Keras-Modells, um das Vorhersage-Verhalten zu steuern."""
    def __init__(self):
        self.return_value = [[0.5]]
    def predict(self, data, verbose=0):
        return self.return_value

class FakeScaler:
    """Eine Mock-Version des Scalers, die einfach die Daten durchreicht."""
    def transform(self, data):
        return data

def clear_lock_file():
    """Löscht die trade_lock.json, falls sie existiert."""
    if os.path.exists(LOCK_FILE_PATH):
        try:
            os.remove(LOCK_FILE_PATH)
            print("-> Lokale 'trade_lock.json' wurde erfolgreich gelöscht.")
        except Exception as e:
            print(f"Warnung: Lock-Datei konnte nicht gelöscht werden: {e}")


@pytest.fixture
def test_setup():
    """
    Bereitet die Testumgebung vor und räumt danach auf.
    """
    print("\n--- Starte umfassenden LIVE JaegerBot-Workflow-Test ---")
    print("\n[Setup] Bereite Testumgebung vor...")

    secret_path = os.path.join(PROJECT_ROOT, 'secret.json')
    if not os.path.exists(secret_path):
        pytest.skip("secret.json nicht gefunden. Überspringe Live-Workflow-Test.")

    with open(secret_path, 'r') as f:
        secrets = json.load(f)

    if not secrets.get('jaegerbot'):
        pytest.skip("Es wird mindestens ein Account in secret.json für den Workflow-Test benötigt.")

    test_account = secrets['jaegerbot'][0]
    telegram_config = secrets.get('telegram', {})

    exchange = Exchange(test_account)
    # WICHTIG: Ändere das Symbol auf BTC/USDT für geringere Mindestanforderung
    symbol = 'BTC/USDT:USDT' 

    # ANGEPASSTE RISIKO-PARAMETER
    params = {
        'market': {'symbol': symbol, 'timeframe': '15m'},
        'strategy': {'prediction_threshold': 0.6},
        'behavior': {'use_longs': True, 'use_shorts': True, 'use_macd_trend_filter': False},
        'risk': {
            'risk_per_trade_pct': 2.0,       
            'risk_reward_ratio': 2.0,
            'initial_sl_pct': 0.5,           # Etwas entspannterer SL für BTC
            'leverage': 7,
            'margin_mode': 'isolated',
            'trailing_stop_activation_rr': 1.5,
            'trailing_stop_callback_rate_pct': 0.5
        }
    }

    model = FakeModel()
    scaler = FakeScaler()

    # Initiales Aufräumen (Housekeeper schließt Positionen und löscht Orders)
    print("-> Führe initiales Aufräumen durch (Remote Bitget)...")
    # Verwende das neue Symbol für den Housekeeper-Aufruf
    housekeeper_routine(exchange, symbol, logging.getLogger("test-logger")) 

    print("-> Führe initiales Aufräumen durch (Lokal)...")
    clear_lock_file()

    print("-> Ausgangszustand ist sauber.")

    yield exchange, model, scaler, params, telegram_config, symbol

    print("\n[Teardown] Räume nach dem Test auf...")
    try:
        # Führt den robusten Housekeeper im Teardown aus, um die offene Position zu schließen.
        housekeeper_routine(exchange, symbol, logging.getLogger("test-logger"))
        final_pos_check = exchange.fetch_open_positions(symbol)
        if final_pos_check:
            print("WARNUNG: Position nach finalem Teardown immer noch offen.")
    except Exception as e:
        print(f"Fehler beim Aufräumen (Remote): {e}")

    print("-> Räume lokale Lock-Datei auf...")
    clear_lock_file()


# NEUE KLASSE zum Mocken der SuperTrendLocal-Initialisierung
class FakeSuperTrendLocal:
    def __init__(self, high, low, close, window, multiplier):
        self.size = len(high) 

    def get_supertrend_direction(self):
        # Gibt immer 1.0 (Long-Trend) zurück, um den Filter im trade_manager zu passieren
        # Die Größe muss der Länge der gefakten OHLCV Daten (100) entsprechen
        return pd.Series([1.0] * self.size)


def test_full_jaegerbot_workflow_on_bitget(test_setup):
    """
    Testet den gesamten Handelsablauf über den trade_manager auf dem konfigurierten Live-Konto.
    """
    exchange, model, scaler, params, telegram_config, symbol = test_setup
    logger = logging.getLogger("test-logger")
    
    # --- DEKLARATION DER MOCK DATEN ---
    DATE_LEN = 100 
    SAFE_LEN = 50 
    BTC_PRICE = 30000.0 # Realistischer BTC-Preis
    
    # 1. Mock Daten für fetch_recent_ohlcv (OHLCV)
    start_dt = '2025-01-01 00:00:00'
    date_range = pd.date_range(start=start_dt, periods=DATE_LEN, freq='15min')
    # Anpassung des Preises auf BTC_PRICE
    ohlcv_mock_data = {
        'open': [BTC_PRICE] * DATE_LEN, 
        'high': [BTC_PRICE + 300] * DATE_LEN, 
        'low': [BTC_PRICE - 300] * DATE_LEN,
        'close': [BTC_PRICE] * DATE_LEN, 
        'volume': [1000] * DATE_LEN
    }
    mock_ohlcv_df = pd.DataFrame(ohlcv_mock_data, index=date_range)

    # 2. Mock Daten für create_ann_features (Features + OHLCV + ATR)
    model.return_value = [[0.9]] 
    feature_cols = [
        'bb_width', 'obv', 'rsi', 'macd_diff', 'day_of_week',
        'returns_lag1', 'returns_lag2', 'atr_normalized', 'adx'
    ]
    fake_data = {col: [0.0] * SAFE_LEN for col in feature_cols}
    fake_data['adx'] = [30.0] * SAFE_LEN
    
    # Anpassung des Preises auf BTC_PRICE
    fake_data['high'] = [BTC_PRICE + 300] * SAFE_LEN
    fake_data['low'] = [BTC_PRICE - 300] * SAFE_LEN
    fake_data['close'] = [BTC_PRICE] * SAFE_LEN
    # Setze einen realistischeren ATR für BTC (z.B. 1% des Preises = 300)
    fake_data['atr'] = [300.0] * SAFE_LEN 

    index_range = pd.date_range(start='2025-01-01', periods=SAFE_LEN, freq='15min')
    fake_features_df = pd.DataFrame(fake_data, index=index_range)
    
    # --- ENDE DEKLARATION DER MOCK DATEN ---

    # --- WICHTIG: Mocken des REALEN Kontostandes ---
    # Setze den Mock auf den tatsächlichen Wert von 25 USDT
    USER_BALANCE = 25.0 
    with patch('jaegerbot.utils.exchange.Exchange.fetch_balance_usdt', return_value=USER_BALANCE):
        with patch('jaegerbot.utils.trade_manager.create_ann_features', return_value=fake_features_df):
            with patch('jaegerbot.utils.exchange.Exchange.fetch_recent_ohlcv', return_value=mock_ohlcv_df):
                with patch('jaegerbot.utils.trade_manager.SuperTrendLocal', new=FakeSuperTrendLocal):
                    
                    print(f"\n[Schritt 1/3] Prüfe Trade-Eröffnung ({symbol}) mit {USER_BALANCE} USDT Kapital...")
                    check_and_open_new_position(exchange, model, scaler, params, telegram_config, logger)
                    time.sleep(5)

    print("\n[Schritt 2/3] Überprüfe, ob die Position und Orders korrekt erstellt wurden...")
    position = exchange.fetch_open_positions(symbol)
    trigger_orders = exchange.fetch_open_trigger_orders(symbol)

    # Hier sollte die Position ERFOLGREICH eröffnet worden sein
    assert position, "FEHLER: Position wurde nicht eröffnet!"
    assert position[0]['marginMode'] == 'isolated', f"FEHLER: Position wurde im falschen Margin-Modus eröffnet: {position[0]['marginMode']}"
    print(f"-> ✔ Position korrekt eröffnet (Isolated, {position[0]['leverage']}x).")

    # Prüfe, dass Orders existieren (mindestens der fixe SL)
    assert len(trigger_orders) >= 1, f"FEHLER: Es muss mindestens 1 Trigger-Order (SL) aktiv sein. Gefunden: {len(trigger_orders)}"

    print(f"-> ✔ {len(trigger_orders)} SL/TSL-Order(s) erfolgreich platziert.")

    print("\n[Schritt 3/3] Test erfolgreich, Aufräumen wird im Teardown durchgeführt.")
    print("\n--- ✅ UMFASSENDER WORKFLOW-TEST ERFOLGREICH! ---")
