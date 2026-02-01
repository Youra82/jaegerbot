# tests/test_workflow.py
# =============================================================================
# JaegerBot: Live-Workflow-Test auf Bitget (Vereinfacht wie StBot/KBot)
# =============================================================================
import pytest
import os
import sys
import json
import logging
import time

# Füge das Projektverzeichnis zum Python-Pfad hinzu
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from jaegerbot.utils.exchange import Exchange
from jaegerbot.utils.trade_manager import housekeeper_routine

# Pfade für Lock-Dateien
LOCK_FILE_PATH = os.path.join(PROJECT_ROOT, 'artifacts', 'db', 'trade_lock.json')
CIRCUIT_BREAKER_PATH = os.path.join(PROJECT_ROOT, 'artifacts', 'db', 'circuit_breaker.json')


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

    yield exchange, telegram_config, symbol, test_logger

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
    1. Trade-Eröffnung mit direktem Market Order
    2. Position-Verifizierung  
    3. Sauberes Schließen
    """
    exchange, telegram_config, symbol, logger = test_setup

    # Check Balance vor dem Test
    bal = exchange.fetch_balance_usdt()
    print(f"\n--- Verfügbares Guthaben für Test: {bal:.4f} USDT ---")

    if bal < 5:
        pytest.skip(f"Nicht genug Guthaben für Test: {bal:.2f} USDT")

    # === DIREKTER TRADE TEST (wie bei KBot/StBot) ===
    print("\n[Schritt 1/3] Eröffne Test-Position direkt...")
    
    # Setze Margin Mode und Leverage
    exchange.set_margin_mode(symbol, 'isolated')
    exchange.set_leverage(symbol, 20)
    
    # Berechne Position Size (15% vom Balance, 20x Leverage)
    ticker = exchange.fetch_ticker(symbol)
    current_price = ticker['last']
    position_value = bal * 0.15 * 20  # 15% Risk * 20x Leverage
    contracts = position_value / current_price
    
    # Eröffne LONG Position
    order = exchange.create_market_order(symbol, 'buy', contracts)
    
    if not order or 'id' not in order:
        pytest.fail("Market Order fehlgeschlagen")
    
    print(f"-> Order platziert: {order.get('id')}")
    time.sleep(3)

    print("\n[Schritt 2/3] Überprüfe Position...")
    position = exchange.fetch_open_positions(symbol)

    # Assert Position
    if not position:
        pytest.fail(f"FEHLER: Position nicht eröffnet. Guthaben: {bal:.2f} USDT")

    assert len(position) == 1
    pos_info = position[0]
    print(f"-> Position erfolgreich eröffnet: {pos_info['side'].upper()} {pos_info['contracts']} PEPE.")
    
    # Prüfe Margin Mode
    margin_mode = pos_info.get('marginMode', 'unknown')
    print(f"-> Margin Mode: {margin_mode}")

    # --- SAUBERES SCHLIESSEN ---
    print("\n[Schritt 3/3] Schließe die Position und räume auf...")

    # 1. Orders löschen VOR dem Schließen
    print("-> Lösche Trigger-Orders VOR dem Schließen...")
    exchange.cleanup_all_open_orders(symbol)
    time.sleep(2)

    # 2. Position schließen
    amount_to_close = abs(float(pos_info.get('contracts', 0)))
    side_to_close = 'sell' if pos_info.get('side', '').lower() == 'long' else 'buy'

    if amount_to_close > 0:
        print(f"-> Schließe Position ({amount_to_close} PEPE)...")
        close_order = exchange.create_market_order(symbol, side_to_close, amount_to_close, params={'reduceOnly': True})
        assert close_order, "FEHLER: Konnte Position nicht schließen!"
        print(f"-> Position erfolgreich geschlossen.")
        time.sleep(3)

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
