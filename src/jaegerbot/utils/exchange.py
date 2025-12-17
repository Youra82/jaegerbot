# src/jaegerbot/utils/exchange.py
import ccxt
import pandas as pd
from datetime import datetime, timezone
import logging
import time # Hinzugefügt: Import der time Bibliothek für sleep

# NEU: Logger für diese Datei holen
logger = logging.getLogger(__name__)

class Exchange:
    def __init__(self, account_config):
        self.account = account_config
        self.exchange = getattr(ccxt, 'bitget')({
            'apiKey': self.account.get('apiKey'),
            'secret': self.account.get('secret'),
            'password': self.account.get('password'),
            'options': {
                'defaultType': 'swap',
            },
            'enableRateLimit': True, # Neu hinzugefügt
        })
        try:
            self.markets = self.exchange.load_markets()
            logger.info("Bitget Märkte erfolgreich geladen.")
        except ccxt.AuthenticationError as e:
            logger.critical(f"FATAL: Bitget Authentifizierungsfehler: {e}. Bitte API-Schlüssel prüfen.")
            self.markets = None
        except Exception as e:
            logger.warning(f"WARNUNG: Fehler beim Laden der Märkte: {e}")
            self.markets = None

    def fetch_recent_ohlcv(self, symbol, timeframe, limit=100):
        # ... (Unveränderter Code von Zeile 37 bis 50)
        if not self.markets: return pd.DataFrame()
        try:
            effective_limit = min(limit, 1000)
            data = self.exchange.fetch_ohlcv(symbol, timeframe, limit=effective_limit)
            if not data: return pd.DataFrame()
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            return df
        except Exception as e:
            logger.error(f"Fehler bei fetch_recent_ohlcv für {symbol}: {e}")
            return pd.DataFrame()

    def fetch_historical_ohlcv(self, symbol, timeframe, start_date_str, end_date_str):
        # ... (Unveränderter Code von Zeile 52 bis 109)
        start_ts = int(self.exchange.parse8601(start_date_str + 'T00:00:00Z'))
        end_ts = int(self.exchange.parse8601(end_date_str + 'T00:00:00Z'))
        all_ohlcv = []

        while start_ts < end_ts:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=start_ts, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            start_ts = ohlcv[-1][0] + 1

        if not all_ohlcv:
            return pd.DataFrame()

        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        return df[~df.index.duplicated(keep='first')].sort_index()

    def fetch_ticker(self, symbol):
        return self.exchange.fetch_ticker(symbol)

    def set_margin_mode(self, symbol, mode='isolated'):
        try:
            self.exchange.set_margin_mode(mode, symbol)
            return True
        except Exception as e:
            if 'Margin mode is the same' not in str(e): 
                logger.warning(f"Warnung: Margin-Modus konnte nicht gesetzt werden: {e}")
            else:
                return True
            return False

    def set_leverage(self, symbol, level=10):
        try:
            self.exchange.set_leverage(level, symbol)
            return True
        except Exception as e:
            if 'Leverage not changed' not in str(e): 
                logger.warning(f"Warnung: Set leverage failed: {e}")
            else:
                return True
            return False

    def create_market_order(self, symbol, side, amount, params={}):
        rounded_amount = float(self.exchange.amount_to_precision(symbol, amount))
        return self.exchange.create_order(symbol, 'market', side, rounded_amount, params=params)

    def place_trigger_market_order(self, symbol, side, amount, trigger_price, params={}):
        rounded_price = float(self.exchange.price_to_precision(symbol, trigger_price))
        rounded_amount = float(self.exchange.amount_to_precision(symbol, amount))
        order_params = {
            'triggerPrice': rounded_price,
            'reduceOnly': params.get('reduceOnly', False)
        }
        order_params.update(params)
        return self.exchange.create_order(symbol, 'market', side, rounded_amount, params=order_params)

    def fetch_open_positions(self, symbol):
        positions = self.exchange.fetch_positions([symbol])
        open_positions = [p for p in positions if p.get('contracts', 0.0) > 0.0]
        return open_positions

    def fetch_open_trigger_orders(self, symbol):
        return self.exchange.fetch_open_orders(symbol, params={'stop': True})

    def fetch_balance_usdt(self):
        # ... (Unveränderter Code von Zeile 219 bis 242)
        try:
            balance = self.exchange.fetch_balance()
            if 'USDT' in balance:
                if 'free' in balance['USDT'] and balance['USDT']['free'] is not None:
                    return float(balance['USDT']['free'])
                elif 'available' in balance['USDT'] and balance['USDT']['available'] is not None:
                    return float(balance['USDT']['available'])
                elif 'total' in balance['USDT'] and balance['USDT']['total'] is not None:
                    return float(balance['USDT']['total'])
            # Zusätzliche Überprüfung für Unified Account (falls vorhanden)
            elif 'info' in balance and 'data' in balance['info'] and isinstance(balance['info']['data'], list):
                for asset_info in balance['info']['data']:
                    if asset_info.get('marginCoin') == 'USDT':
                        if 'available' in asset_info and asset_info['available'] is not None:
                            return float(asset_info['available'])
                        elif 'equity' in asset_info and asset_info['equity'] is not None:
                            return float(asset_info['equity'])
            return 0
        except Exception as e:
            logger.error(f"Fehler beim Abrufen des Kontostandes: {e}")
            return 0

    def cleanup_all_open_orders(self, symbol):
        # ... (Unveränderter Code von Zeile 248 bis 284)
        cancelled_count = 0
        try:
            # Versuche, alle Trigger-Orders zu stornieren (inkl. TSL, SL, TP)
            logger.info(f"Sende Befehl 'cancelAllOrders' (Trigger/Stop) für {symbol}...")
            self.exchange.cancel_all_orders(symbol, params={'productType': 'USDT-FUTURES', 'stop': True})
            cancelled_count += 1
            time.sleep(0.5)
        except ccxt.ExchangeError as e:
            if 'Order not found' in str(e) or 'no order to cancel' in str(e).lower() or '22001' in str(e):
                logger.info("Keine Trigger-Orders zum Stornieren gefunden.")
            else:
                logger.error(f"Fehler beim Stornieren von Trigger-Orders: {e}")
        except Exception as e:
            logger.error(f"Unerwarteter Fehler beim Stornieren von Trigger-Orders: {e}")
        
        # Normale Orders stornieren
        try:
            logger.info(f"Sende Befehl 'cancelAllOrders' (Normal) für {symbol}...")
            self.exchange.cancel_all_orders(symbol, params={'productType': 'USDT-FUTURES', 'stop': False})
            cancelled_count += 1
            time.sleep(0.5)
        except ccxt.ExchangeError as e:
            if 'Order not found' in str(e) or 'no order to cancel' in str(e).lower() or '22001' in str(e):
                logger.info("Keine normalen Orders zum Stornieren gefunden.")
            else:
                logger.error(f"Fehler beim Stornieren normaler Orders: {e}")
        except Exception as e:
            logger.error(f"Unerwarteter Fehler beim Stornieren normaler Orders: {e}")

        return cancelled_count

    # *** TSL-Platzierung nach TitanBot-Logik (unverändert) ***
    def place_trailing_stop_order(self, symbol, side, amount, activation_price, callback_rate_decimal, params={}):
        """
        Platziert eine Trailing Stop Market Order (Stop-Loss) über ccxt für Bitget.
        Nutzt die Bitget-spezifischen 'trailingTriggerPrice' und 'trailingPercent'-Parameter.
        """
        if not self.markets: return None
        try:
            # Runden der Werte
            rounded_activation = float(self.exchange.price_to_precision(symbol, activation_price))
            rounded_amount = float(self.exchange.amount_to_precision(symbol, amount))
            if rounded_amount <= 0:
                logger.error(f"FEHLER: Berechneter TSL-Betrag ist Null ({rounded_amount}).")
                return None

            # In Prozent umwandeln (z.B. 0.5% für TSL)
            callback_rate_float = callback_rate_decimal * 100 

            # Verwende die Bitget-spezifischen Parameter in den `params`
            order_params = {
                **params,
                'trailingTriggerPrice': rounded_activation,
                'trailingPercent': callback_rate_float,
                'productType': 'USDT-FUTURES'
            }

            logger.info(f"Sende TSL Order (MARKET): Side={side}, Amount={rounded_amount}, Activation={rounded_activation}, Callback={callback_rate_float}%")

            # Der Order-Typ ist 'market', da Trailing Stop in Bitget über spezielle Parameter definiert wird
            return self.exchange.create_order(symbol, 'market', side, rounded_amount, params=order_params)

        except Exception as e:
            logger.error(f"FEHLER beim Platzieren des Trailing Stop ({symbol}, {side}): {e}", exc_info=True)
            # Fehler weitergeben, damit der trade_manager ihn fangen und den Fallback (fixen SL) ausführen kann
            raise e
