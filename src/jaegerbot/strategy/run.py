# src/jaegerbot/strategy/run.py
"""Ein Live-Lauf fuer eine Strategie: python src/jaegerbot/strategy/run.py --symbol BTC/USDT:USDT --timeframe 4h"""
import os
import sys
import json
import argparse
import logging
from logging.handlers import RotatingFileHandler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from jaegerbot.utils.exchange import Exchange
from jaegerbot.utils.telegram import send_message
from jaegerbot.utils.trade_manager import full_trade_cycle, strategy_key


def setup_logging(symbol, timeframe):
    key = strategy_key(symbol, timeframe)
    os.makedirs(os.path.join(PROJECT_ROOT, 'logs'), exist_ok=True)
    logger = logging.getLogger(f'jaegerbot_{key}')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = RotatingFileHandler(os.path.join(PROJECT_ROOT, 'logs', f'jaegerbot_{key}.log'),
                                 maxBytes=5 * 1024 * 1024, backupCount=3, encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(f'%(asctime)s [%(levelname)s] {symbol}|{timeframe}: %(message)s', datefmt='%H:%M:%S'))
        logger.addHandler(fh)
        logger.addHandler(ch)
        logger.propagate = False
    return logger


def load_config(symbol, timeframe):
    path = os.path.join(PROJECT_ROOT, 'src', 'jaegerbot', 'strategy', 'configs',
                        f"config_{strategy_key(symbol, timeframe)}.json")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser(description="JaegerBot v5 Live-Lauf")
    p.add_argument('--symbol', required=True)
    p.add_argument('--timeframe', required=True)
    a = p.parse_args()
    logger = setup_logging(a.symbol, a.timeframe)
    try:
        config = load_config(a.symbol, a.timeframe)
        with open(os.path.join(PROJECT_ROOT, 'settings.json'), 'r', encoding='utf-8') as f:
            live_settings = json.load(f)['live_trading_settings']
        with open(os.path.join(PROJECT_ROOT, 'secret.json'), 'r', encoding='utf-8') as f:
            secrets = json.load(f)
        accounts = secrets.get('jaegerbot') or []
        telegram = secrets.get('telegram', {})
    except Exception as e:
        logger.critical(f"Initialisierung fehlgeschlagen: {e}", exc_info=True)
        sys.exit(1)
    if not accounts:
        logger.critical("Kein 'jaegerbot'-Account in secret.json.")
        sys.exit(1)

    for account in accounts:
        try:
            exchange = Exchange(account)
            if not exchange.markets:
                logger.critical("Maerkte nicht geladen -- Lauf abgebrochen.")
                continue
            full_trade_cycle(exchange, config, live_settings, telegram, logger)
        except Exception as e:
            logger.critical(f"Fehler im Handelszyklus: {e}", exc_info=True)
            try:
                send_message(telegram.get('bot_token'), telegram.get('chat_id'),
                             f"🚨 jaegerbot {a.symbol} ({a.timeframe}): Fehler im Handelszyklus\n{e}")
            except Exception:
                pass
    logger.info(f">>> jaegerbot-Lauf {a.symbol} ({a.timeframe}) fertig <<<")


if __name__ == '__main__':
    main()
