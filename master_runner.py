# master_runner.py
"""JaegerBot v5 Master Runner (Cron alle 15 Min, mit flock).

Alle aktiven Strategien in EINEM Prozess, nacheinander in fester Reihenfolge (wie der
Portfolio-Simulator): Durchgang 1 nur Ausstiege, Durchgang 2 Einstiege. Parallele Prozesse
koennten sonst gleichzeitig die max. Positionszahl oder dasselbe Symbol belegen.
"""
import os
import sys
import json
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from jaegerbot.utils.exchange import Exchange
from jaegerbot.utils.telegram import send_message
from jaegerbot.utils.trade_manager import full_trade_cycle, strategy_key
from jaegerbot.strategy.run import setup_logging, load_config


def start_background(script, log_name):
    path = os.path.join(PROJECT_ROOT, script)
    if not os.path.exists(path):
        return
    os.makedirs(os.path.join(PROJECT_ROOT, 'logs'), exist_ok=True)
    subprocess.Popen([sys.executable, path], stdout=open(os.path.join(PROJECT_ROOT, 'logs', log_name), 'a'),
                     stderr=subprocess.STDOUT)


def main():
    print("=======================================================")
    print("JaegerBot Master Runner v5 (ANN-Ensemble, Live = Backtest)")
    print("=======================================================")
    start_background('auto_optimizer_scheduler.py', 'auto_optimizer_trigger.log')
    start_background('daily_self_check.py', 'daily_self_check.log')

    with open(os.path.join(PROJECT_ROOT, 'settings.json'), 'r', encoding='utf-8') as f:
        live_settings = json.load(f)['live_trading_settings']
    with open(os.path.join(PROJECT_ROOT, 'secret.json'), 'r', encoding='utf-8') as f:
        secrets = json.load(f)
    accounts = secrets.get('jaegerbot') or []
    telegram = secrets.get('telegram', {})
    if not accounts:
        print("Fehler: Kein 'jaegerbot'-Account in secret.json.")
        return

    strategies = sorted((s for s in live_settings.get('active_strategies', []) if s.get('active', True)),
                        key=lambda s: strategy_key(s['symbol'], s['timeframe']))
    if not strategies:
        print("Keine aktiven Strategien in settings.json.")
        return

    exchange = Exchange(accounts[0])
    if not exchange.markets:
        print("Fehler: Maerkte nicht geladen.")
        return

    jobs = []
    for s in strategies:
        try:
            jobs.append((load_config(s['symbol'], s['timeframe']), setup_logging(s['symbol'], s['timeframe'])))
        except FileNotFoundError:
            print(f"Warnung: Config fehlt fuer {s['symbol']} {s['timeframe']} -- uebersprungen.")

    candles = {}
    for phase in ('ausstiege', 'einstiege'):
        for config, logger in jobs:
            key = strategy_key(config['market']['symbol'], config['market']['timeframe'])
            try:
                df = full_trade_cycle(exchange, config, live_settings, telegram, logger,
                                      df=candles.get(key), manage_only=(phase == 'ausstiege'))
                if phase == 'ausstiege' and df is not None:
                    candles[key] = df
            except Exception as e:
                logger.critical(f"Fehler ({phase}): {e}", exc_info=True)
                try:
                    send_message(telegram.get('bot_token'), telegram.get('chat_id'),
                                 f"🚨 jaegerbot {key}: Fehler ({phase})\n{e}")
                except Exception:
                    pass
    print(f"Lauf fertig: {len(jobs)} Strategien.")


if __name__ == '__main__':
    main()
