# master_runner.py
"""JaegerBot v5 Master Runner (Cron alle 15 Min, mit flock).

Strategien koennen zu verschiedenen Profilen gehoeren (settings: active_strategies[].profile, Standard 'trend';
'geohtf' = Short-Regel mit Tageskanaelen). Ein Profil = eigenes Modell/Features -> eigener Unterprozess.
Reihenfolge wie im Kombi-Backtest: Durchgang 1 Ausstiege ALLER Profile, Durchgang 2 Einstiege, Profil 'trend'
(Long) vor den anderen, innerhalb eines Profils nach Strategie-Schluessel. Ein Coin ist nur einmal belegt
(trade_manager prueft offene Positionen des ganzen Kontos).
"""
import os
import sys
import json
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))


def start_background(script, log_name):
    path = os.path.join(PROJECT_ROOT, script)
    if not os.path.exists(path):
        return
    os.makedirs(os.path.join(PROJECT_ROOT, 'logs'), exist_ok=True)
    subprocess.Popen([sys.executable, path], stdout=open(os.path.join(PROJECT_ROOT, 'logs', log_name), 'a'),
                     stderr=subprocess.STDOUT)


def load_json(name):
    with open(os.path.join(PROJECT_ROOT, name), 'r', encoding='utf-8') as f:
        return json.load(f)


def profiles_in(strategies):
    names = {s.get('profile', 'trend') for s in strategies if s.get('active', True)}
    return sorted(names, key=lambda p: (p != 'trend', p))


def run_child(profile, phase):
    """Unterprozess: eine Phase fuer alle Strategien eines Profils (JAEGER_PROFILE ist gesetzt)."""
    from jaegerbot.strategy import trade_rules as tr
    from jaegerbot.utils.exchange import Exchange
    from jaegerbot.utils.telegram import send_message
    from jaegerbot.utils.trade_manager import full_trade_cycle, strategy_key
    from jaegerbot.strategy.run import setup_logging, load_config
    assert tr.PROFILE == profile, f"Profil {tr.PROFILE} != {profile}"
    live = load_json('settings.json')['live_trading_settings']
    secrets = load_json('secret.json')
    telegram = secrets.get('telegram', {})
    team = sorted((s for s in live.get('active_strategies', [])
                   if s.get('active', True) and s.get('profile', 'trend') == profile),
                  key=lambda s: strategy_key(s['symbol'], s['timeframe']))
    exchange = Exchange(secrets['jaegerbot'][0])
    if not exchange.markets:
        print(f"[{profile}] Fehler: Maerkte nicht geladen.")
        return
    for s in team:
        logger = setup_logging(s['symbol'], s['timeframe'])
        try:
            config = load_config(s['symbol'], s['timeframe'])
            full_trade_cycle(exchange, config, live, telegram, logger, manage_only=(phase == 'ausstiege'))
        except Exception as e:
            logger.critical(f"Fehler ({phase}): {e}", exc_info=True)
            try:
                send_message(telegram.get('bot_token'), telegram.get('chat_id'),
                             f"🚨 jaegerbot[{profile}] {s['symbol']} {s['timeframe']}: Fehler ({phase})\n{e}")
            except Exception:
                pass
    print(f"[{profile}] {phase}: {len(team)} Strategien")


def main():
    if len(sys.argv) == 4 and sys.argv[1] == '--child':
        run_child(sys.argv[2], sys.argv[3])
        return
    print("=======================================================")
    print("JaegerBot Master Runner v5 (ANN-Ensembles, Live = Backtest)")
    print("=======================================================")
    start_background('auto_optimizer_scheduler.py', 'auto_optimizer_trigger.log')
    start_background('daily_self_check.py', 'daily_self_check.log')
    secrets = load_json('secret.json')
    if not secrets.get('jaegerbot'):
        print("Fehler: Kein 'jaegerbot'-Account in secret.json.")
        return
    profiles = profiles_in(load_json('settings.json')['live_trading_settings'].get('active_strategies', []))
    if not profiles:
        print("Keine aktiven Strategien in settings.json.")
        return
    for phase in ('ausstiege', 'einstiege'):
        for profile in profiles:
            env = {**os.environ, 'JAEGER_PROFILE': profile, 'PYTHONPATH': os.path.join(PROJECT_ROOT, 'src')}
            subprocess.run([sys.executable, os.path.abspath(__file__), '--child', profile, phase],
                           cwd=PROJECT_ROOT, env=env)
    print(f"Lauf fertig: Profile {', '.join(profiles)}.")


if __name__ == '__main__':
    main()
