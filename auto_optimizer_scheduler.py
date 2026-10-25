#!/usr/bin/env python3
"""Quartals-Pipeline (vom master_runner im Hintergrund gestartet, prueft selbst ob faellig).

Faellig, sobald fuer den aktuellen Quartalsbeginn (ab dem 2. Tag) in einem Timeframe noch kein
Modell existiert. Ablauf: Kerzen nachladen -> neues Quartalsmodell trainieren + Vorhersagen ->
Configs neu bestaetigen -> Portfolio + Risiko in settings.json -> Telegram.

Keine woechentliche Neuwahl: im Test vom 25.10.2026 senkte die 26W-Wochenauswahl die Rendite meist.

Aufruf: python auto_optimizer_scheduler.py [--force]
"""
import os
import sys
import json
import time
import argparse
import shutil
import subprocess
from datetime import datetime

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from jaegerbot.utils.data import load_settings
from jaegerbot.utils.telegram import send_message
from jaegerbot.strategy import model as mdl

LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
LOCK_FILE = os.path.join(PROJECT_ROOT, 'artifacts', 'db', '.optimization_in_progress')
LOCK_MAX_AGE_H = 12


def _log(msg):
    os.makedirs(LOG_DIR, exist_ok=True)
    line = f"{datetime.now().isoformat(timespec='seconds')} AUTO-OPTIMIZER {msg}"
    with open(os.path.join(LOG_DIR, 'auto_optimizer.log'), 'a', encoding='utf-8') as f:
        f.write(line + '\n')
    print(line, flush=True)


def _telegram(text):
    try:
        with open(os.path.join(PROJECT_ROOT, 'secret.json'), 'r', encoding='utf-8') as f:
            t = json.load(f).get('telegram', {})
        if t.get('bot_token') and t.get('chat_id'):
            send_message(t['bot_token'], t['chat_id'], text)
    except Exception as e:
        _log(f"Telegram fehlgeschlagen: {e}")


def live_profiles(settings):
    """Profile des Live-Teams mit ihren Timeframes; 'trend' zuerst."""
    out = {}
    for x in settings['live_trading_settings'].get('active_strategies', []):
        if x.get('active', True):
            out.setdefault(x.get('profile', 'trend'), set()).add(x['timeframe'])
    return dict(sorted(out.items(), key=lambda kv: (kv[0] != 'trend', kv[0])))


def _model_file(profile, tf, cutoff):
    base = os.path.join(PROJECT_ROOT, 'artifacts', 'models' + ('' if profile == 'trend' else f'_{profile}'))
    return os.path.join(base, tf, f"ensemble_{cutoff:%Y-%m-%d}.joblib")


def is_due(settings):
    now = pd.Timestamp.now(tz='UTC')
    q = mdl.quarter_starts(now, now)[0]
    if now < q + pd.Timedelta(days=1):
        return False
    profiles = live_profiles(settings)
    tfs = {'trend': set(settings['research']['timeframes'])} if list(profiles) == ['trend'] else profiles
    return any(not os.path.exists(_model_file(pr, tf, q)) for pr, ts in tfs.items() for tf in ts)


def run(cmd, name, profile='trend'):
    _log(f"START {name} [{profile}]: {' '.join(cmd)}")
    t = time.time()
    with open(os.path.join(LOG_DIR, f"pipeline_{name}.log"), 'a', encoding='utf-8') as out:
        rc = subprocess.call(cmd, cwd=PROJECT_ROOT, stdout=out, stderr=subprocess.STDOUT,
                             env={**os.environ, 'PYTHONPATH': os.path.join(PROJECT_ROOT, 'src'),
                                  'JAEGER_PROFILE': profile})
    _log(f"ENDE {name}: rc={rc} ({time.time() - t:.0f}s)")
    if rc != 0:
        raise RuntimeError(f"{name} fehlgeschlagen (rc={rc}), siehe logs/pipeline_{name}.log")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--force', action='store_true')
    a = p.parse_args()
    s = load_settings()
    if not a.force and not is_due(s):
        return
    if os.path.exists(LOCK_FILE) and time.time() - os.path.getmtime(LOCK_FILE) < LOCK_MAX_AGE_H * 3600:
        _log("laeuft bereits -- uebersprungen")
        return
    os.makedirs(os.path.dirname(LOCK_FILE), exist_ok=True)
    open(LOCK_FILE, 'w').close()
    t0 = time.time()
    py = sys.executable
    backup = make_backup()
    _log(f"Sicherung: {backup}")
    profiles = live_profiles(s)
    if list(profiles) != ['trend']:
        multi_profile_pipeline(s, profiles, backup, t0, py)
        return
    try:
        _telegram("jaegerbot: Quartals-Pipeline gestartet (Daten, Modell, Configs, Portfolio). Sicherung angelegt.")
        run([py, 'download_data.py'], 'download')
        run([py, '-m', 'jaegerbot.analysis.walkforward', '--only-new'], 'walkforward')
        check_models(s)
        run([py, '-m', 'jaegerbot.analysis.optimizer'], 'optimizer')
        run([py, 'run_portfolio_optimizer.py', '--write'], 'portfolio')
        if not load_settings()['live_trading_settings'].get('active_strategies'):
            raise RuntimeError("Neues Team ist leer")
        run([py, os.path.join('tests', 'parity_check.py')], 'parity')
        with open(os.path.join(LOG_DIR, 'pipeline_portfolio.log'), 'r', encoding='utf-8') as f:
            tail = ''.join(f.readlines()[-3:])
        _telegram(f"jaegerbot: Quartals-Pipeline fertig + Selbstpruefung bestanden "
                  f"({(time.time() - t0) / 60:.0f} min).\n{tail}")
    except Exception as e:
        _log(f"FEHLER: {e} -- spiele Sicherung {backup} zurueck")
        restore_backup(backup)
        _telegram(f"🚨 jaegerbot: Quartals-Pipeline fehlgeschlagen ({e}). Alter Stand (Modelle, Configs, Team) "
                  f"wurde zurueckgespielt; der Bot handelt weiter mit dem Vorquartal-Modell.")
    finally:
        os.remove(LOCK_FILE)


def multi_profile_pipeline(s, profiles, backup, t0, py):
    """Kombi-Team (z. B. Long 'trend' + Short 'mensch'): je Profil neues Quartalsmodell + Vorhersagen, Modell- und
    Paritaetspruefung. Regeln, Team und Risiko bleiben FEST -- die Kombination wurde unter Vorregistrierung gewaehlt
    (forschung/kanal_varianten) und soll nicht still automatisch umgestellt werden."""
    try:
        _telegram(f"jaegerbot: Quartals-Pipeline (Kombi-Team {', '.join(profiles)}) gestartet. Sicherung angelegt.")
        run([py, 'download_data.py', '--timeframes', *sorted({tf for ts in profiles.values() for tf in ts})], 'download')
        for prof, tfs in profiles.items():
            run([py, '-m', 'jaegerbot.analysis.walkforward', '--only-new', '--timeframes', *sorted(tfs)],
                f'walkforward_{prof}', prof)
            run([py, os.path.abspath(__file__), '--check-models', prof, *sorted(tfs)], f'check_{prof}', prof)
            run([py, os.path.join('tests', 'parity_check.py'), '--candles', '30'], f'parity_{prof}', prof)
        _telegram(f"jaegerbot: Quartals-Pipeline fertig ({(time.time() - t0) / 60:.0f} min). Neue Quartalsmodelle "
                  f"fuer {', '.join(profiles)} aktiv, Paritaet bestanden. Regeln/Team/Risiko unveraendert -- "
                  f"Neubewertung der Kombination von Hand empfohlen.")
    except Exception as e:
        _log(f"FEHLER: {e} -- spiele Sicherung {backup} zurueck")
        restore_backup(backup)
        _telegram(f"🚨 jaegerbot: Quartals-Pipeline fehlgeschlagen ({e}). Alter Stand zurueckgespielt; "
                  f"der Bot handelt weiter mit den Vorquartal-Modellen.")
    finally:
        os.remove(LOCK_FILE)


BACKUP_ROOT = os.path.join(PROJECT_ROOT, 'artifacts', 'backups')
BACKUP_ITEMS = [os.path.join('artifacts', 'models'), os.path.join('artifacts', 'models_mensch'),
                os.path.join('src', 'jaegerbot', 'strategy', 'configs'),
                os.path.join('src', 'jaegerbot', 'strategy', 'configs_mensch'), 'settings.json']
KEEP_BACKUPS = 6


def make_backup():
    dst = os.path.join(BACKUP_ROOT, datetime.now().strftime('%Y-%m-%d_%H%M%S'))
    for item in BACKUP_ITEMS:
        src = os.path.join(PROJECT_ROOT, item)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(dst, item))
        elif os.path.exists(src):
            os.makedirs(os.path.dirname(os.path.join(dst, item)) or dst, exist_ok=True)
            shutil.copy2(src, os.path.join(dst, item))
    old = sorted(os.listdir(BACKUP_ROOT))[:-KEEP_BACKUPS]
    for name in old:
        shutil.rmtree(os.path.join(BACKUP_ROOT, name), ignore_errors=True)
    return dst


def restore_backup(src):
    for item in BACKUP_ITEMS:
        b, live = os.path.join(src, item), os.path.join(PROJECT_ROOT, item)
        if os.path.isdir(b):
            shutil.rmtree(live, ignore_errors=True)
            shutil.copytree(b, live)
        elif os.path.exists(b):
            shutil.copy2(b, live)


def check_models(settings, timeframes=None):
    """Neue Quartalsmodelle plausibel? Schwellen geordnet in (0,1), Vorhersagen nicht konstant."""
    now = pd.Timestamp.now(tz='UTC')
    q = mdl.quarter_starts(now, now)[0]
    for tf in (timeframes or settings['research']['timeframes']):
        if q not in mdl.available_cutoffs(tf):
            raise RuntimeError(f"Modell {tf} {q:%Y-%m-%d} fehlt nach dem Training")
        ens = mdl.load(tf, q)
        for side in ('L', 'S'):
            thr = [ens.quantiles[f"thr{side}_{int(round(x * 1000))}"] for x in (0.9, 0.95, 0.975, 0.99)]
            if not (0 < thr[0] < thr[-1] < 1) or thr != sorted(thr):
                raise RuntimeError(f"Modell {tf} {side}: Schwellen unplausibel {thr}")
            if ens.n_train[side] < 10000:
                raise RuntimeError(f"Modell {tf} {side}: nur {ens.n_train[side]} Trainingslabels")


if __name__ == '__main__':
    if len(sys.argv) >= 3 and sys.argv[1] == '--check-models':     # Unterprozess mit JAEGER_PROFILE
        from jaegerbot.strategy import trade_rules as _tr
        assert _tr.PROFILE == sys.argv[2], f"Profil {_tr.PROFILE} != {sys.argv[2]}"
        check_models(load_settings(), sys.argv[3:])
        print(f"Modelle {sys.argv[2]} {sys.argv[3:]} in Ordnung")
    else:
        main()
