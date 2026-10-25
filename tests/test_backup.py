"""Quartals-Pipeline: Sicherung vor dem Lauf und Ruecksicherung bei Fehler stellen den alten Stand her."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import auto_optimizer_scheduler as sched


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)


def test_backup_and_restore(tmp_path, monkeypatch):
    root = str(tmp_path)
    monkeypatch.setattr(sched, 'PROJECT_ROOT', root)
    monkeypatch.setattr(sched, 'BACKUP_ROOT', os.path.join(root, 'artifacts', 'backups'))
    model = os.path.join(root, 'artifacts', 'models', '4h', 'ensemble_2026-10-01.joblib')
    cfg = os.path.join(root, 'src', 'jaegerbot', 'strategy', 'configs', 'config_BTCUSDTUSDT_4h.json')
    settings = os.path.join(root, 'settings.json')
    _write(model, 'alt-modell'); _write(cfg, 'alt-config'); _write(settings, 'alt-team')

    backup = sched.make_backup()

    # Pipeline "beschaedigt" den Stand: neues Modell, geaenderte Config, leeres Team
    _write(model, 'kaputt'); _write(cfg, 'kaputt'); _write(settings, 'kaputt')
    _write(os.path.join(root, 'artifacts', 'models', '4h', 'ensemble_2027-01-01.joblib'), 'neu')

    sched.restore_backup(backup)
    assert open(model, encoding='utf-8').read() == 'alt-modell'
    assert open(cfg, encoding='utf-8').read() == 'alt-config'
    assert open(settings, encoding='utf-8').read() == 'alt-team'
    # nach der Ruecksicherung gibt es das fehlerhafte neue Quartalsmodell nicht mehr
    assert not os.path.exists(os.path.join(root, 'artifacts', 'models', '4h', 'ensemble_2027-01-01.joblib'))
