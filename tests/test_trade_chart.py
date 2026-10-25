"""Telegram-Chart je Trade: Eroeffnung und Schliessung senden ein PNG; scheitert das Bild, kommt trotzdem der Text."""
import os
import shutil
import logging
import pandas as pd
import pytest

from jaegerbot.utils import trade_manager as tm
from test_trade_cycle import FakeExchange, _active_config_with_signal
from jaegerbot.utils.data import load_data

LOG = logging.getLogger('test')
TG = {'bot_token': 'x', 'chat_id': 'y'}
KEEP = os.environ.get('JAEGER_KEEP_CHARTS')     # Ordner: erzeugte Bilder zum Ansehen behalten


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(tm, 'STATE_DIR', str(tmp_path))
    config, event = _active_config_with_signal()
    if config is None:
        pytest.skip('keine lokalen Daten/Vorhersagen')
    return config, event, load_data(config['market']['symbol'], config['market']['timeframe'])


def _capture(monkeypatch, sent, texts):
    def fake_photo(token, chat, path, caption=''):
        with open(path, 'rb') as f:
            head = f.read(8)
        sent.append((caption, os.path.getsize(path), head))
        if KEEP:
            os.makedirs(KEEP, exist_ok=True)
            shutil.copy(path, os.path.join(KEEP, f"{len(sent)}_{os.path.basename(path)}"))
        return True
    monkeypatch.setattr(tm, 'send_photo', fake_photo)
    monkeypatch.setattr(tm, 'send_message', lambda token, chat, text: texts.append(text) or True)


def test_chart_on_open_and_close(setup, monkeypatch):
    config, ev, raw = setup
    tf = config['market']['timeframe']
    sent, texts = [], []
    _capture(monkeypatch, sent, texts)
    live = {'risk_per_trade_pct': 1.0, 'max_open_positions': 10, 'max_leverage': 20}
    ex = FakeExchange(raw)
    ex.until = ev['time'] - pd.Timedelta(tf)
    tm.full_trade_cycle(ex, config, live, TG, LOG, now=ev['time'] + pd.Timedelta(minutes=2))
    assert len(sent) == 1 and 'eröffnet' in sent[0][0]
    assert sent[0][2].startswith(b'\x89PNG') and sent[0][1] > 20_000
    # Schliessung (SL/TP an der Boerse oder Zeitausstieg)
    ex.until = ev['exit_time'] - pd.Timedelta(tf)
    if ev['reason'] != 'time':
        ex.positions.clear()
    tm.full_trade_cycle(ex, config, live, TG, LOG, now=ev['exit_time'] + pd.Timedelta(minutes=2), manage_only=True)
    assert len(sent) == 2 and 'geschlossen' in sent[1][0]
    assert sent[1][2].startswith(b'\x89PNG')
    assert not texts                                   # kein Rueckfall auf Text noetig


def test_chart_failure_falls_back_to_text(setup, monkeypatch):
    config, ev, raw = setup
    tf = config['market']['timeframe']
    sent, texts = [], []
    _capture(monkeypatch, sent, texts)
    import jaegerbot.utils.trade_chart as tc
    monkeypatch.setattr(tc, 'render_trade_chart', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('kaputt')))
    ex = FakeExchange(raw)
    ex.until = ev['time'] - pd.Timedelta(tf)
    tm.full_trade_cycle(ex, config, {'risk_per_trade_pct': 1.0, 'max_open_positions': 10}, TG, LOG,
                        now=ev['time'] + pd.Timedelta(minutes=2))
    assert len(ex.orders) == 1                         # Order trotzdem platziert
    assert not sent and len(texts) == 1 and 'eröffnet' in texts[0]
