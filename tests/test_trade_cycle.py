"""Live-Code (trade_manager.full_trade_cycle) gegen eine Schein-Boerse auf echten Kerzen:
Einstieg genau wie das Backtest-Signal, Ausstieg ueber SL/TP (Boerse) und ueber Zeitablauf."""
import os
import glob
import json
import logging
import pandas as pd
import pytest

from jaegerbot.analysis.backtester import signal_events
from jaegerbot.strategy import trade_rules as tr
from jaegerbot.utils import trade_manager as tm
from jaegerbot.utils.data import load_data

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LOG = logging.getLogger('test')


class FakeExchange:
    def __init__(self, candles, equity=1000.0):
        self.candles, self.equity = candles, equity
        self.until = None
        self.positions, self.orders, self.closed = [], [], []

    def fetch_recent_ohlcv(self, symbol, timeframe, limit):
        return self.candles[self.candles.index <= self.until].tail(limit)

    def fetch_open_positions_strict(self, symbol=None):
        return [p for p in self.positions if symbol is None or p['symbol'] == symbol]

    def fetch_realized_equity_usdt(self):
        return self.equity

    def fetch_ticker(self, symbol):
        return {'last': float(self.candles.loc[self.until, 'close'])}

    def set_margin_mode(self, *a, **k):
        pass

    def set_leverage(self, *a, **k):
        pass

    def place_market_order_with_tpsl(self, symbol, side, amount, sl, tp):
        self.orders.append({'symbol': symbol, 'side': side, 'amount': amount, 'sl': sl, 'tp': tp})
        self.positions.append({'symbol': symbol, 'side': 'long' if side == 'buy' else 'short',
                               'contracts': amount, 'initialMargin': 0.0})
        return {'id': 'x1'}

    def cancel_all_orders_for_symbol(self, symbol):
        return 0

    def close_position_market(self, symbol, side, amount):
        self.closed.append((symbol, side, amount))
        self.positions = [p for p in self.positions if p['symbol'] != symbol]


def _active_config_with_signal():
    cdir = tr.profile_dir(os.path.join(PROJECT_ROOT, 'src', 'jaegerbot', 'strategy', 'configs'))   # Configs des aktiven Profils
    for path in sorted(glob.glob(os.path.join(cdir, 'config_*.json'))):
        with open(path, encoding='utf-8') as f:
            c = json.load(f)
        coin, tf = c['market']['symbol'].split('/')[0], c['market']['timeframe']
        ev = [e for e in signal_events(coin, tf, c['strategy']) if e['reason'] in ('sl', 'tp', 'time')]
        if ev:
            return c, ev[-1]
    return None, None


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(tm, 'STATE_DIR', str(tmp_path))
    config, event = _active_config_with_signal()
    if config is None:
        pytest.skip('keine lokalen Daten/Vorhersagen (erst Pipeline laufen lassen)')
    raw = load_data(config['market']['symbol'], config['market']['timeframe'])
    return config, event, raw


def test_entry_matches_backtest_and_exit(setup):
    config, ev, raw = setup
    tf = config['market']['timeframe']
    live = {'risk_per_trade_pct': 1.0, 'max_open_positions': 10, 'max_leverage': 20}
    ex = FakeExchange(raw)
    signal_ts = ev['time'] - pd.Timedelta(tf)
    ex.until = signal_ts
    tm.full_trade_cycle(ex, config, live, {}, LOG, now=ev['time'] + pd.Timedelta(minutes=2))
    assert len(ex.orders) == 1
    o = ex.orders[0]
    assert o['side'] == ('buy' if ev['side'] == 'long' else 'sell')
    assert abs(o['sl'] - ev['sl']) < 1e-12 and abs(o['tp'] - ev['tp']) < 1e-12
    expected = tr.position_size(1000.0, 1.0, tr.fill_price(ev['signal_close'], ev['side'], True), ev['sl'])
    assert abs(o['amount'] - expected) < 1e-9

    # gleiche Kerze erneut: kein zweiter Einstieg
    tm.full_trade_cycle(ex, config, live, {}, LOG, now=ev['time'] + pd.Timedelta(minutes=17))
    assert len(ex.orders) == 1

    exit_ts = ev['exit_time'] - pd.Timedelta(tf)
    if ev['reason'] == 'time':
        ex.until = exit_ts
        tm.full_trade_cycle(ex, config, live, {}, LOG, now=ev['exit_time'] + pd.Timedelta(minutes=2), manage_only=True)
        assert ex.closed and not ex.positions                     # Zeitausstieg per Market
    else:
        ex.positions.clear()                                       # Boerse hat SL/TP ausgefuehrt
        ex.until = exit_ts
        tm.full_trade_cycle(ex, config, live, {}, LOG, now=ev['exit_time'] + pd.Timedelta(minutes=2), manage_only=True)
        assert not ex.closed
    assert tm.load_state(tm.strategy_key(config['market']['symbol'], tf)).get('position') is None


def test_old_signal_not_traded(setup):
    config, ev, raw = setup
    tf = config['market']['timeframe']
    ex = FakeExchange(raw)
    ex.until = ev['time'] - pd.Timedelta(tf)
    tm.full_trade_cycle(ex, config, {'risk_per_trade_pct': 1.0}, {}, LOG, now=ev['time'] + pd.Timedelta(minutes=45))
    assert not ex.orders


def test_symbol_busy_and_max_positions(setup):
    config, ev, raw = setup
    tf = config['market']['timeframe']
    ex = FakeExchange(raw)
    ex.until = ev['time'] - pd.Timedelta(tf)
    ex.positions = [{'symbol': config['market']['symbol'], 'side': 'long', 'contracts': 1, 'initialMargin': 1}]
    tm.full_trade_cycle(ex, config, {'risk_per_trade_pct': 1.0, 'max_open_positions': 10}, {}, LOG,
                        now=ev['time'] + pd.Timedelta(minutes=2))
    assert not ex.orders
