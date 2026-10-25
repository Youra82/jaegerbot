import numpy as np
import pandas as pd

from jaegerbot.strategy import trade_rules as tr
from jaegerbot.utils.data_check import check_candles, gaps


def _row(pL, pS, ema20=2.0, ema50=1.0, q=0.95, thr=0.6):
    return {'pL': pL, 'pS': pS, 'ema20': ema20, 'ema50': ema50,
            tr.thr_col('L', q): thr, tr.thr_col('S', q): thr}


def test_entry_signal_sides_and_trend_filter():
    cfg = {'sides': 'L', 'quantile': 0.95, 'ema_filter': True}
    assert tr.entry_signal(_row(0.7, 0.1), cfg) == 'long'
    assert tr.entry_signal(_row(0.7, 0.1, ema20=1.0, ema50=2.0), cfg) is None   # Long nur im Aufwaertstrend
    assert tr.entry_signal(_row(0.1, 0.9), cfg) is None                         # Short nicht erlaubt
    cfg_ls = {'sides': 'LS', 'quantile': 0.95, 'ema_filter': False}
    assert tr.entry_signal(_row(0.65, 0.9), cfg_ls) == 'short'                  # staerkerer Ueberschuss gewinnt


def test_levels_risk_reward():
    sl, tp = tr.levels(100.0, 1.0, 'long')
    assert sl == 100 - tr.SL_ATR_MULT and tp == 100 + tr.SL_ATR_MULT * tr.RISK_REWARD
    sl, tp = tr.levels(100.0, 1.0, 'short')
    assert sl == 100 + tr.SL_ATR_MULT and tp == 100 - tr.SL_ATR_MULT * tr.RISK_REWARD


def test_simulate_exit_sl_first_gap_and_time():
    o = np.array([100, 100, 100, 100.0]); c = o.copy()
    h = np.array([100, 105, 100, 100.0]); l = np.array([100, 97, 100, 100.0])
    # SL und TP in derselben Kerze -> SL (pessimistisch)
    assert tr.simulate_exit(o, h, l, c, 0, 'long', 98, 104, 3) == (1, 98, 'sl')
    # Eroeffnung schon unter SL -> Fuellung zur Eroeffnung
    o2 = np.array([100, 95, 100, 100.0])
    assert tr.simulate_exit(o2, np.maximum(o2, h), np.minimum(o2, l), c, 0, 'long', 98, 104, 3) == (1, 95, 'sl')
    # nichts getroffen -> Zeitausstieg zum Schluss der Kerze i+hold
    flat = np.full(5, 100.0)
    assert tr.simulate_exit(flat, flat, flat, flat, 0, 'long', 98, 104, 3) == (3, 100.0, 'time')
    # Daten enden vor dem Ausstieg -> offen
    assert tr.simulate_exit(flat[:3], flat[:3], flat[:3], flat[:3], 0, 'long', 98, 104, 3) is None


def test_position_size_and_leverage():
    qty = tr.position_size(1000, 1.0, 100.0, 98.0)            # 10 USDT Risiko / 2 Abstand
    assert abs(qty - 5.0) < 1e-12
    assert tr.position_size(8, 1.0, 100.0, 98.0) is None      # 4 USDT < 5 USDT Mindestgroesse
    lev = tr.leverage_for(100.0, 96.0, 20)                    # SL 4 % -> floor(0.8/0.04) = 20
    assert lev == 20 and tr.leverage_for(100.0, 90.0, 20) == 8


def test_trade_pnl_includes_costs():
    pnl, _ = tr.trade_pnl('long', 1.0, 100.0, 100.0)
    assert pnl < 0                                              # Gebuehren + Schlupf bei Ausstieg zum Einstiegskurs


def test_check_candles_detects_gap_and_staleness():
    idx = pd.date_range('2026-01-01', periods=10, freq='4h', tz='UTC')
    df = pd.DataFrame({'open': 1.0, 'high': 1.0, 'low': 1.0, 'close': 1.0, 'volume': 1.0}, index=idx)
    assert check_candles(df, '4h') == []
    holed = df.drop(idx[4])
    assert gaps(holed, '4h')[0][1] == 1 and check_candles(holed, '4h')
    now = idx[-1] + pd.Timedelta(days=2)
    assert any('veraltet' in p for p in check_candles(df, '4h', max_age=pd.Timedelta(hours=5), now=now))
