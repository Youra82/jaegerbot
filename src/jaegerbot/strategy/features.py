# src/jaegerbot/strategy/features.py
"""ANN-Features je Kerze -- eine Funktion fuer Training, Backtest und Live.

Nur Rolling-Fenster und schnell abklingende Indikatoren: live (1000 Kerzen) ergibt dieselben
Werte wie der Backtest auf Jahren. OBV fehlt bewusst (kumulativ ueber die ganze Historie --
im alten jaegerbot sah das Modell live dadurch andere Eingaben als im Training).
"""
import numpy as np
import pandas as pd
import ta

from jaegerbot.strategy import trade_rules as tr
from jaegerbot.strategy.human_geometry import human_geometry, NAMES as human_geometry_names
from jaegerbot.strategy.candle_patterns import candle_patterns, NAMES as candle_pattern_names

FEATURES = [
    'bb_width', 'bb_pband', 'rsi', 'macd_diff', 'macd', 'atr_normalized', 'adx', 'adx_pos', 'adx_neg',
    'volume_ratio', 'mfi', 'cmf', 'price_to_ema20', 'price_to_ema50', 'stoch_k', 'stoch_d', 'williams_r',
    'roc', 'cci', 'price_to_resistance', 'price_to_support', 'high_low_range', 'close_to_high',
    'close_to_low', 'day_of_week', 'hour_of_day', 'returns_lag1', 'returns_lag2', 'returns_lag3',
    'hist_volatility', 'body_to_atr', 'upper_wick_ratio', 'lower_wick_ratio', 'candle_direction',
    'body_midpoint_ratio', 'bull_streak', 'bear_streak', 'dist_to_struct_high', 'dist_to_struct_low',
    'price_in_range_20', 'price_in_range_50', 'fib_position', 'in_fib_zone', 'volume_direction',
    'buying_pressure', 'selling_pressure',
]

_G = tr.GEO_CFG
GEO_WINDOWS, HTF_WINDOWS = tuple(_G['windows']), tuple(_G['htf_windows'])
_GEO_KEYS = ('drift', 'er', 'pos', 'width', 'tu', 'tl')
GEO_COLUMNS = [f"geo_{k}_{n}" for n in GEO_WINDOWS for k in _GEO_KEYS]
HTF_COLUMNS = [f"htf_geo_{k}_{n}" for n in HTF_WINDOWS for k in _GEO_KEYS]
HUMAN_COLUMNS = ['hg_' + x for x in human_geometry_names] + ['hgd_' + x for x in human_geometry_names]
CANDLE_COLUMNS = ['cp_' + x for x in candle_pattern_names] + ['cpd_' + x for x in candle_pattern_names]
if tr.GEO_FEATURES:            # Profil 'geo' (forschung/ann_kanaltypen): Kanal-Geometrie als Eingaben
    FEATURES = FEATURES + GEO_COLUMNS
if tr.HTF_FEATURES:            # Profil 'geohtf': dieselbe Geometrie auf dem uebergeordneten Timeframe
    FEATURES = FEATURES + HTF_COLUMNS
if _G['human']:                # Variante 'Mensch': Struktur, Trendlinien, Zonen, Ausbruch/Retest (4h + Tag)
    FEATURES = FEATURES + HUMAN_COLUMNS
if _G['candles']:              # Kerzenformen-Bibliothek (4h + Tag)
    FEATURES = FEATURES + CANDLE_COLUMNS

# Kerzen Vorlauf, bis alle Features eingeschwungen sind (EMA50/ADX/MACD klingen nach ~4x Fenster ab);
# mit uebergeordneten Kanaelen entsprechend mehr (Profil-Einstellung warmup)
WARMUP_CANDLES = int(_G['warmup']) if tr.HTF_FEATURES else 300


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Erwartet open/high/low/close/volume mit UTC-Zeitindex; liefert FEATURES + atr (+ Rohspalten)."""
    df = df.copy()
    o, h, l, c, v = df['open'], df['high'], df['low'], df['close'], df['volume']

    bb = ta.volatility.BollingerBands(close=c, window=20, window_dev=2)
    df['bb_width'] = bb.bollinger_wband()
    df['bb_pband'] = bb.bollinger_pband()

    vol_sma = v.rolling(20).mean()
    df['volume_ratio'] = (v / vol_sma).replace([np.inf, -np.inf], np.nan)
    df['mfi'] = ta.volume.money_flow_index(h, l, c, v, window=14)
    df['cmf'] = ta.volume.chaikin_money_flow(h, l, c, v, window=20)

    df['rsi'] = ta.momentum.rsi(c, window=14)
    macd = ta.trend.MACD(c, window_slow=26, window_fast=12, window_sign=9)
    # MACD in Preis-Einheiten waere nicht zwischen Coins vergleichbar -> relativ zum Kurs
    df['macd_diff'] = macd.macd_diff() / c
    df['macd'] = macd.macd() / c

    df['atr'] = ta.volatility.AverageTrueRange(high=h, low=l, close=c, window=14).average_true_range()
    df['atr_normalized'] = df['atr'] / c * 100

    df['adx'] = ta.trend.adx(h, l, c, window=14)
    df['adx_pos'] = ta.trend.adx_pos(h, l, c, window=14)
    df['adx_neg'] = ta.trend.adx_neg(h, l, c, window=14)

    df['ema20'] = ta.trend.ema_indicator(c, window=20)
    df['ema50'] = ta.trend.ema_indicator(c, window=50)
    df['price_to_ema20'] = (c - df['ema20']) / df['ema20']
    df['price_to_ema50'] = (c - df['ema50']) / df['ema50']

    df['stoch_k'] = ta.momentum.stoch(h, l, c, window=14, smooth_window=3)
    df['stoch_d'] = ta.momentum.stoch_signal(h, l, c, window=14, smooth_window=3)
    df['williams_r'] = ta.momentum.williams_r(h, l, c, lbp=14)
    df['roc'] = ta.momentum.roc(c, window=12)
    df['cci'] = ta.trend.cci(h, l, c, window=20)

    res, sup = h.rolling(20).max(), l.rolling(20).min()
    df['price_to_resistance'] = (res - c) / c
    df['price_to_support'] = (c - sup) / c

    rng = h - l
    df['high_low_range'] = rng / c
    df['close_to_high'] = (h - c) / (rng + 1e-12)
    df['close_to_low'] = (c - l) / (rng + 1e-12)

    df['day_of_week'] = df.index.dayofweek
    df['hour_of_day'] = df.index.hour

    ret = c.pct_change()
    df['returns_lag1'] = ret.shift(1)
    df['returns_lag2'] = ret.shift(2)
    df['returns_lag3'] = ret.shift(3)
    df['hist_volatility'] = ret.rolling(20).std() * np.sqrt(252)

    body = (c - o).abs()
    df['body_to_atr'] = body / (df['atr'] + 1e-12)
    df['upper_wick_ratio'] = (h - np.maximum(c, o)) / (rng + 1e-12)
    df['lower_wick_ratio'] = (np.minimum(c, o) - l) / (rng + 1e-12)
    df['candle_direction'] = np.sign(c - o)
    df['body_midpoint_ratio'] = ((c + o) / 2 - l) / (rng + 1e-12)
    df['bull_streak'] = (df['candle_direction'] == 1).rolling(5).sum()
    df['bear_streak'] = (df['candle_direction'] == -1).rolling(5).sum()

    piv_h, piv_l = h.rolling(20).max(), l.rolling(20).min()
    df['dist_to_struct_high'] = (piv_h - c) / (df['atr'] + 1e-12)
    df['dist_to_struct_low'] = (c - piv_l) / (df['atr'] + 1e-12)
    df['price_in_range_20'] = (c - l.rolling(20).min()) / (h.rolling(20).max() - l.rolling(20).min() + 1e-12)
    df['price_in_range_50'] = (c - l.rolling(50).min()) / (h.rolling(50).max() - l.rolling(50).min() + 1e-12)

    rh10, rl10 = h.shift(1).rolling(10).max(), l.shift(1).rolling(10).min()
    rr10 = rh10 - rl10
    df['fib_position'] = (c - rl10) / (rr10 + 1e-12)
    df['in_fib_zone'] = ((c >= rl10 + rr10 * 0.382) & (c <= rl10 + rr10 * 0.618)).astype(float)

    df['volume_direction'] = df['candle_direction'] * df['volume_ratio']
    df['buying_pressure'] = df['close_to_low'] * df['volume_ratio']
    df['selling_pressure'] = df['close_to_high'] * df['volume_ratio']
    ha, la, ca, atr_a = h.to_numpy(float), l.to_numpy(float), c.to_numpy(float), df['atr'].to_numpy(float)
    extra = {}
    if tr.HTF_FEATURES:
        extra.update(higher_tf_features(df))
    if tr.GEO_FEATURES:
        extra.update(channel_geometry(ha, la, ca, atr_a, GEO_WINDOWS, _G['edge'], 'geo_'))
    if _G['human']:
        extra.update(human_geometry(ha, la, ca, atr_a, k=5, lookback=300, prefix='hg_'))
    if _G['candles']:
        extra.update(candle_patterns(o.to_numpy(float), ha, la, ca, atr_a, 'cp_'))
    if extra:
        df = pd.concat([df, pd.DataFrame(extra, index=df.index)], axis=1)
    return df


def higher_tf_features(df):
    """Merkmale des uebergeordneten Timeframes (Tag oder Woche, aus den Kerzen von df gebildet). Eine Kerze sieht nur
    Perioden, die zu ihrem Schluss abgeschlossen sind; angeschnittene erste Periode wird verworfen.
    ATR = einfacher 14-Perioden-Mittelwert der True Range (exakt gleich live/Backtest)."""
    rule = _G['htf']
    period = pd.Timedelta('7D') if rule == '1W' else pd.Timedelta('1D')
    norm = df.index.normalize()
    key = norm - pd.to_timedelta(df.index.dayofweek, unit='D') if rule == '1W' else norm
    g = df[['open', 'high', 'low', 'close']].groupby(key)
    d = pd.DataFrame({'open': g['open'].first(), 'high': g['high'].max(), 'low': g['low'].min(),
                      'close': g['close'].last()})
    if df.index[0] > d.index[0]:
        d = d.iloc[1:]                                  # erste Periode angeschnitten
    prev = d['close'].shift(1)
    trange = pd.concat([d['high'] - d['low'], (d['high'] - prev).abs(), (d['low'] - prev).abs()], axis=1).max(axis=1)
    atr_d = trange.rolling(14).mean().to_numpy(float)
    hd, ld, cd, od = (d[x].to_numpy(float) for x in ('high', 'low', 'close', 'open'))
    vals = channel_geometry(hd, ld, cd, atr_d, HTF_WINDOWS, _G['edge'], 'htf_geo_')
    if _G['human']:
        vals.update(human_geometry(hd, ld, cd, atr_d, k=3, lookback=150, prefix='hgd_'))
    if _G['candles']:
        vals.update(candle_patterns(od, hd, ld, cd, atr_d, 'cpd_'))
    per = pd.DataFrame(vals, index=d.index + period)    # gueltig ab Periodenende
    tf = df.index[1] - df.index[0] if len(df) > 1 else pd.Timedelta('4h')
    close_time = pd.DataFrame({'t': df.index + tf})
    m = pd.merge_asof(close_time, per.rename_axis('t').reset_index(), on='t', direction='backward')
    return {col: m[col].to_numpy() for col in per.columns}


def _touches(mask):
    n, last = 0, -10
    for k in np.flatnonzero(mask):
        if k - last >= 3:
            n += 1
            last = k
    return n


def channel_geometry(h, l, c, atr, windows, edge, prefix):
    """Reine Kursgeometrie je Kerze i aus den Kerzen i-N+1..i (nur Vergangenheit): Regressions-Mittellinie,
    Huelle der Hochs/Tiefs als Kanal, Drift/Breite, Geradlinigkeit, Lage im Kanal, Breite/ATR, Beruehrungen."""
    n = len(c)
    out = {f"{prefix}{k}_{N}": np.full(n, np.nan) for N in windows for k in _GEO_KEYS}
    for N in windows:
        x = np.arange(N, dtype=float) - (N - 1) / 2
        sxx = (x ** 2).sum()
        for i in range(N - 1, n):
            s = slice(i - N + 1, i + 1)
            cc = c[s]
            m = (x * (cc - cc.mean())).sum() / sxx
            center = cc.mean() + m * x
            up = center + (h[s] - center).max()
            lo = center + (l[s] - center).min()
            W = up[-1] - lo[-1]
            if W <= 0 or not np.isfinite(atr[i]) or atr[i] <= 0:
                continue
            path = np.abs(np.diff(cc)).sum()
            out[f"{prefix}drift_{N}"][i] = m * N / W
            out[f"{prefix}er_{N}"][i] = abs(cc[-1] - cc[0]) / path if path > 0 else 0.0
            out[f"{prefix}pos_{N}"][i] = (cc[-1] - lo[-1]) / W
            out[f"{prefix}width_{N}"][i] = W / atr[i]
            out[f"{prefix}tu_{N}"][i] = _touches(h[s] >= up - edge * W)
            out[f"{prefix}tl_{N}"][i] = _touches(l[s] <= lo + edge * W)
    return out
